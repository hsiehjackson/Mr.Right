import pdb
import utils
import json
import pickle
import torch
import os
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from metric import score
from scheduler import CosineLRScheduler
from tqdm import tqdm

class TextToMultiTrainer(pl.LightningModule):
    def __init__(self,
                 args,
                 config,
                 model,
                 tokenizer,
                 ):
        super().__init__()
        self.args = args
        self.config = config
        self.arg_opt = utils.AttrDict(config['optimizer'])
        self.arg_sche = utils.AttrDict(config['schedular'])
        self.model = model
        self.tokenizer = tokenizer
        self.automatic_optimization = False # pytorch lightning turn off Optimize
        self.step_size = 100
        self.warmup_iterations = self.arg_sche.warmup_epochs*self.step_size  
        self.save_hyperparameters()
        
    def training_step(self, train_batch, idx):
        opt = self.optimizers()
        opt.zero_grad()

        query = train_batch['query_str_tensor']
        doc_text = train_batch['doc_str_tensor']
        doc_image = train_batch['doc_image_tensor']
        doc_id = train_batch['doc_id']
        context_labels = train_batch.get('context_labels', None)

        loss_ita, loss_ctx_labels, loss_itm = self.model.forward(
            query, doc_text, doc_image, doc_id, context_labels, self.args.neg_matching, self.args.neg_matchingv2)            
        loss = loss_ita + loss_ctx_labels + loss_itm
        self.manual_backward(loss)
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
        
        opt.step()
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("loss_ita", loss_ita, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        if self.args.ctx_prediction:
            self.log("loss_ctx", loss_ctx_labels, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
        if self.args.neg_matching:
            self.log("loss_itm", loss_itm, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        sch = self.lr_schedulers()
        # step every `n` epochs
        if self.current_epoch==0 and self.global_step%self.step_size==0 and self.global_step<=self.warmup_iterations: 
            sch.step(self.global_step//self.step_size)
        if self.trainer.is_last_batch:
            sch.step(self.current_epoch+self.arg_sche.warmup_epochs+1)
        return loss
    

    def validation_step(self, val_batch, idx, dataloader_idx=0):     
        if dataloader_idx == 0: # query
            img_query = val_batch['img_query_str_tensor']
            txt_query = val_batch['txt_query_str_tensor']
            multi_query = val_batch['multi_query_str_tensor']
            doc_id = val_batch['doc_id']

            
            img_query['input_ids'] = img_query['input_ids'].view(img_query['input_ids'].shape[0],-1)
            img_query['attention_mask'] = img_query['attention_mask'].view(img_query['input_ids'].shape[0],-1)

            if "token_type_ids" in img_query.keys():
                img_query["token_type_ids"] = img_query["token_type_ids"].view(img_query['input_ids'].shape[0],-1)
            
            txt_query['input_ids'] = txt_query['input_ids'].view(txt_query['input_ids'].shape[0],-1)
            txt_query['attention_mask'] = txt_query['attention_mask'].view(txt_query['input_ids'].shape[0],-1)

            if "token_type_ids" in txt_query.keys():
                txt_query["token_type_ids"] = txt_query["token_type_ids"].view(txt_query['input_ids'].shape[0],-1)

            multi_query['input_ids'] = multi_query['input_ids'].view(multi_query['input_ids'].shape[0],-1)
            multi_query['attention_mask'] = multi_query['attention_mask'].view(multi_query['input_ids'].shape[0],-1)

            if "token_type_ids" in multi_query.keys():
                multi_query["token_type_ids"] = multi_query["token_type_ids"].view(multi_query['input_ids'].shape[0],-1)
                
            img_query_embeds, img_query_feats = self.model.output_query_feats(img_query)
            txt_query_embeds, txt_query_feats = self.model.output_query_feats(txt_query)
            multi_query_embeds, multi_query_feats = self.model.output_query_feats(multi_query)

            result = {
                "img_query_text": val_batch['img_query_str'],
                "img_query_feats": img_query_feats,
                "txt_query_text": val_batch['txt_query_str'],
                "txt_query_feats": txt_query_feats,
                "multi_query_text": val_batch['multi_query_str'],
                "multi_query_feats": multi_query_feats,
                "docs_id": doc_id,
            }
            if self.args.re_ranking:
                result["img_query_embeds"] = img_query_embeds
                result["img_query_att"] = img_query['attention_mask']
                result["txt_query_embeds"] = txt_query_embeds
                result["txt_query_att"] = txt_query['attention_mask']
                result["multi_query_embeds"] = multi_query_embeds
                result["multi_query_att"] = multi_query['attention_mask']

        elif dataloader_idx == 1: # document
            doc_text = val_batch['doc_str_tensor']
            doc_id = val_batch['doc_id']
            doc_image = val_batch['doc_image_tensor']
            doc_text['input_ids'] = doc_text['input_ids'].view(doc_text['input_ids'].shape[0],-1)
            doc_text['attention_mask'] = doc_text['attention_mask'].view(doc_text['input_ids'].shape[0],-1)
            if "token_type_ids" in doc_text.keys():
                doc_text["token_type_ids"] = doc_text["token_type_ids"].view(doc_text['input_ids'].shape[0],-1)

            docs_embeds,docs_feats,doc_masks = self.model.output_doc_feats(doc_text,doc_image)
            result = {
                "docs_text": val_batch['doc_str'],
                "docs_image": val_batch['image_path'],
                "docs_feats": docs_feats,
                "docs_id": doc_id
            }
        
            # ===== Context Prediction =====
            if self.args.ctx_prediction:
                prediction_scores = self.model.context(docs_embeds)
                mean_prediction_scores = torch.mean(prediction_scores,1)
                top_context = torch.topk(mean_prediction_scores,20,dim=1)
                result["top_context"] = top_context
        
            # ===== Re-rank =====
            if self.args.re_ranking:
                result["doc_embeds"] = docs_embeds
                result["doc_att"] = doc_masks
                
        return result

            

    def validation_epoch_end(self, validation_step_outputs):
        queries = validation_step_outputs[0]
        all_queries_doc_ids = torch.stack([feat for output in queries for feat in output["docs_id"]])
        all_img_queries = torch.stack([feat for output in queries for feat in output["img_query_feats"]])
        all_img_queries_text = [str(feat) for output in queries for feat in output["img_query_text"]]
        all_txt_queries = torch.stack([feat for output in queries for feat in output["txt_query_feats"]])
        all_txt_queries_text = [str(feat) for output in queries for feat in output["txt_query_text"]]
        all_multi_queries = torch.stack([feat for output in queries for feat in output["multi_query_feats"]])
        all_multi_queries_text = [str(feat) for output in queries for feat in output["multi_query_text"]]

        docs = validation_step_outputs[1]
        all_docs_ids = torch.stack([feat for output in docs for feat in output["docs_id"]])
        all_docs = torch.stack([feat for output in docs for feat in output["docs_feats"]])
        all_docs_captions = [str(feat) for output in docs for feat in output["docs_text"]]
        all_docs_images = [str(feat) for output in docs for feat in output["docs_image"]]
        
        if self.local_rank != None:
            # id
            all_queries_doc_ids_list = [torch.zeros_like(all_queries_doc_ids) for _ in range(dist.get_world_size())]
            dist.all_gather(all_queries_doc_ids_list, all_queries_doc_ids)
            all_queries_doc_ids = torch.cat(all_queries_doc_ids_list, dim=0)  
            temp_all_queries_doc_ids = all_queries_doc_ids
            all_queries_doc_ids, rm_repeat_indices = self.unique(all_queries_doc_ids)  

                    
            # image_query
            all_img_queries_list = [torch.zeros_like(all_img_queries) for _ in range(dist.get_world_size())]
            dist.all_gather(all_img_queries_list, all_img_queries)
            all_img_queries = torch.cat(all_img_queries_list, dim=0)

            all_img_queries = all_img_queries[rm_repeat_indices]
            
            
            all_img_queries_text_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_img_queries_text_list, all_img_queries_text)
            all_img_queries_text = [text for queries_text in all_img_queries_text_list for text in queries_text]
            all_img_queries_text = [all_img_queries_text[i] for i in rm_repeat_indices.tolist()]
            
            # text_query
            all_txt_queries_list = [torch.zeros_like(all_txt_queries) for _ in range(dist.get_world_size())]
            dist.all_gather(all_txt_queries_list, all_txt_queries)
            all_txt_queries = torch.cat(all_txt_queries_list, dim=0)
            all_txt_queries = all_txt_queries[rm_repeat_indices]

            all_txt_queries_text_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_txt_queries_text_list, all_txt_queries_text)
            all_txt_queries_text = [text for queries_text in all_txt_queries_text_list for text in queries_text]
            all_txt_queries_text = [all_txt_queries_text[i] for i in rm_repeat_indices.tolist()]
            
            # multi_query
            all_multi_queries_list = [torch.zeros_like(all_multi_queries) for _ in range(dist.get_world_size())]
            dist.all_gather(all_multi_queries_list, all_multi_queries)
            all_multi_queries = torch.cat(all_multi_queries_list, dim=0)
            all_multi_queries = all_multi_queries[rm_repeat_indices]

            all_multi_queries_text_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_multi_queries_text_list, all_multi_queries_text)
            all_multi_queries_text = [text for queries_text in all_multi_queries_text_list for text in queries_text]
            all_multi_queries_text = [all_multi_queries_text[i] for i in rm_repeat_indices.tolist()]
            
            # multimodal_doc
            all_doc_ids_list = [torch.zeros_like(all_docs_ids) for _ in range(dist.get_world_size())]
            dist.all_gather(all_doc_ids_list, all_docs_ids)
            all_docs_ids = torch.cat(all_doc_ids_list, dim=0)
            
            all_docs_ids, rm_repeat_doc_indices = self.unique(all_docs_ids)   
            
            all_docs_list = [torch.zeros_like(all_docs) for _ in range(dist.get_world_size())]
            dist.all_gather(all_docs_list, all_docs)
            all_docs = torch.cat(all_docs_list, dim=0)
            all_docs = all_docs[rm_repeat_doc_indices]

            all_docs_captions_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_docs_captions_list, all_docs_captions)
            all_docs_captions = [caption for docs_captions in all_docs_captions_list for caption in docs_captions]
            all_docs_captions = [all_docs_captions[i] for i in rm_repeat_doc_indices.tolist()]

            all_docs_image_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_docs_image_list, all_docs_images)
            all_docs_images = [image for docs_images in all_docs_image_list for image in docs_images]
            all_docs_images = [all_docs_images[i] for i in rm_repeat_doc_indices.tolist()]

            if self.args.re_ranking:
                all_docs_embeds = torch.stack([feat for output in docs for feat in output["doc_embeds"]])
                all_docs_masks = torch.stack([feat for output in docs for feat in output["doc_att"]])

                all_docs_embeds_list = [torch.zeros_like(all_docs_embeds) for _ in range(dist.get_world_size())]
                dist.all_gather(all_docs_embeds_list, all_docs_embeds)
                all_docs_embeds = torch.cat(all_docs_embeds_list, dim=0)
                all_docs_embeds = all_docs_embeds[all_docs_ids]
                
                all_docs_masks_list = [torch.zeros_like(all_docs_masks) for _ in range(dist.get_world_size())]
                dist.all_gather(all_docs_masks_list, all_docs_masks)
                all_docs_masks = torch.cat(all_docs_masks_list, dim=0)
                all_docs_masks = all_docs_masks[all_docs_ids]

        img_sims_matrix = all_img_queries @ all_docs.t()
        txt_sims_matrix = all_txt_queries @ all_docs.t()
        multi_sims_matrix = all_multi_queries @ all_docs.t()
        matrix_list = [img_sims_matrix,txt_sims_matrix,multi_sims_matrix]
        labels = F.one_hot(all_queries_doc_ids, len(all_docs)).to(self.device)
        if self.args.re_ranking:
            all_img_queries_embeds = torch.stack([feat for output in queries for feat in output["img_query_embeds"]])
            all_img_queries_masks = torch.stack([feat for output in queries for feat in output["img_query_att"]])
            all_txt_queries_embeds = torch.stack([feat for output in queries for feat in output["txt_query_embeds"]])
            all_txt_queries_masks = torch.stack([feat for output in queries for feat in output["txt_query_att"]])
            all_multi_queries_embeds = torch.stack([feat for output in queries for feat in output["multi_query_embeds"]])
            all_multi_queries_masks = torch.stack([feat for output in queries for feat in output["multi_query_att"]])
            
            score_matrix_i2m = torch.full((len(all_img_queries),len(all_docs)),-100.0).to(self.device)
            score_matrix_t2m = torch.full((len(all_txt_queries),len(all_docs)),-100.0).to(self.device)
            score_matrix_m2m = torch.full((len(all_multi_queries),len(all_docs)),-100.0).to(self.device)
            for type in range(3):
                temp_matrix = matrix_list[type]
                for i,sims in enumerate(tqdm(temp_matrix)): 
                    topk_sim, topk_idx = sims.topk(k=self.args.test_rank, dim=0)
                    if type == 0:
                        queries = all_img_queries_embeds[i].unsqueeze(0).repeat(self.args.test_rank,1,1)
                        queries_mask = all_img_queries_masks[i].unsqueeze(0).repeat(self.args.test_rank,1)
                    elif type == 1:
                        queries = all_txt_queries_embeds[i].unsqueeze(0).repeat(self.args.test_rank,1,1)
                        queries_mask = all_txt_queries_masks[i].unsqueeze(0).repeat(self.args.test_rank,1)
                    else:
                        queries = all_multi_queries_embeds[i].unsqueeze(0).repeat(self.args.test_rank,1,1)
                        queries_mask = all_multi_queries_masks[i].unsqueeze(0).repeat(self.args.test_rank,1)

                    docs = all_docs_embeds[topk_idx]
                    docs_masks = all_docs_masks[topk_idx]
                    
                    itm_logits = self.model.matching_classifier(
                        query_embeds=queries,
                        query_attns=queries_mask,
                        multi_embeds=docs,
                        multi_attns=docs_masks
                    )
                    if type == 0:
                        score_matrix_i2m[i,topk_idx] = itm_logits[:,1]
                    elif type == 1:
                        score_matrix_t2m[i,topk_idx] = itm_logits[:,1]
                    else:
                        score_matrix_m2m[i,topk_idx] = itm_logits[:,1]
                if type == 0: 
                    img_sims_matrix = score_matrix_i2m
                    img_sims_matrix = img_sims_matrix.cpu()
                elif type == 1:
                    txt_sims_matrix = score_matrix_t2m
                    txt_sims_matrix = txt_sims_matrix.cpu()
                else:
                    multi_sims_matrix = score_matrix_m2m
                    multi_sims_matrix = multi_sims_matrix.cpu()
            matrix_list = [img_sims_matrix,txt_sims_matrix,multi_sims_matrix]

        loss_i2m = -torch.sum(F.log_softmax(img_sims_matrix / self.model.temp, dim=1)*labels,dim=1).mean()
        loss_t2m = -torch.sum(F.log_softmax(txt_sims_matrix / self.model.temp, dim=1)*labels,dim=1).mean()
        loss_m2m = -torch.sum(F.log_softmax(multi_sims_matrix / self.model.temp, dim=1)*labels,dim=1).mean()
        img_output_score = score(img_sims_matrix,all_queries_doc_ids)
        txt_output_score = score(txt_sims_matrix,all_queries_doc_ids)
        multi_output_score = score(multi_sims_matrix,all_queries_doc_ids)
        
        val_loss = (loss_i2m + loss_t2m + loss_m2m) / 3
        
        if self.args.ctx_prediction:
            loss_ctx_pred = torch.stack([output["loss_ctx_pred"] for output in validation_step_outputs]).mean()
            val_loss = (val_loss+loss_ctx_pred) / 2
            self.log("val_loss_t2m", loss_t2m, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val_loss_ctx_pred", loss_ctx_pred, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("img_r1", img_output_score['r1'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("img_r5", img_output_score['r5'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("img_r10", img_output_score['r10'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("img_r_mean", img_output_score['r_mean'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("img_mrr10", img_output_score['mrr10'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("txt_r1", txt_output_score['r1'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("txt_r5", txt_output_score['r5'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("txt_r10", txt_output_score['r10'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("txt_r_mean", txt_output_score['r_mean'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("txt_mrr10", txt_output_score['mrr10'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("multi_r1", multi_output_score['r1'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("multi_r5", multi_output_score['r5'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("multi_r10", multi_output_score['r10'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("multi_r_mean", multi_output_score['r_mean'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("multi_mrr10", multi_output_score['mrr10'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {
                "val_loss":val_loss,
                "loss_t2m":loss_t2m,
                "loss_i2m":loss_i2m,
                "loss_m2m":loss_m2m,
                "img_r1":img_output_score['r1'],
                "img_r5":img_output_score['r5'],
                "img_r10":img_output_score['r10'],
                "img_r_mean":img_output_score['r_mean'],
                "img_mrr10":img_output_score['mrr10'],
                "txt_r1":txt_output_score['r1'],
                "txt_r5":txt_output_score['r5'],
                "txt_r10":txt_output_score['r10'],
                "txt_r_mean":txt_output_score['r_mean'],
                "txt_mrr10":txt_output_score['mrr10'],
                "multi_r1":multi_output_score['r1'],
                "multi_r5":multi_output_score['r5'],
                "multi_r10":multi_output_score['r10'],
                "multi_r_mean":multi_output_score['r_mean'],
                "multi_mrr10":multi_output_score['mrr10']
        }
        

    def test_step(self, test_batch, idx, dataloader_idx=0):
        if dataloader_idx == 0: # query
            img_query = test_batch['img_query_str_tensor']
            txt_query = test_batch['txt_query_str_tensor']
            multi_query = test_batch['multi_query_str_tensor']
            doc_id = test_batch['doc_id']

            
            img_query['input_ids'] = img_query['input_ids'].view(img_query['input_ids'].shape[0],-1)
            img_query['attention_mask'] = img_query['attention_mask'].view(img_query['input_ids'].shape[0],-1)

            if "token_type_ids" in img_query.keys():
                img_query["token_type_ids"] = img_query["token_type_ids"].view(img_query['input_ids'].shape[0],-1)
            
            txt_query['input_ids'] = txt_query['input_ids'].view(txt_query['input_ids'].shape[0],-1)
            txt_query['attention_mask'] = txt_query['attention_mask'].view(txt_query['input_ids'].shape[0],-1)

            if "token_type_ids" in txt_query.keys():
                txt_query["token_type_ids"] = txt_query["token_type_ids"].view(txt_query['input_ids'].shape[0],-1)

            multi_query['input_ids'] = multi_query['input_ids'].view(multi_query['input_ids'].shape[0],-1)
            multi_query['attention_mask'] = multi_query['attention_mask'].view(multi_query['input_ids'].shape[0],-1)

            if "token_type_ids" in multi_query.keys():
                multi_query["token_type_ids"] = multi_query["token_type_ids"].view(multi_query['input_ids'].shape[0],-1)
                
            img_query_embeds, img_query_feats = self.model.output_query_feats(img_query)
            txt_query_embeds, txt_query_feats = self.model.output_query_feats(txt_query)
            multi_query_embeds, multi_query_feats = self.model.output_query_feats(multi_query)


            result = {
                "img_query_text": test_batch['img_query_str'],
                "img_query_feats": img_query_feats,
                "txt_query_text": test_batch['txt_query_str'],
                "txt_query_feats": txt_query_feats,
                "multi_query_text": test_batch['multi_query_str'],
                "multi_query_feats": multi_query_feats,
                "docs_id": doc_id,
            }
            if self.args.re_ranking:
                result["img_query_embeds"] = img_query_embeds
                result["img_query_att"] = img_query['attention_mask']
                result["txt_query_embeds"] = txt_query_embeds
                result["txt_query_att"] = txt_query['attention_mask']
                result["multi_query_embeds"] = multi_query_embeds
                result["multi_query_att"] = multi_query['attention_mask']

        elif dataloader_idx == 1: # document
            doc_text = test_batch['doc_str_tensor']
            doc_id = test_batch['doc_id']
            doc_image = test_batch['doc_image_tensor']
            doc_text['input_ids'] = doc_text['input_ids'].view(doc_text['input_ids'].shape[0],-1)
            doc_text['attention_mask'] = doc_text['attention_mask'].view(doc_text['input_ids'].shape[0],-1)
            if "token_type_ids" in doc_text.keys():
                doc_text["token_type_ids"] = doc_text["token_type_ids"].view(doc_text['input_ids'].shape[0],-1)

            docs_embeds,docs_feats,doc_masks = self.model.output_doc_feats(doc_text,doc_image)
            result = {
                "docs_text": test_batch['doc_str'],
                "docs_image": test_batch['image_path'],
                "docs_feats": docs_feats,
                "docs_id": doc_id
            }
        
            # ===== Context Prediction =====
            if self.args.ctx_prediction:
                prediction_scores = self.model.context(docs_embeds)
                mean_prediction_scores = torch.mean(prediction_scores,1)
                top_context = torch.topk(mean_prediction_scores,20,dim=1)
                result["top_context"] = top_context
        
            # ===== Re-rank =====
            if self.args.re_ranking:
                result["doc_embeds"] = docs_embeds
                result["doc_att"] = doc_masks
                
        return result
    
    def test_epoch_end(self, test_step_outputs):
        queries = test_step_outputs[0]
        all_queries_doc_ids = torch.stack([feat for output in queries for feat in output["docs_id"]])
        all_img_queries = torch.stack([feat for output in queries for feat in output["img_query_feats"]])
        all_img_queries_text = [str(feat) for output in queries for feat in output["img_query_text"]]
        all_txt_queries = torch.stack([feat for output in queries for feat in output["txt_query_feats"]])
        all_txt_queries_text = [str(feat) for output in queries for feat in output["txt_query_text"]]
        all_multi_queries = torch.stack([feat for output in queries for feat in output["multi_query_feats"]])
        all_multi_queries_text = [str(feat) for output in queries for feat in output["multi_query_text"]]

        docs = test_step_outputs[1]
        all_docs_ids = torch.stack([feat for output in docs for feat in output["docs_id"]])
        all_docs = torch.stack([feat for output in docs for feat in output["docs_feats"]])
        all_docs_captions = [str(feat) for output in docs for feat in output["docs_text"]]
        all_docs_images = [str(feat) for output in docs for feat in output["docs_image"]]
        
        if self.local_rank != None:
            # id
            all_queries_doc_ids_list = [torch.zeros_like(all_queries_doc_ids) for _ in range(dist.get_world_size())]
            dist.all_gather(all_queries_doc_ids_list, all_queries_doc_ids)
            all_queries_doc_ids = torch.cat(all_queries_doc_ids_list, dim=0)  
            temp_all_queries_doc_ids = all_queries_doc_ids
            all_queries_doc_ids, rm_repeat_indices = self.unique(all_queries_doc_ids)  

                    
            # image_query
            all_img_queries_list = [torch.zeros_like(all_img_queries) for _ in range(dist.get_world_size())]
            dist.all_gather(all_img_queries_list, all_img_queries)
            all_img_queries = torch.cat(all_img_queries_list, dim=0)

            all_img_queries = all_img_queries[rm_repeat_indices]
            
            
            all_img_queries_text_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_img_queries_text_list, all_img_queries_text)
            all_img_queries_text = [text for queries_text in all_img_queries_text_list for text in queries_text]
            all_img_queries_text = [all_img_queries_text[i] for i in rm_repeat_indices.tolist()]
            
            # text_query
            all_txt_queries_list = [torch.zeros_like(all_txt_queries) for _ in range(dist.get_world_size())]
            dist.all_gather(all_txt_queries_list, all_txt_queries)
            all_txt_queries = torch.cat(all_txt_queries_list, dim=0)
            all_txt_queries = all_txt_queries[rm_repeat_indices]

            all_txt_queries_text_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_txt_queries_text_list, all_txt_queries_text)
            all_txt_queries_text = [text for queries_text in all_txt_queries_text_list for text in queries_text]
            all_txt_queries_text = [all_txt_queries_text[i] for i in rm_repeat_indices.tolist()]
            
            # multi_query
            all_multi_queries_list = [torch.zeros_like(all_multi_queries) for _ in range(dist.get_world_size())]
            dist.all_gather(all_multi_queries_list, all_multi_queries)
            all_multi_queries = torch.cat(all_multi_queries_list, dim=0)
            all_multi_queries = all_multi_queries[rm_repeat_indices]

            all_multi_queries_text_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_multi_queries_text_list, all_multi_queries_text)
            all_multi_queries_text = [text for queries_text in all_multi_queries_text_list for text in queries_text]
            all_multi_queries_text = [all_multi_queries_text[i] for i in rm_repeat_indices.tolist()]
            
            # multimodal_doc
            all_doc_ids_list = [torch.zeros_like(all_docs_ids) for _ in range(dist.get_world_size())]
            dist.all_gather(all_doc_ids_list, all_docs_ids)
            temp_all_queries_list = all_doc_ids_list
            all_docs_ids = torch.cat(all_doc_ids_list, dim=0)
            temp_all_queries_list_fix = all_docs_ids
            
            all_docs_ids, rm_repeat_doc_indices = self.unique(all_docs_ids)   
            
            all_docs_list = [torch.zeros_like(all_docs) for _ in range(dist.get_world_size())]
            dist.all_gather(all_docs_list, all_docs)
            all_docs = torch.cat(all_docs_list, dim=0)
            temp_all_docs = all_docs
            all_docs = all_docs[rm_repeat_doc_indices]

            all_docs_captions_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_docs_captions_list, all_docs_captions)
            all_docs_captions = [caption for docs_captions in all_docs_captions_list for caption in docs_captions]
            temp_all_docs_captions = all_docs_captions
            all_docs_captions = [all_docs_captions[i] for i in rm_repeat_doc_indices.tolist()]

            all_docs_image_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_docs_image_list, all_docs_images)
            all_docs_images = [image for docs_images in all_docs_image_list for image in docs_images]
            all_docs_images = [all_docs_images[i] for i in rm_repeat_doc_indices.tolist()]

            if self.args.re_ranking:
                all_docs_embeds = torch.stack([feat for output in docs for feat in output["doc_embeds"]])
                all_docs_masks = torch.stack([feat for output in docs for feat in output["doc_att"]])

                all_docs_embeds_list = [torch.zeros_like(all_docs_embeds) for _ in range(dist.get_world_size())]
                dist.all_gather(all_docs_embeds_list, all_docs_embeds)
                all_docs_embeds = torch.cat(all_docs_embeds_list, dim=0)
                all_docs_embeds = all_docs_embeds[all_docs_ids]
                
                all_docs_masks_list = [torch.zeros_like(all_docs_masks) for _ in range(dist.get_world_size())]
                dist.all_gather(all_docs_masks_list, all_docs_masks)
                all_docs_masks = torch.cat(all_docs_masks_list, dim=0)
                all_docs_masks = all_docs_masks[all_docs_ids]
        
        
        if self.local_rank == 0:
            output_dir = self.args.pickle_output
            doc_path = os.path.join(output_dir,"multimodal_documents.pickle")
            img_path = os.path.join(output_dir,"img_query.pickle")
            txt_path = os.path.join(output_dir,"txt_query.pickle")
            multi_path = os.path.join(output_dir,"multi_query.pickle")
            labels_path = os.path.join(output_dir,"labels.pickle")
            with open(doc_path, 'wb') as handle:
                pickle.dump(all_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(img_path, 'wb') as handle:
                pickle.dump(all_img_queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(txt_path, 'wb') as handle:
                pickle.dump(all_txt_queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(multi_path, 'wb') as handle:
                pickle.dump(all_multi_queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(labels_path, 'wb') as handle:
                pickle.dump(all_queries_doc_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Finish saving pickle. Now you can compute the score.")
        return {"img":True}
        
        img_sims_matrix = all_img_queries @ all_docs.t()
        txt_sims_matrix = all_txt_queries @ all_docs.t()
        multi_sims_matrix = all_multi_queries @ all_docs.t()
        matrix_list = [img_sims_matrix,txt_sims_matrix,multi_sims_matrix]
        
        if self.args.re_ranking:
            all_img_queries_embeds = torch.stack([feat for output in queries for feat in output["img_query_embeds"]])
            all_img_queries_masks = torch.stack([feat for output in queries for feat in output["img_query_att"]])
            all_txt_queries_embeds = torch.stack([feat for output in queries for feat in output["txt_query_embeds"]])
            all_txt_queries_masks = torch.stack([feat for output in queries for feat in output["txt_query_att"]])
            all_multi_queries_embeds = torch.stack([feat for output in queries for feat in output["multi_query_embeds"]])
            all_multi_queries_masks = torch.stack([feat for output in queries for feat in output["multi_query_att"]])
            
            score_matrix_i2m = torch.full((len(all_img_queries),len(all_docs)),-100.0).to(self.device)
            score_matrix_t2m = torch.full((len(all_txt_queries),len(all_docs)),-100.0).to(self.device)
            score_matrix_m2m = torch.full((len(all_multi_queries),len(all_docs)),-100.0).to(self.device)
            for type in range(3):
                temp_matrix = matrix_list[type]
                for i,sims in enumerate(tqdm(temp_matrix)): 
                    topk_sim, topk_idx = sims.topk(k=self.args.test_rank, dim=0)
                    if type == 0:
                        queries = all_img_queries_embeds[i].unsqueeze(0).repeat(self.args.test_rank,1,1)
                        queries_mask = all_img_queries_masks[i].unsqueeze(0).repeat(self.args.test_rank,1)
                    elif type == 1:
                        queries = all_txt_queries_embeds[i].unsqueeze(0).repeat(self.args.test_rank,1,1)
                        queries_mask = all_txt_queries_masks[i].unsqueeze(0).repeat(self.args.test_rank,1)
                    else:
                        queries = all_multi_queries_embeds[i].unsqueeze(0).repeat(self.args.test_rank,1,1)
                        queries_mask = all_multi_queries_masks[i].unsqueeze(0).repeat(self.args.test_rank,1)

                    docs = all_docs_embeds[topk_idx]
                    docs_masks = all_docs_masks[topk_idx]
                    
                    itm_logits = self.model.matching_classifier(
                        query_embeds=queries,
                        query_attns=queries_mask,
                        multi_embeds=docs,
                        multi_attns=docs_masks
                    )
                    if type == 0:
                        score_matrix_i2m[i,topk_idx] = itm_logits[:,1]
                    elif type == 1:
                        score_matrix_t2m[i,topk_idx] = itm_logits[:,1]
                    else:
                        score_matrix_m2m[i,topk_idx] = itm_logits[:,1]
                if type == 0: 
                    img_sims_matrix = score_matrix_i2m
                    img_sims_matrix = img_sims_matrix.cpu()
                elif type == 1:
                    txt_sims_matrix = score_matrix_t2m
                    txt_sims_matrix = txt_sims_matrix.cpu()
                else:
                    multi_sims_matrix = score_matrix_m2m
                    multi_sims_matrix = multi_sims_matrix.cpu()
            matrix_list = [img_sims_matrix,txt_sims_matrix,multi_sims_matrix]

        
        # img_output_score = score(img_sims_matrix,labels)
        # txt_output_score = score(txt_sims_matrix,labels)
        # multi_output_score = score(multi_sims_matrix,labels)
        
        img_output_score = score_v2(img_sims_matrix,all_queries_doc_ids)
        txt_output_score = score_v2(txt_sims_matrix,all_queries_doc_ids)
        multi_output_score = score_v2(multi_sims_matrix,all_queries_doc_ids)
        

        # output context
        if self.args.ctx_prediction:
            all_topk_context_values = torch.stack([feat for output in docs for feat in output["top_context"].values])
            all_topk_context_indices = torch.stack([feat for output in docs for feat in output["top_context"].indices])

        output_data = {"img_score":img_output_score,"txt_score":txt_output_score,"multi_score":multi_output_score,"result":[]}
        # output result
        for index,doc_scores in enumerate(img_sims_matrix):
            temp_dict = {"img":{},"txt":{},"multi":{}}
            for type in range(3):
                if type == 0:
                    inds = torch.argsort(img_sims_matrix[index], descending=True)[:10].tolist()
                    temp_all_queries_text = all_img_queries_text
                elif type == 1:
                    inds = torch.argsort(txt_sims_matrix[index], descending=True)[:10].tolist()
                    temp_all_queries_text = all_txt_queries_text
                else:
                    inds = torch.argsort(multi_sims_matrix[index], descending=True)[:10].tolist()
                    temp_all_queries_text = all_multi_queries_text


                temp = {"id": all_queries_doc_ids[index].item(),
                        "query":temp_all_queries_text[index],
                        "true_doc": all_docs_captions[all_queries_doc_ids[index]],
                        "true_image":all_docs_images[all_queries_doc_ids[index]],
                        }
                rank = 1e20
                true_id = all_queries_doc_ids[index]
                if inds[0] == true_id:
                    temp["correct"] = True
                else:
                    temp["correct"] = False
                for rank,top_id in enumerate(inds):
                    dic_key_cap = str(rank)+"_doc"
                    dic_key_img = str(rank)+"_img"
                    temp[dic_key_cap] = all_docs_captions[top_id]
                    temp[dic_key_img] = all_docs_images[top_id]
                    if self.args.ctx_prediction:
                        context = self.tokenizer.decode(all_topk_context_indices[top_id])
                        dic_key_ctx = str(rank)+"_ctx"
                        temp[dic_key_ctx] = context
                if type == 0:
                    temp_dict["img"] = temp
                elif type == 1:
                    temp_dict["txt"] = temp
                else:
                    temp_dict["multi"] = temp
                
            output_data["result"].append(temp_dict)
            
        with open(self.args.test_output, "w") as outfile:
            json.dump(output_data, outfile, indent = 4)

        self.log("img_r1", img_output_score['r1'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("img_r5", img_output_score['r5'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("img_r10", img_output_score['r10'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("img_r_mean", img_output_score['r_mean'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("img_mrr10", img_output_score['mrr10'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("txt_r1", txt_output_score['r1'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("txt_r5", txt_output_score['r5'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("txt_r10", txt_output_score['r10'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("txt_r_mean", txt_output_score['r_mean'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("txt_mrr10", txt_output_score['mrr10'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("multi_r1", multi_output_score['r1'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("multi_r5", multi_output_score['r5'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("multi_r10", multi_output_score['r10'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("multi_r_mean", multi_output_score['r_mean'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("multi_mrr10", multi_output_score['mrr10'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {
                "img_r1":img_output_score['r1'],
                "img_r5":img_output_score['r5'],
                "img_r10":img_output_score['r10'],
                "img_r_mean":img_output_score['r_mean'],
                "img_mrr10":img_output_score['mrr10'],
                "txt_r1":txt_output_score['r1'],
                "txt_r5":txt_output_score['r5'],
                "txt_r10":txt_output_score['r10'],
                "txt_r_mean":txt_output_score['r_mean'],
                "txt_mrr10":txt_output_score['mrr10'],
                "multi_r1":multi_output_score['r1'],
                "multi_r5":multi_output_score['r5'],
                "multi_r10":multi_output_score['r10'],
                "multi_r_mean":multi_output_score['r_mean'],
                "multi_mrr10":multi_output_score['mrr10']
                }
            

    def configure_optimizers(self):
        # optimizer
        opt_args = dict(lr=float(self.arg_opt.lr), weight_decay=float(self.arg_opt.weight_decay))

        if hasattr(self.arg_opt, 'opt_eps') and self.arg_opt.opt_eps is not None:
            opt_args['eps'] = self.args.opt_eps
        if hasattr(self.arg_opt, 'opt_betas') and self.arg_opt.opt_betas is not None:
            opt_args['betas'] = self.args.opt_betas
        if hasattr(self.arg_opt, 'opt_args') and self.arg_opt.opt_args is not None:
            opt_args.update(self.args.opt_args)

        if self.arg_opt.opt == "adamW":
            optimizer = torch.optim.AdamW(self.model.parameters(),**opt_args)

        # scheduler
        lr_scheduler = None
        if self.arg_sche.sched == 'cosine':
            lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=self.trainer.max_epochs,
                t_mul=getattr(self.arg_sche, 'lr_cycle_mul', 1.),
                lr_min=float(self.arg_sche.min_lr),
                decay_rate=self.arg_sche.decay_rate,
                warmup_lr_init=float(self.arg_sche.warmup_lr),
                warmup_t=self.arg_sche.warmup_epochs,
                cycle_limit=getattr(self.arg_sche, 'lr_cycle_limit', 1),
            )
            # num_epochs = lr_scheduler.get_cycle_length() + self.arg_sche.cooldown_epochs
            # self.trainer.max_epochs = num_epochs

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    
    # https://github.com/pytorch/pytorch/issues/36748
    def unique(self,x, dim=-1):
        unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
        perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([dim]), perm.flip([dim])
        return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)