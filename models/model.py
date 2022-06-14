import pdb
import torch
import torch.nn.functional as F
from torch import nn
from models.ALBEF.models.model_retrieval import ALBEF
from models.ALBEF.models.vit import interpolate_pos_embed
from models.ALBEF.models.xbert import BertOnlyMLMHead,BertConfig
from models.ViLT.vilt.modules import ViLTransformerSS
from models.METER.meter.modules import METERTransformerSS
from models.matching import MatchingModel

class TextToMultiModel(nn.Module):
    def __init__(self,  
                 args = None,               
                 config = None, 
                 tokenizer = None,
                     
                 ):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer 
        embed_dim = config.embed_dim   
        
        # Choose pretrain
        if args.pretrain == "ALBEF":
            self.model =  ALBEF(config.text_encoder,tokenizer,config)
            text_width = self.model.text_encoder.config.hidden_size
            bert_config = self.model.bert_config
            if config.checkpoint!="":
                checkpoint = torch.load(config.checkpoint, map_location='cpu')
                state_dict = checkpoint['model']
                # reshape positional embedding to accomodate for image resolution change
                pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],self.model.visual_encoder)         
                state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
                for key in list(state_dict.keys()):
                    if 'bert' in key:
                        encoder_key = key.replace('bert.','')         
                        state_dict[encoder_key] = state_dict[key] 
                        del state_dict[key]                
                msg = self.model.load_state_dict(state_dict,strict=False)
                print('load checkpoint from %s'%config.checkpoint) 
                print(msg)
        elif args.pretrain == "ViLT":
            self.model = ViLTransformerSS(config)
            text_width = config.hidden_size
            bert_config = BertConfig.from_json_file(config.bert_config)
        elif args.pretrain == "METER":
            self.model = METERTransformerSS(config)
            text_width = config.hidden_size
            bert_config = BertConfig.from_json_file(config.bert_config)   

        if args.pretrain == "METER" and (self.args.embeds_feats == "cls" or self.args.embeds_feats == "iavg_tcls"):
            self.multi_proj = nn.Linear(config.hidden_size*2, embed_dim)
        else:
            self.multi_proj = nn.Linear(config.hidden_size, embed_dim)

        self.query_proj = nn.Linear(text_width, embed_dim)
        
        if 'vocab_size' in config:
                bert_config.vocab_size = config.vocab_size
          
        self.context = BertOnlyMLMHead(bert_config) 

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']  
        
        # Matching
        self.matching_classifier = MatchingModel(args = args, config=bert_config, text_width=text_width, n_layers=1)
        
        # create the queue
        self.register_buffer("multi_queue", torch.randn(embed_dim, self.queue_size))
        # self.register_buffer("query_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1,self.queue_size),-100))
        self.register_buffer("queue_num", torch.zeros(1, dtype=torch.long))  

        self.multi_queue = nn.functional.normalize(self.multi_queue, dim=0)
        # self.query_queue = nn.functional.normalize(self.query_queue, dim=0)
    
    def forward_feats(self, doc_image, doc_text, query):
        result = {}

        if self.args.pretrain == "ALBEF":  
            output = self.model(doc_image,doc_text,query) 
        elif self.args.pretrain == "ViLT":
            output = self.model(batch={
                "image":doc_image,
                "text_ids":doc_text['input_ids'],
                "text_masks":doc_text['attention_mask'],
                "query_ids":query['input_ids'],
                "query_masks":query['attention_mask'],
            })
        elif self.args.pretrain == "METER":
            output = self.model(batch={
                "image":doc_image,
                "text_ids":doc_text['input_ids'],
                "text_masks":doc_text['attention_mask'],
                "query_ids":query['input_ids'],
                "query_masks":query['attention_mask'],
            })
            
        result["query_embeds"] = output["query_embeds"]
        result["query_atts"] = output["query_atts"]
        result["multi_embeds"] = output["multi_embeds"]
        result["multi_atts"] = output["multi_atts"]

        # How to get feature
        if self.args.embeds_feats == "avg":
            avg_query_embeds = (result["query_embeds"] * result["query_atts"].unsqueeze(-1)).sum(dim=1) / result["query_atts"].sum(dim=1).unsqueeze(-1)
            result["query_feat"] = F.normalize(self.query_proj(avg_query_embeds),dim=-1)
            avg_multi_embeds = (result["multi_embeds"] * result["multi_atts"].unsqueeze(-1)).sum(dim=1) / result["multi_atts"].sum(dim=1).unsqueeze(-1)
            result["multi_feat"] = F.normalize(self.multi_proj(avg_multi_embeds),dim=-1)
        elif self.args.embeds_feats == "cls":
            result["query_feat"] = output["query_cls"].float()
            
            if self.args.pretrain == "METER": #  METER has two cls token
                multi_embeds = torch.cat([output["text_cls"], output["img_cls"]], dim=-1).float()
                result["multi_feat"] = F.normalize(self.multi_proj(multi_embeds),dim=-1)
            else:
                result["multi_feat"] = output["multi_cls"].float()
        elif self.args.embeds_feats == "iavg_tcls":
            result["query_feat"] = output["query_cls"].float()
            text_cls = output["text_cls"]
            avg_img_embeds = (output["image_feats"] * output["image_masks"].unsqueeze(-1)).sum(dim=1) / output["image_masks"].sum(dim=1).unsqueeze(-1)
            concat_embeds = torch.cat([text_cls, avg_img_embeds], dim=-1).float()
            avg_multi_feat = F.normalize(self.multi_proj(concat_embeds),dim=-1)
            result["multi_feat"] = avg_multi_feat.float()
            
        return result
    
    def forward(self, query, doc_text, doc_image, doc_id, context_labels=None, matching=None, matchingv2=None):
        query['input_ids'] = query['input_ids'].view(query['input_ids'].shape[0],-1)
        query['attention_mask'] = query['attention_mask'].view(query['input_ids'].shape[0],-1)
        if "token_type_ids" in query:
            query['token_type_ids'] = query['token_type_ids'].view(query['input_ids'].shape[0],-1)
          
        doc_text['input_ids'] = doc_text['input_ids'].view(doc_text['input_ids'].shape[0],-1)
        doc_text['attention_mask'] = doc_text['attention_mask'].view(doc_text['input_ids'].shape[0],-1)
        if "token_type_ids" in doc_text:
            doc_text['token_type_ids'] = doc_text['token_type_ids'].view(doc_text['input_ids'].shape[0],-1)
            
        result = self.forward_feats(doc_image, doc_text, query)
        multi_feat = result['multi_feat'] # B, 1, H
        query_feat = result['query_feat'] # B, 1, H
        multi_embeds = result['multi_embeds'] # B, L, H
        multi_atts = result['multi_atts'] # B, L, H
        query_embeds = result['query_embeds']
        query_atts = result['query_atts']

        # [TODO] 
        # why only query with doc similarity
        # why not add doc with query similarity?
        with torch.no_grad():
            multi_feat_all = torch.cat([multi_feat.t(),self.multi_queue.clone().detach()],dim=1)
            # query_feat_all = torch.cat([query_feat.t(),self.query_queue.clone().detach()],dim=1)
    
        sim_q2m = query_feat @ multi_feat_all / self.temp
        # sim_m2q = multi_feat @ query_feat_all / self.temp
        
        idx = doc_id.view(-1,1) # batch_size, id [0,1,2,0]
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  # 1, queue_size [[0,1,2,0,12,13,1,0,2]]
        pos_idx = torch.eq(idx, idx_all).float()       # 1,queue_size [[1,0,0,1,0,0,0,1,0],...]
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)     # normalzie [[0.33,0,0,0.33,0,0,0,0.33,0],...]
        loss_q2m = -torch.sum(F.log_softmax(sim_q2m, dim=1)*sim_targets,dim=1).mean()
        # loss_m2q = -torch.sum(F.log_softmax(sim_m2q, dim=1)*sim_targets,dim=1).mean()
        # loss_ita = (loss_q2m + loss_m2q)/2
        loss_ita = loss_q2m
        self._dequeue_and_enqueue(multi_feat, None, idx) # text_feat_m
        # self._dequeue_and_enqueue(multi_feat, query_feat, idx) # text_feat_m

        # ===== Matching Loss =====        
        # Matching Classification
        # Pos (multi_embeds, query_embeds)
        # Neg (multi_embeds_neg, query_embeds)
        # Neg (multi_embeds_image+multi_embeds_neg_text ,query_embeds)
        # Neg (multi_embeds_text+multi_embeds_neg_image ,query_embeds)
        # Neg (multi_embeds, query_embeds_neg)
        loss_itm = 0.0
        if matching:
            with torch.no_grad():
                bs = doc_image.size(0)
                mask = torch.eq(idx, idx.T)
                weights_q2m = F.softmax(sim_q2m[:, :bs]+1e-4, dim=1)
                weights_q2m.masked_fill_(mask, 1e-10)
                # weights_m2q = F.softmax(sim_m2q[:, :bs]+1e-4, dim=1)
                # weights_m2q.masked_fill_(mask, 1e-10)

            # [New]
            if matchingv2:
                neg_doc_text_inps = []
                neg_doc_text_atts = []
                neg_doc_image = []
            
            multi_embeds_neg = []
            multi_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_q2m[b], 1).item()
                multi_embeds_neg.append(multi_embeds[neg_idx])
                multi_atts_neg.append(multi_atts[neg_idx])
                #[New]
                if matchingv2:
                    neg_doc_text_inps.append(doc_text['input_ids'][neg_idx])
                    neg_doc_text_atts.append(doc_text['attention_mask'][neg_idx])
                    neg_doc_image.append(doc_image[neg_idx])
            multi_embeds_neg = torch.stack(multi_embeds_neg, dim=0)
            multi_atts_neg = torch.stack(multi_atts_neg, dim=0)


            # [New]
            if matchingv2:
                neg_doc_text_inps = torch.stack(neg_doc_text_inps, dim=0)
                neg_doc_text_atts = torch.stack(neg_doc_text_atts, dim=0)
                neg_doc_text = {'input_ids': neg_doc_text_inps, 'attention_mask': neg_doc_text_atts}
                neg_doc_image = torch.stack(neg_doc_image, dim=0)
                result_neg_image = self.forward_feats(neg_doc_image, doc_text ,query)
                multi_embeds_image_neg = result_neg_image['multi_embeds']
                multi_atts_image_neg = result_neg_image['multi_atts']
                result_neg_text = self.forward_feats(doc_image, neg_doc_text ,query)
                multi_embeds_text_neg = result_neg_text['multi_embeds']
                multi_atts_text_neg = result_neg_text['multi_atts']

            # Quadra
            if matchingv2:
                query_embeds_matching = torch.cat([query_embeds, query_embeds, query_embeds, query_embeds], dim=0)
                query_attn_matching = torch.cat([query_atts, query_atts, query_atts, query_atts], dim=0)
                multi_embeds_matching = torch.cat([multi_embeds, multi_embeds_neg, multi_embeds_image_neg, multi_embeds_text_neg], dim=0)
                multi_attn_matching = torch.cat([multi_atts, multi_atts_neg, multi_atts_image_neg, multi_atts_text_neg], dim=0)
            else:
            # Binary
                query_embeds_matching = torch.cat([query_embeds, query_embeds], dim=0)
                query_attn_matching = torch.cat([query_atts, query_atts], dim=0)
                multi_embeds_matching = torch.cat([multi_embeds, multi_embeds_neg], dim=0)
                multi_attn_matching = torch.cat([multi_atts, multi_atts_neg], dim=0)
            
            # Triple
            # query_embeds_matching = torch.cat([query_embeds, query_embeds, query_embeds_neg], dim=0)
            # query_attn_matching = torch.cat([query_atts, query_atts, query_atts_neg], dim=0)
            # multi_embeds_matching = torch.cat([multi_embeds, multi_embeds_neg, multi_embeds], dim=0)
            # multi_attn_matching = torch.cat([multi_atts, multi_atts_neg, multi_atts], dim=0)
            

            itm_logits = self.matching_classifier(
                query_embeds=query_embeds_matching,
                query_attns=query_attn_matching,
                multi_embeds=multi_embeds_matching,
                multi_attns=multi_attn_matching
            )

            itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),
                                    torch.zeros(bs * 3 if matchingv2 else bs * 1,dtype=torch.long)],
                                   dim=0).to(doc_image.device)
            # [TODO] Binary_CE
            loss_itm = F.cross_entropy(itm_logits, itm_labels)    
        
        loss_ctx_pred = 0.0
        # ===== Context Prediction Loss =====
        if context_labels is not None:
            ctx_targets = context_labels / context_labels.sum(1,keepdim=True)
            prediction_scores = self.context(multi_embeds)
            mean_prediction_scores = torch.mean(prediction_scores,1)
            loss_ctx_pred = -torch.sum(F.log_softmax(mean_prediction_scores, dim=1)*ctx_targets,dim=1).mean()
        
        return loss_ita, loss_ctx_pred, loss_itm 
    
    def output_itm_logits(self, query, doc_text, doc_image):
        query['input_ids'] = query['input_ids'].view(query['input_ids'].shape[0],-1)
        query['attention_mask'] = query['attention_mask'].view(query['input_ids'].shape[0],-1)
        if "token_type_ids" in query:
            query['token_type_ids'] = query['token_type_ids'].view(query['input_ids'].shape[0],-1)
          
        doc_text['input_ids'] = doc_text['input_ids'].view(doc_text['input_ids'].shape[0],-1)
        doc_text['attention_mask'] = doc_text['attention_mask'].view(doc_text['input_ids'].shape[0],-1)
        if "token_type_ids" in doc_text:
            doc_text['token_type_ids'] = doc_text['token_type_ids'].view(doc_text['input_ids'].shape[0],-1)
            
        result = self.forward_feats(doc_image, doc_text, query)
        multi_feat = result['multi_feat'] # B, 1, H
        query_feat = result['query_feat'] # B, 1, H
        multi_embeds = result['multi_embeds'] # B, L, H
        multi_atts = result['multi_atts'] # B, L, H
        query_embeds = result['query_embeds']
        query_atts = result['query_atts']
        
        itm_logits = self.matching_classifier(
            query_embeds=query_embeds,
            query_attns=query_atts,
            multi_embeds=multi_embeds,
            multi_attns=multi_atts
        )
        return itm_logits
        
        
    
    @torch.no_grad()
    def output_query_feats(self,query):
        if self.args.pretrain == "ALBEF":
            query_output = self.model.text_encoder(query['input_ids'], attention_mask = query["attention_mask"], mode='text')  
            query_embeds = query_output.last_hidden_state
            query_masks = query["attention_mask"]
            query_cls = F.normalize(self.model.text_proj(query_embeds[:,0,:]),dim=-1)
        elif self.args.pretrain == "ViLT":
            query_embeds = self.model.text_embeddings(query['input_ids'])  
            query_masks = query['attention_mask']
            for i, blk in enumerate(self.model.transformer.blocks):
                query_embeds, _ = blk(query_embeds, mask=query_masks)       
            query_embeds = self.model.transformer.norm(query_embeds)
            query_cls = F.normalize(self.query_proj(query_embeds[:,0,:]),dim=-1)
        elif self.args.pretrain == "METER":
            query_embeds =  self.model.text_transformer.embeddings(input_ids=query['input_ids'])
            query_masks = query['attention_mask']
            device = query_embeds.device
            input_shape = query_masks.size()
            extend_query_masks = self.model.text_transformer.get_extended_attention_mask(query_masks, input_shape, device)
            for layer in self.model.text_transformer.encoder.layer:
                query_embeds = layer(query_embeds, extend_query_masks)[0]
            query_embeds = self.model.cross_modal_text_transform(query_embeds)
            query_cls = self.model.cross_modal_text_pooler(query_embeds)
            
        if self.args.embeds_feats == "avg":
            avg_query_embeds = (query_embeds * query_masks.unsqueeze(-1)).sum(dim=1) / query_masks.sum(dim=1).unsqueeze(-1)
            query_feat = F.normalize(self.query_proj(avg_query_embeds),dim=-1)
        elif self.args.embeds_feats == "cls":
            query_feat = query_cls.float()
        elif self.args.embeds_feats == "iavg_tcls":
            query_feat = query_cls.float()
            
        return query_embeds,query_feat

    @torch.no_grad()
    def output_doc_feats(self,doc_text,doc_image):
        if self.args.pretrain == "ALBEF":
            image_embeds = self.model.visual_encoder(doc_image) 
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(doc_image.device)

            doc_text_output = self.model.text_encoder(doc_text['input_ids'], attention_mask = doc_text["attention_mask"], mode='text')  
            doc_text_embeds = doc_text_output.last_hidden_state
            output = self.model.text_encoder(encoder_embeds = doc_text_embeds, 
                                    attention_mask = doc_text["attention_mask"],
                                    encoder_hidden_states = image_embeds,
                                    encoder_attention_mask = image_atts,      
                                    return_dict = True,
                                    mode = 'fusion',
                                    )
            multi_embeds = output.last_hidden_state 
            multi_atts = doc_text["attention_mask"]
            multi_cls = F.normalize(self.model.text_proj(multi_embeds[:,0,:]),dim=-1)
        elif self.args.pretrain == "ViLT":
            output = self.model.output_multi(batch={
                "image":doc_image,
                "text_ids":doc_text['input_ids'],
                "text_masks":doc_text['attention_mask'],
            })
            multi_embeds = output["multi_embeds"]
            multi_atts = output["multi_atts"]
            multi_cls = output["multi_cls"]
        elif self.args.pretrain == "METER":
            output = self.model.output_multi(batch={
                "image":doc_image,
                "text_ids":doc_text['input_ids'],
                "text_masks":doc_text['attention_mask'],
            })
            multi_embeds = output["multi_embeds"] 
            multi_atts = output["multi_atts"]
            text_cls = output["text_cls"]
            img_cls = output["cls_feats_image"]

        if self.args.embeds_feats == "avg":
            avg_multi_embeds = (multi_embeds * multi_atts.unsqueeze(-1)).sum(dim=1) / multi_atts.sum(dim=1).unsqueeze(-1)
            avg_multi_feat = F.normalize(self.multi_proj(avg_multi_embeds),dim=-1)
        elif self.args.embeds_feats == "cls":
            if self.args.pretrain == "METER": # METER has two cls token
                multi_embeds = torch.cat([text_cls, img_cls], dim=-1).float()
                avg_multi_feat = F.normalize(self.multi_proj(multi_embeds),dim=-1)
            else:
                avg_multi_feat = multi_cls.float()
        elif self.args.embeds_feats == "iavg_tcls":
            text_cls = output["text_cls"]
            img_embeds = output["image_feats"]
            avg_img_embeds = (output["image_feats"] * output["image_masks"].unsqueeze(-1)).sum(dim=1) / output["image_masks"].sum(dim=1).unsqueeze(-1)
            concat_embeds = torch.cat([text_cls, avg_img_embeds], dim=-1).float()
            avg_multi_feat = F.normalize(self.multi_proj(concat_embeds),dim=-1)
            
        return multi_embeds,avg_multi_feat, multi_atts
                 
    @torch.no_grad()
    def _dequeue_and_enqueue(self, multi_feat, query_feat, idx):
        # gather keys before updating queue
        idxs = concat_all_gather(idx)
        
        if multi_feat is not None:
            multi_feats = concat_all_gather(multi_feat)
            batch_size = multi_feats.shape[0]
            ptr = int(self.queue_num)
            assert self.queue_size % batch_size == 0  # for simplicity
            self.multi_queue[:, ptr:ptr + batch_size] = multi_feats.T
            
        if query_feat is not None:
            query_feats = concat_all_gather(query_feat)
            batch_size = query_feats.shape[0]
            ptr = int(self.queue_num)
            assert self.queue_size % batch_size == 0  # for simplicity
            self.query_queue[:, ptr:ptr + batch_size] = query_feats.T

        # replace the keys at ptr (dequeue and enqueue)
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_num[0] = ptr  
        

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    group = torch.distributed.group.WORLD
    world_size = torch.distributed.get_world_size(group)
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, group, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output        

