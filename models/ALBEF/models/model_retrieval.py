from functools import partial
from models.ALBEF.models.vit import VisionTransformer
from models.ALBEF.models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 

        embed_dim = config['embed_dim']        
        vision_width = config['vision_width']  
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        self.bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=self.bert_config, add_pooling_layer=False)      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)   
   
    def forward(self, doc_image, doc_text, query):
        output = dict()
        query_output = self.text_encoder(query["input_ids"], 
                                         attention_mask=query["attention_mask"],                      
                                         return_dict=True, 
                                         mode='text')
        query_embeds = query_output.last_hidden_state
        query_feat = F.normalize(self.text_proj(query_embeds[:,0,:]),dim=-1) 
        
        image_embeds = self.visual_encoder(doc_image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(doc_image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1) 


        text_output = self.text_encoder(doc_text["input_ids"], attention_mask = doc_text["attention_mask"],                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 

        multi_output = self.text_encoder(encoder_embeds = text_embeds, 
                                        attention_mask = doc_text["attention_mask"],
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )   
        multi_embeds = multi_output.last_hidden_state 
        multi_atts = torch.ones(multi_embeds.size()[:-1],dtype=torch.long).to(doc_image.device)
        multi_feat = F.normalize(self.text_proj(multi_embeds[:,0,:]),dim=-1)

           
        output['query_embeds'] = query_embeds
        output['query_cls'] =   query_feat
        output['query_atts'] = query["attention_mask"]
        output['doctext_embeds'] = text_embeds
        output['doctext_cls'] = text_feat
        output['img_embeds'] = image_embeds
        output['img_cls'] = image_feat
        output['multi_embeds'] = multi_embeds
        output['multi_cls'] = multi_feat
        output['multi_atts'] = doc_text["attention_mask"]
         
        return output
