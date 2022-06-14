import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingModel(nn.Module):
    def __init__(self, args, config, text_width, n_layers):
        super().__init__()
        
        self.config = config
        from models.ALBEF.models.xbert import BertModel
        self.config.num_hidden_layers = 4
        self.config.fusion_layer = 0
        self.itm_transformer = BertModel(self.config, add_pooling_layer=False)
        self.itm_head = nn.Linear(text_width, 2)
        
    def forward(self, query_embeds, query_attns, multi_embeds, multi_attns):
        
        output = self.itm_transformer(
            encoder_embeds = query_embeds, 
            attention_mask = query_attns, 
            encoder_hidden_states = multi_embeds,
            encoder_attention_mask = multi_attns,
            return_dict = True,
            mode = 'fusion',
        )

        
        embeddings = output.last_hidden_state[:, 0, :]
        logits = self.itm_head(embeddings)
        return logits
        
        
if __name__ == "__main__":
    from models.ALBEF.models.xbert import BertConfig
    import yaml

    
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    
    with open('configs/model_retrieval.yaml') as f:
        config = yaml.safe_load(f)

    config = AttrDict(config)
    
    
    bert_config = BertConfig.from_json_file(config['bert_config'])
    
    m = MatchingModel(config=bert_config, text_width=bert_config.hidden_size, n_layers=1)
    
    e1 = torch.rand(5, 10, 768)
    e2 = torch.rand(5, 7, 768)
    a1 = torch.ones(e1.size()[:-1],dtype=torch.long)
    a2 = torch.ones(e2.size()[:-1],dtype=torch.long)
    
    o = m(e1, a1, e2, a2)
    print(o.shape)
    