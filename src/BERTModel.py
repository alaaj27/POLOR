import torch.nn as nn
import torch

class FineTunedBERT(nn.Module):

    def __init__(self, bert, nLabels=2, pool="mean"):
      
        super(FineTunedBERT, self).__init__()
        
        self.bert = bert
        self.pool = pool
        
        self.pipeline = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, nLabels))
        

    def forward(self, input_ids_, attention_mask_ , return_hidden=False):
        

        hiddenState, ClsPooled = self.bert(input_ids = input_ids_,
                              attention_mask=attention_mask_).values()
        
            
        if self.pool.lower() == "max":
            embeddings = self.max_pooling(hiddenState, attention_mask_)
            
        elif self.pool.lower() == "cls":
            embeddings = ClsPooled
            
        elif self.pool.lower() == "mean":
            embeddings = self.mean_pooling(hiddenState, attention_mask_)

            
        else:
            raise ValueError('Pooling value error.')


        logits = self.pipeline(embeddings)
        
        if return_hidden:
            return hiddenState, logits
        
        return embeddings, logits

    
    def mean_pooling (self, hidden_state, attention_mask):
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        pooled_embeddings = torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9) 
        
        return pooled_embeddings
