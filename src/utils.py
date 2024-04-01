from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from transformers import BertTokenizerFast
import torch
import numpy as np

def custom_collate(samples, maxlen=80):
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    AnchorTokens = {"input_ids":[], "attention_mask":[]}
    posTokens = {"input_ids":[], "attention_mask":[]}
    negTokens = {"input_ids":[], "attention_mask":[]}
    for s in samples:
        
        tokens = tokenizer.batch_encode_plus([s["Anchor"]], max_length = maxlen,
                                             padding="max_length",truncation=True)
        AnchorTokens["input_ids"].append(torch.tensor(tokens["input_ids"][0]))
        AnchorTokens["attention_mask"].append(torch.tensor(tokens["attention_mask"][0]))

    for s in samples:
        tokens_p =tokenizer.batch_encode_plus(s["Positive"], max_length = maxlen, padding="max_length",
                                               truncation=True)
        
        posTokens["input_ids"].append(torch.tensor(tokens_p["input_ids"]))
        posTokens["attention_mask"].append(torch.tensor(tokens_p["attention_mask"]))
        
        tokens_n = tokenizer.batch_encode_plus(s["Negative"], max_length = maxlen, padding="max_length",
                                               truncation=True)
        
        negTokens["input_ids"].append(torch.tensor(tokens_n["input_ids"]))
        negTokens["attention_mask"].append(torch.tensor(tokens_n["attention_mask"]))
    

    AnchorTokens["input_ids"] = torch.stack(AnchorTokens["input_ids"])
    AnchorTokens["attention_mask"] = torch.stack(AnchorTokens["attention_mask"])
    
    posTokens["input_ids"] = torch.stack(posTokens["input_ids"])
    posTokens["attention_mask"] = torch.stack(posTokens["attention_mask"])
    
    negTokens["input_ids"]= torch.stack(negTokens["input_ids"])
    negTokens["attention_mask"] = torch.stack(negTokens["attention_mask"])
    
    IDs = [s['ArticleID'] for s in samples]
    Labels = [s['Label'] for s in samples]
    
    return { 
        'AnchorTokens': AnchorTokens,
        'posTokens': posTokens,
        'negTokens': negTokens,
        "Labels":torch.tensor(Labels),
        "ArticleIDs":torch.tensor(IDs)}   


def perArticleAccuracy(total_predictions , total_labels, total_ids ):
    
    id_true = {ID:[0 for i in range(len(set(total_labels)))] for ID in list(set(total_ids)) }
    id_predicted = {ID:[0 for i in range(len(set(total_labels)))] for ID in list(set(total_ids))}
    
    for l_true , l_pred, ID in zip(total_labels, total_predictions, total_ids ):
        id_true[ID][l_true] +=1
        id_predicted[ID][l_pred] +=1
        
    perArticlePreds , perArticleTrue  = [] , []
    
    for ID in list(set(total_ids)):
        perArticlePreds.append(
            np.array(id_predicted[ID]).argmax()
        )
        
        perArticleTrue.append(
            np.array(id_true[ID]).argmax()
        )
    
    return accuracy_score(perArticleTrue, perArticlePreds), confusion_matrix(perArticleTrue,perArticlePreds)
