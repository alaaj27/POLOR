import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import warnings
from transformers import AutoModel, BertTokenizerFast
from transformers import AdamW
from Additive import AdditiveAttention
from FineTunedBERTForBiasDetection import FineTunedBERTForBiasDetection
from BERTModel import FineTunedBERT
from MultiTripletDataset import MultiTripletDataset


def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default= 80)
    parser.add_argument("--triplet_size", type=int, default= 5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--maxlen", type=int, default=80)    
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--loss_objective", type=str, default="MMA")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--distance_norm", type=int, default=2)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--train_path", type=str, default="../data/train.csv")
    parser.add_argument("--test_path", type=str, default="../data/test.csv")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    
    arguments = get_args()
    
    
    #Setting up the data
    train_set = pd.read_csv(arguments.train_path)
    train, val = train_test_split(train_set, test_size= 0.15, random_state=42, stratify=train_set["Label"])
    test = pd.read_csv(arguments.test_path)

    TripletSize = arguments.triplet_size
    
    train_data = MultiTripletDataset(train["ArticleID"], train["Publisher"], train["SentenceID"],train["Anchor"],
                         train["Label"],train[["Anchor","Label","SentenceID"]], TripletSize=TripletSize)

    val_data = MultiTripletDataset(val["ArticleID"],val["Publisher"], val["SentenceID"], val["Anchor"],
                             val["Label"],val[["Anchor","Label","SentenceID"]], TripletSize=TripletSize)

    test_data = MultiTripletDataset(test["ArticleID"],test["Publisher"], test["SentenceID"], test["Anchor"],
                             test["Label"],test[["Anchor","Label","SentenceID"]], TripletSize=TripletSize)
    
    
    
    #preparing the model 
    device = arguments.device
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
       
        bert = AutoModel.from_pretrained("bert-base-uncased")
        FineTunedBERTModel = nn.DataParallel(FineTunedBERT(bert, nLabels=2, pool=arguments.pooling))
        
        class_weights = compute_class_weight('balanced', np.unique(train_data.Labels), train_data.Labels)
        weights= torch.tensor(class_weights,dtype=torch.float).to(device)
        lossFunction = nn.CrossEntropyLoss(weight=weights)

    
    AdditiveAttentionModel = AdditiveAttention(bert.config.hidden_size)
    

    optimizer = AdamW(list(FineTunedBERTModel.module.parameters()) +
                      list(AdditiveAttentionModel.parameters()),
                      lr = arguments.lr, 
                      eps = 1e-8)
   
    
    model = FineTunedBERTForBiasDetection(train_data, val_data, test_data,
                                          optimizer, lossFunction,
                                          FineTunedBERTModel, AdditiveAttentionModel,
                                          distance_norm = arguments.distance_norm,
                                          margin = arguments.margin,
                                          alpha = arguments.alpha,
                                          beta = arguments.beta,
                                          maxLength= arguments.maxlen,
                                          batch_size = arguments.batch_size,
                                          LossType=arguments.loss_objective,
                                          device = device
                                         )

    
    model.trainer(epochs=arguments.epochs)

