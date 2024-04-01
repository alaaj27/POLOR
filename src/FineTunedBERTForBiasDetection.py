from torch.utils.data import DataLoader
from utils import custom_collate, perArticleAccuracy
import torch
import random
import torch.nn as nn
from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np 
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

class FineTunedBERTForBiasDetection:
    
    def __init__(self, train_data, val_data, test_data,
                 optimizer, lossFunction,
                 FineTunedBERTModel, AdditiveAttentionModel,
                 distance_norm = 2, margin = 1, alpha = 0.25, beta = 0.25,
                 maxLength=80, batch_size = 80, LossType = "MMA", device = "cuda"
                ):
        
        
        self.history = dict()
        
        self.batch_size = batch_size
        
        collate_fn = lambda batch: custom_collate(batch, maxlen=maxLength)
        
        self.train_data = DataLoader(train_data, batch_size=self.batch_size, shuffle=False,
                                     collate_fn=collate_fn)
        
        self.val_data = DataLoader(val_data, batch_size=self.batch_size,
                                   shuffle=False, collate_fn=collate_fn)
        
        self.test_data = DataLoader(test_data, batch_size=self.batch_size,
                                    shuffle=False, collate_fn=collate_fn)
        
        self.device = device 
        
        self.LossType = LossType
        
        self.alpha = alpha
        
        self.beta = beta
        
        self.distance_norm = distance_norm
        
        self.margin = margin
        
        self.model = FineTunedBERTModel.to(self.device)
        
        self.AdditiveAttentionModel = AdditiveAttentionModel.to(self.device)
        
        self.optimizer = optimizer
        
        self.lossFunction = lossFunction

        self.model.module.bert.requires_grad_(True)        
    
    
    def ComputeBertLoss(self, batch):
        """
            Loss obtained by Anchor only.
        """
        
        Embeddings, preds = self.model(batch["AnchorTokens"]["input_ids"].to(self.device), 
                                       batch["AnchorTokens"]["attention_mask"].to(self.device))
     
        
        bert_loss = self.lossFunction(preds, batch["Labels"].to(self.device))
        
        return bert_loss, preds.to("cpu"), Embeddings.to("cpu")
        
    

    def ComputeRandomTripletLoss(self, batch):
        
      
        anchEmbeddings, _ = self.model(torch.tensor(batch["AnchorTokens"]["input_ids"]).to(self.device), 
                                           torch.tensor(batch["AnchorTokens"]["attention_mask"]).to(self.device))
        
        
        pos_batch_input_ids=[]
        pos_batch_attentions=[]

        neg_batch_input_ids=[]        
        neg_batch_attentions=[]

        
        batch_size=  batch["posTokens"]["attention_mask"].size(0)
        TripletSize = batch["posTokens"]["attention_mask"].size(1)
        
        for i in range(batch_size):
            
            #Sample a random positive example
            sample = random.sample(range(0, TripletSize), 1)[0]
            
            pos_batch_attentions.append(batch["posTokens"]["attention_mask"][i][sample])
            pos_batch_input_ids.append( batch["posTokens"]["input_ids"][i][sample])

            #Sample a random negative example
            sample = random.sample(range(0, TripletSize), 1)[0]

            neg_batch_attentions.append(batch["negTokens"]["attention_mask"][i][sample])
            neg_batch_input_ids.append(batch["negTokens"]["input_ids"][i][sample])
            
        
        posEmbeddings, _ = self.model(torch.stack(pos_batch_input_ids).to(self.device), 
                                      torch.stack(pos_batch_attentions).to(self.device))
        
        negEmbeddings, _ = self.model(torch.stack(neg_batch_input_ids).to(self.device), 
                                      torch.stack(neg_batch_attentions).to(self.device))
             
        
        triplet_loss = nn.TripletMarginLoss(margin=self.margin, p = self.distance_norm)
        RandomTLoss = triplet_loss(anchEmbeddings, posEmbeddings, negEmbeddings)
        
        
        return RandomTLoss
    
    def ComputeInBatchMeanTripletLoss(self, batch):
        
        
        batch_size, mini_batch_size ,seq_len = batch["posTokens"]["input_ids"].size()       
        
        
        anchors, _ = self.model(batch["AnchorTokens"]["input_ids"].to(self.device), 
                                       batch["AnchorTokens"]["attention_mask"].to(self.device))
                
        positives, _ = self.model(batch["posTokens"]["input_ids"].view(-1, seq_len).to(self.device),
                                         batch["posTokens"]["attention_mask"].view(-1, seq_len).to(self.device))

        negatives, _ = self.model(batch["negTokens"]["input_ids"].view(-1, seq_len).to(self.device),
                                         batch["negTokens"]["attention_mask"].view(-1, seq_len).to(self.device))
    
        embedding_size = negatives.size(-1)
        
        positives = positives.view(batch_size, mini_batch_size,embedding_size).mean(dim=1)
        negatives = negatives.view(batch_size, mini_batch_size,embedding_size).mean(dim=1)
        
        
        triplet_loss = nn.TripletMarginLoss(margin=self.margin, p = self.distance_norm)
        InBatchMeanTLoss = triplet_loss(anchors,
                                          positives,
                                          negatives)
        return InBatchMeanTLoss
    

    def ComputeAdditiveAttentionTL(self, batch):
        
        
        batch_size, mini_batch_size ,seq_len = batch["posTokens"]["input_ids"].size()       
        
        
        anchors, _ = self.model(batch["AnchorTokens"]["input_ids"].to(self.device), 
                                       batch["AnchorTokens"]["attention_mask"].to(self.device))
                
        positives, _ = self.model(batch["posTokens"]["input_ids"].view(-1, seq_len).to(self.device),
                                         batch["posTokens"]["attention_mask"].view(-1, seq_len).to(self.device))

        negatives, _ = self.model(batch["negTokens"]["input_ids"].view(-1, seq_len).to(self.device),
                                         batch["negTokens"]["attention_mask"].view(-1, seq_len).to(self.device))
    
        embedding_size = negatives.size(-1)
        
        positives = positives.view(batch_size, mini_batch_size,embedding_size)
        negatives = negatives.view(batch_size, mini_batch_size,embedding_size)
        
        
        weightedPositives = self.AdditiveAttentionModel(positives)
        weightedNegatives = self.AdditiveAttentionModel(negatives)
        
        triplet_loss = nn.TripletMarginLoss(margin=self.margin, p = self.distance_norm)
        AdditiveAttentionTL = triplet_loss(anchors,
                                          weightedPositives,
                                          weightedNegatives)
        
        return AdditiveAttentionTL

    
    def ComputeMinMaxAdditive(self, batch):
        
        
        batch_size, mini_batch_size ,seq_len = batch["posTokens"]["input_ids"].size()       
        
        
        anchors, _ = self.model(batch["AnchorTokens"]["input_ids"].to(self.device), 
                                       batch["AnchorTokens"]["attention_mask"].to(self.device))
                
        positives, _ = self.model(batch["posTokens"]["input_ids"].view(-1, seq_len).to(self.device),
                                         batch["posTokens"]["attention_mask"].view(-1, seq_len).to(self.device))

        negatives, _ = self.model(batch["negTokens"]["input_ids"].view(-1, seq_len).to(self.device),
                                         batch["negTokens"]["attention_mask"].view(-1, seq_len).to(self.device))
    
        embedding_size = negatives.size(-1)
        
        positives = positives.view(batch_size, mini_batch_size,embedding_size)
        negatives = negatives.view(batch_size, mini_batch_size,embedding_size)
        
        
        # Compute the distance between each anchor and positives
        # and get the distant positive for each anchor
        
        dists = torch.cdist(anchors.unsqueeze(1), positives, p=self.distance_norm)
        distant_idx_p = torch.argmax(dists, dim=2)        
        distantPosExamples = positives.gather(1,distant_idx_p.unsqueeze(-1).expand(-1, -1, embedding_size)).squeeze(1)

        # Compute the distance between each anchor and negatives
        # and get the closest negative for each anchor
        dists = torch.cdist(anchors.unsqueeze(1), negatives, p=2)
        closest_idx_n = torch.argmin(dists, dim=2)        
        closestNegExamples = negatives.gather(1,closest_idx_n.unsqueeze(-1).expand(-1, -1, embedding_size)).squeeze(1)

        
        
        triplet_loss = nn.TripletMarginLoss(margin=self.margin, p = self.distance_norm)
        InBatchMinMaxTLoss = triplet_loss(anchors,
                                          distantPosExamples.to(self.device),
                                          closestNegExamples.to(self.device))
        
        
        
        weightedPositives = self.AdditiveAttentionModel(positives)
        weightedNegatives = self.AdditiveAttentionModel(negatives)
        
        triplet_loss = nn.TripletMarginLoss(margin=self.margin, p = self.distance_norm)
        AdditiveAttentionTL = triplet_loss(anchors,
                                          weightedPositives,
                                          weightedNegatives)
   
        return InBatchMinMaxTLoss, AdditiveAttentionTL
    

    def ComputeMinMaxTLoss(self, batch):
        
        
        batch_size, mini_batch_size ,seq_len = batch["posTokens"]["input_ids"].size()       
        
        
        anchors, _ = self.model(batch["AnchorTokens"]["input_ids"].to(self.device), 
                                       batch["AnchorTokens"]["attention_mask"].to(self.device))
                
        positives, _ = self.model(batch["posTokens"]["input_ids"].view(-1, seq_len).to(self.device),
                                         batch["posTokens"]["attention_mask"].view(-1, seq_len).to(self.device))

        negatives, _ = self.model(batch["negTokens"]["input_ids"].view(-1, seq_len).to(self.device),
                                         batch["negTokens"]["attention_mask"].view(-1, seq_len).to(self.device))
    
        embedding_size = negatives.size(-1)
        
        positives = positives.view(batch_size, mini_batch_size,embedding_size)
        negatives = negatives.view(batch_size, mini_batch_size,embedding_size)
        
        
        # Compute the distance between each anchor and positives
        # and get the distant positive for each anchor
        
        dists = torch.cdist(anchors.unsqueeze(1), positives, p=self.distance_norm)
        distant_idx_p = torch.argmax(dists, dim=2)        
        distantPosExamples = positives.gather(1,distant_idx_p.unsqueeze(-1).expand(-1, -1, embedding_size)).squeeze(1)

        # Compute the distance between each anchor and negatives
        # and get the closest negative for each anchor
        dists = torch.cdist(anchors.unsqueeze(1), negatives, p=2)
        closest_idx_n = torch.argmin(dists, dim=2)        
        closestNegExamples = negatives.gather(1,closest_idx_n.unsqueeze(-1).expand(-1, -1, embedding_size)).squeeze(1)

        
        
        triplet_loss = nn.TripletMarginLoss(margin=self.margin, p = self.distance_norm)
        InBatchMinMaxTLoss = triplet_loss(anchors,
                                          distantPosExamples.to(self.device),
                                          closestNegExamples.to(self.device))
        
        return InBatchMinMaxTLoss
       

    
    def train_loop (self):


        size = len(self.train_data.dataset)
        num_batches = len(self.train_data)
        
        print(f"size:{size}, num_batches:{num_batches}")
        
        training_loss, triplet_loss, correct = 0, 0, 0
        total_bert_loss=0

        
        total_predictions = []
        total_labels = []
        total_ids = []
        
        self.model.train()
        for batch in tqdm(self.train_data):

            labels = batch["Labels"]
            Article_IDs = batch["ArticleIDs"]

            
            bert_loss, preds, _ = self.ComputeBertLoss(batch)
            
            
            if self.LossType == "RT": #RandomTripletLoss
                
                RandomTL = self.ComputeRandomTripletLoss(batch)
                total_loss = self.alpha * RandomTL + (1 - self.alpha) * bert_loss
            
            elif self.LossType =="BM" : #BatchMeanTripletLoss
                
                BatchMeanTL = self.ComputeInBatchMeanTripletLoss(batch)
                total_loss = self.alpha * BatchMeanTL + (1 - self.alpha) * bert_loss
              
            
            elif self.LossType =="MM" : #ComputeMinMaxTLoss

                    MinMaxTL= self.ComputeMinMaxTLoss(batch)
                    total_loss = self.alpha * MinMaxTL + (1 - self.alpha) * bert_loss
                    
            elif self.LossType =="AA" : #ComputeAdditiveAttentionTL
                
                AATL = self.ComputeAdditiveAttentionTL(batch)
                triplet_loss+= AATL.item() * self.alpha
                total_loss = self.alpha * AATL + (1 - self.alpha) * bert_loss

    
            elif self.LossType =="MMA" : #ComputeMinMaxAdditive
                

                MMTL, AATL = self.ComputeMinMaxAdditive(batch)
                
                triplet_loss += (self.alpha * AATL.item()) + (self.beta * MMTL.item())
                
                total_loss = (self.alpha * AATL) + (self.beta * MMTL) + (1-self.alpha-self.beta) * bert_loss


            else:
                raise ValueError(f'LossType value error [{self.LossType}]: Available Losses are RT, BM, WM, MD, or Joint.')
            
            total_bert_loss += bert_loss.item()
        
            training_loss += total_loss.item()

            
            correct += (preds.argmax(1) == labels).type(torch.float).sum().item()


            total_predictions.extend((preds.argmax(1)).type(torch.int).to('cpu').numpy())
            total_labels.extend(labels.type(torch.int).to('cpu').numpy())
            total_ids.extend(Article_IDs.type(torch.int).to('cpu').numpy())
            
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                
                
        training_loss /= num_batches
        triplet_loss /= num_batches
        total_bert_loss /= num_batches
        accuracy = correct / size
        ArticleAccuracy, ArtiConf = perArticleAccuracy(total_predictions , total_labels, total_ids )

        print("\t", dict(Counter(total_predictions)), "  true:",dict(Counter(total_labels)))
        
        
        res=[[accuracy, ArticleAccuracy, training_loss, total_bert_loss, triplet_loss ]]
        
        print(tabulate(res, headers=["Accuracy(Sent)", "Accuracy(Art)", "total loss", "bert loss", "triplet loss" ]))
        
        return training_loss, accuracy, ArticleAccuracy, confusion_matrix(total_labels, total_predictions) , ArtiConf
            
 
    
    def val_loop (self, validation, articleAccuracy=False, val_type="Validation"):
        
      
        size = len(validation.dataset)
        num_batches = len(validation)
        
        val_loss, correct, total_bert_loss, triplet_loss = 0, 0, 0, 0

        total_predictions = []
        total_labels = []
        total_ids = []
        
        self.model.eval()
        for batch in tqdm(validation):

            with torch.no_grad():

                labels = batch["Labels"]
                Article_IDs = batch["ArticleIDs"]

                bert_loss, preds, _ = self.ComputeBertLoss(batch)
                
                
                if self.LossType == "RT": #RandomTripletLoss

                    RandomTL = self.ComputeRandomTripletLoss(batch)
                    total_loss = self.alpha * RandomTL + (1 - self.alpha) * bert_loss

                elif self.LossType =="BM" : #BatchMeanTripletLoss

                    BatchMeanTL = self.ComputeInBatchMeanTripletLoss(batch)
                    total_loss = self.alpha * BatchMeanTL + (1 - self.alpha) * bert_loss


                elif self.LossType =="MM" : #ComputeMinMaxTLoss

                        MinMaxTL= self.ComputeMinMaxTLoss(batch)
                        total_loss = self.alpha * MinMaxTL + (1 - self.alpha) * bert_loss

                elif self.LossType =="AA" : #ComputeAdditiveAttentionTL

                    AATL = self.ComputeAdditiveAttentionTL(batch)
                    triplet_loss+= AATL.item() * self.alpha
                    total_loss = self.alpha * AATL + (1 - self.alpha) * bert_loss


                elif self.LossType =="MMA" : #ComputeMinMaxAdditive


                    MMTL, AATL = self.ComputeMinMaxAdditive(batch)

                    triplet_loss += (self.alpha * AATL.item()) + (self.beta * MMTL.item())

                    total_loss = (self.alpha * AATL) + (self.beta * MMTL) + (1-self.alpha-self.beta) * bert_loss
                    
                else:
                    raise ValueError(f'LossType value error [{self.LossType}]: Available Losses are RT, BM, WM, MD, or Joint.')



                total_bert_loss += bert_loss.item()
                val_loss += total_loss.item() 
                
                
                correct += (preds.argmax(1) == labels).type(torch.float).sum().item()
                
                
                total_predictions.extend((preds.argmax(1)).type(torch.int).to('cpu').numpy())
                total_labels.extend(labels.type(torch.int).to('cpu').numpy())
                total_ids.extend(Article_IDs.type(torch.int).to('cpu').numpy())
                
                
        val_loss /= num_batches
        triplet_loss /= num_batches
        total_bert_loss /= num_batches
        accuracy = correct / size
        
        if articleAccuracy:
            ArticleAccuracy, ArtiConf = perArticleAccuracy(total_predictions , total_labels, total_ids)

        else:
            ArticleAccuracy = 0
            
        
        print("\t", dict(Counter(total_predictions)), "  true:",dict(Counter(total_labels)))

        res=[[accuracy, ArticleAccuracy, val_loss, total_bert_loss, triplet_loss ]]
        
        print(val_type + " ...")
        print(tabulate(res, headers=["Accuracy(Sent)", "Accuracy(Art)", "total loss", "bert loss", "triplet loss" ]))

        return val_loss , accuracy, ArticleAccuracy, confusion_matrix(total_labels,total_predictions) , ArtiConf

    
    
    def trainer(self, epochs=5):
                           
        
        train_losses = []
        train_accurices = []
        train_ArticleAccuracies = []
        train_ArtiConf =[]
        train_SentConf =[]
        
        val_losses = []
        val_accurices = []
        val_ArticleAccuracies = []
        val_ArtiConf =[]
        val_SentConf =[]
        
        test_losses = []
        test_accurices = []
        test_ArticleAccuracies = []
        test_ArtiConf=[]
        test_SentConf=[]
        
        
        for epoch in range(epochs):
            
            print(f"Epoch {epoch+1} of {epochs}")
            print("-------------------------------")
            
            t_loss, t_acc, t_ArticleAccuracy, t_SentConf, t_ArtiConf = self.train_loop()          
            train_losses.append(t_loss)
            train_accurices.append(t_acc)
            train_ArticleAccuracies.append(t_ArticleAccuracy)
            train_SentConf.append(t_SentConf)
            train_ArtiConf.append(t_ArtiConf)
        
            
            v_loss, v_acc, v_ArticleAccuracy, v_SentConf, v_ArtiConf = self.val_loop(self.val_data,
                                                                                     articleAccuracy=True,
                                                                                     val_type="Validation")
            
            val_losses.append(v_loss)
            val_accurices.append(v_acc)
            val_ArticleAccuracies.append(v_ArticleAccuracy)
            val_SentConf.append(v_SentConf)
            val_ArtiConf.append(v_ArtiConf)


            Test_loss, Test_acc, Test_perArt_acc, te_SentConf, te_ArtiConf = self.val_loop(self.test_data,
                                                                                           articleAccuracy=True, 
                                                                                           val_type="Testing")
            test_losses.append(Test_loss)
            test_accurices.append(Test_acc)
            test_ArticleAccuracies.append(Test_perArt_acc)
            test_SentConf.append(te_SentConf)
            test_ArtiConf.append(te_ArtiConf)

           
        
            
        self.history["train_loss"] = train_losses
        self.history["train_accuracy"] = train_accurices
        self.history["train_ArticleAccuracy"] = train_ArticleAccuracies
        self.history["train_SentConf"] = train_SentConf
        self.history["train_ArtiConf"] = train_ArtiConf
        
        self.history["val_loss"] = val_losses
        self.history["val_accuracy"] = val_accurices
        self.history["val_ArticleAccuracies"] = val_ArticleAccuracies
        self.history["val_SentConf"] = val_SentConf
        self.history["val_ArtiConf"] = val_ArtiConf

        self.history["test_loss"] = test_losses #[0 for i in range(len(val_losses))]
        self.history["test_accuracy"] = test_accurices
        self.history["test_ArticleAccuracies"] = test_ArticleAccuracies
        self.history["test_SentConf"] = test_SentConf
        self.history["test_ArtiConf"] = test_ArtiConf
        
        