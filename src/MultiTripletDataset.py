from torch.utils.data import Dataset


class MultiTripletDataset(Dataset):
    def __init__(self, ArticleIDs, AnchorPublisher, SentenceIDs, Sentences, Labels, SampleFrom, TripletSize=100):
        self.Labels = Labels.to_list()
        self.Anchor = Sentences.to_list()
        self.ArticleIDs = ArticleIDs.to_list()
        self.SentenceIDs = SentenceIDs.to_list()
        self.TripletSize = TripletSize
        self.positives = []
        self.negatives = []
               
        
        for L,ID in zip(self.Labels, self.SentenceIDs):
            
            #Assigning triplets of Positive and negative examples
            #Ensure the triplet does not contain the Anchor
            posSamples = SampleFrom[(SampleFrom.Label == L) & (SampleFrom.SentenceID != ID)]["Anchor"]
            negSamples = SampleFrom[(SampleFrom.Label != L) & (SampleFrom.SentenceID != ID)]["Anchor"]
            
            
            self.positives.append(posSamples.sample(n=self.TripletSize).tolist())
            self.negatives.append(negSamples.sample(n=self.TripletSize).tolist())
                    
        
    def __len__(self):
            return len(self.Labels)

    def __getitem__(self, idx):
        
        sample = {"Anchor": self.Anchor[idx], "ArticleID":self.ArticleIDs[idx],
                  "SentenceID":self.SentenceIDs[idx], "TripletSize":self.TripletSize,
                  "Positive":self.positives[idx], "Negative": self.negatives[idx],
                  "Label": self.Labels[idx]}
            
        return sample