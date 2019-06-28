import numpy as np
from nltk.corpus import wordnet as wn
import math
import heapq
from Net import SemanticNet
import sys

def sm(l, i):
    return math.exp(l[i])/sum([math.exp(ele) for ele in l])

class doc_retr_train():
    def __init__(self, train_data, wikis, embedding_class, model_path, epoches = 1, batch_size=128, embedding_size=50, mode="demo"):
        self.train_data = train_data
        self.wikis = wikis
        self.embedding_class = embedding_class
        self.model_path = model_path
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.epoches = epoches
        self.pad = [[0. for _ in range(self.embedding_size)]]
        if mode in ["demo"]:
            self.NSMN = SemanticNet.NSMN(embedding_size,3,3,3,hidden_dim=3,lstm_layers=3,classify_num=2)
        else:
            self.NSMN = SemanticNet.NSMN(embedding_size, 40, 40, 30, hidden_dim=128, lstm_layers=3, classify_num=2)
        self.dataset1 = []
        self.dataset2 = []
        self.labelset = []
        self.adversarial_databatches1 = []
        self.adversarial_databatches2 = []
        self.adversarial_labelbatches = []
        for data in self.train_data:
            evids = data["evidence"][0]
            for evid in evids:
                if evid[2] == "":
                    continue
                if "-LRB-" in evid[2] and "-RRB-" in evid[2]:
                    self.dataset1.append(data["claim"])
                    self.dataset2.append(evid[2]+" . "+wikis[evid[2]][0])
                    self.labelset.append(0)
                    tmps = evid[2].split("-LRB-")[0]
                    for title in wikis:
                        if tmps in title and "-LRB-" in title and "-RRB-" in title and title != evid[2]:
                            if title.split("-LRB-")[0] == tmps:
                                self.dataset1.append(data["claim"])
                                self.dataset2.append(title+" . "+wikis[title][0])
                                self.labelset.append(1)
        print("train data process complete")
        print(len(self.train_data),len(self.dataset1),len(self.dataset2),len(self.labelset))

    def train(self,adversarial_train=False):
        for epoch in range(self.epoches):
            for i, (data1_batch, data2_batch, label_batch) in enumerate(self.get_batches(self.batch_size,self.pad)):
                loss = self.NSMN.train_batch(data1_batch,data2_batch,label_batch)
                print("Epoch %d batch %d: loss %s" % (epoch,i,float(loss)))
            if adversarial_train:
                for i in range(len(self.adversarial_databatches1)):
                    loss = self.NSMN.train_batch(self.adversarial_databatches1[i],
                                                 self.adversarial_databatches2[i],
                                                 self.adversarial_labelbatches[i])
                    print("Epoch %d adversarial batch %d: loss %s" % (epoch, i, float(loss)))
        self.NSMN.save_model(self.model_path)


    def get_batches(self,batch_size,pad):
        def pad_batch(batch, pad):
            max_length = max([len(sentence) for sentence in batch])
            # max_length = 300
            return [sentence + pad * (max_length - len(sentence)) for sentence in batch]

        for batch_i in range(len(self.dataset1) // batch_size):
            start_i = batch_i * batch_size
            data1_batch = self.dataset1[start_i: start_i + batch_size]  # claim
            data2_batch = self.dataset2[start_i: start_i + batch_size]  # evidence
            label_batch = self.labelset[start_i: start_i + batch_size]

            data1_batch_embedding = [self.embedding_class.doc_retr_embed(data) for data in data1_batch]
            data2_batch_embedding = [self.embedding_class.doc_retr_embed(data) for data in data2_batch]

            pad_data1_batch_embedding = np.array(pad_batch(data1_batch_embedding, pad))
            pad_data2_batch_embedding = np.array(pad_batch(data2_batch_embedding, pad))
            label_batch = np.array(label_batch)

            yield pad_data1_batch_embedding, pad_data2_batch_embedding, label_batch

    def add_adversarial_data(self):
        for i, (data1_batch, data2_batch, label_batch) in enumerate(self.get_batches(self.batch_size, self.pad)):
            u,v = self.NSMN.generate_attack_sample(data1_batch,data2_batch,label_batch)
            self.adversarial_databatches1.append(u)
            self.adversarial_databatches2.append(v)
            self.adversarial_labelbatches.append(label_batch)
            print("\tbatch %d" % i)


class doc_retr_infer():
    def __init__(self, wikis, embedding_class, model, embedding_size=50, mode="demo"):
        self.wikis = wikis
        self.embedding_class = embedding_class
        if mode in ["demo"]:
            self.NSMN = SemanticNet.NSMN(embedding_size,3,3,3,hidden_dim=3,lstm_layers=3,classify_num=2)
        else:
            self.NSMN = SemanticNet.NSMN(embedding_size, 40, 40, 30, hidden_dim=128, lstm_layers=3, classify_num=2)
        self.load_model(model)


    def load_model(self, model):
        self.NSMN.load_model(model)

    def use_model(self, data1, data2):
        input1 = np.array(self.embedding_class.doc_retr_embed(data1))
        input2 = np.array(self.embedding_class.doc_retr_embed(data2))
        if len(input1) == 0 or len(input2) == 0:
            return 0,1
        res = self.NSMN.inference_once(input1,input2)
        mp=res[0]
        mm=res[1]
        return mp,mm

    def infer(self, claim, Singularization=False, threshold=0.5, k=5):
        docs = []
        docs_disambi = []
        for key in self.wikis:
            if key.replace("_"," ").lower() in claim.lower():
                docs.append(key)
            elif key.split("-LRB-")[0].replace("_"," ").lower() in claim.lower():
                docs_disambi.append(key)
        if Singularization and len(docs+docs_disambi) == 0:
            new_claim = " ".join([wn.morphy(word) for word in claim.split(" ")])
            for key in self.wikis:
                if key.replace("_", " ").lower() in new_claim.lower():
                    docs.append(key)
                elif key.split("-LRB-")[0].replace("_", " ").lower() in new_claim.lower():
                    docs_disambi.append(key)
        disambi_scores = []
        for doc in docs_disambi:
            mp,mm = self.use_model(claim, doc+self.wikis[doc][0])
            score = sm([mp,mm],0)
            if score >= threshold:
                disambi_scores.append((doc, score))
        return [(doc,1.) for doc in docs]+heapq.nlargest(k,disambi_scores,key=lambda x: x[1])
        #return docs+[item[0] for item in heapq.nlargest(k,disambi_scores,key=lambda x: x[1])]



