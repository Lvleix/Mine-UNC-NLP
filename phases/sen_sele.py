import numpy as np
import math
import heapq
from Net import SemanticNet

def sm(l, i):
    return math.exp(l[i])/sum([math.exp(ele) for ele in l])

class sen_sele_train():
    def __init__(self, train_data, wikis, embedding_class, batch_size=128, embedding_size=50):
        self.train_data = train_data
        self.wikis = wikis
        self.embedding_class = embedding_class
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.pad = [[0. for _ in range(self.embedding_size)]]
        self.dataset1 = []
        self.dataset2 = []
        self.labelset = []
        for data in self.train_data:
            evids = data["evidence"][0]
            for evid in evids:
                if evid[2] == "":
                    continue
                for i in range(len(wikis[evid[2]])):
                    if len(wikis[evid[2]][i]) == 0:
                        continue
                    self.dataset1.append(data["claim"])
                    self.dataset2.append(wikis[evid[2]][i])
                    if i == evid[3]:
                        self.labelset.append(0)
                    else:
                        self.labelset.append(1)

    def train(self):
        pass

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

            data1_batch_embedding = [self.embedding_class.sen_sele_embed(data) for data in data1_batch]
            data2_batch_embedding = [self.embedding_class.sen_sele_embed(data) for data in data2_batch]

            pad_data1_batch_embedding = np.array(pad_batch(data1_batch_embedding, pad))
            pad_data2_batch_embedding = np.array(pad_batch(data2_batch_embedding, pad))
            label_batch = np.array(label_batch)

            yield pad_data1_batch_embedding, pad_data2_batch_embedding, label_batch


class sen_sele_infer():
    def __init__(self, wikis, embedding_class, model):
        self.wikis = wikis
        self.embedding_class = embedding_class
        self.load_model(model)


    def load_model(self, model):
        pass

    def use_model(self, data1, data2):
        input1 = self.embedding_class.sen_sele_embed(data1)
        input2 = self.embedding_class.sen_sele_embed(data2)
        mp=1
        mm=0
        return mp,mm

    def infer(self, claim, docs, threshold=0.5, k=5):
        scores = []
        for doc in docs:
            for sen in self.wikis[doc]:
                if len(sen) == 0:
                    continue
                mp,mm = self.use_model(claim, sen)
                score = sm([mp,mm],0)
                if score > threshold:
                    scores.append((sen, score))
        return heapq.nlargest(k,scores,key=lambda x: x[1])
        #return [item[0] for item in heapq.nlargest(k,scores,key=lambda x: x[1])]



