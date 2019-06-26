import numpy as np
import math
import heapq
from Net import SemanticNet
from phases import doc_retr
from phases import sen_sele

def sm(l, i):
    return math.exp(l[i])/sum([math.exp(ele) for ele in l])

class cla_veri_train():
    def __init__(self, train_data, wikis, embedding_class, doc_retr_model, sen_sele_model, model_path, epoches=1, batch_size=32, embedding_size=50):
        self.train_data = train_data
        self.wikis = wikis
        self.embedding_class = embedding_class
        self.model_path = model_path
        self.epoches = epoches
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.doc_retr_class = doc_retr.doc_retr_infer(wikis,embedding_class,doc_retr_model,self.embedding_size)
        self.sen_sele_class = sen_sele.sen_sele_infer(wikis,embedding_class,sen_sele_model,self.embedding_size)
        self.pad = [[0. for _ in range(self.embedding_size)]]
        self.NSMN = SemanticNet.NSMN(embedding_size,3,3,3,hidden_dim=3,lstm_layers=3,classify_num=3)
        self.dataset1 = []
        self.dataset2 = []
        self.labelset = []
        for data in self.train_data:
            evids = data["evidence"][0]
            if data["verifiable"] == "VERIFIABLE":
                self.dataset1.append(data["claim"])
                self.dataset2.append(" ".join([wikis[evid[2]][evid[3]] for evid in evids]))
                self.labelset.append({"SUPPORTS":0,"REFUTES":1,"NOT ENOUGH INFO":2}[data["label"]])
            else:
                docs = self.doc_retr_class.infer(data["claim"],threshold=0)
                sens = self.sen_sele_class.infer(data["claim"],docs,threshold=0)
                if len(sens) == 0:
                    continue
                self.dataset1.append(data["claim"])
                self.dataset2.append(" ".join([sen for sen,score in sens][:5]))
                self.labelset.append(2)
        print("train data process complete")


    def train(self):
        for epoch in range(self.epoches):
            for i, (data1_batch, data2_batch, label_batch) in enumerate(self.get_batches(self.batch_size,self.pad)):
                loss = self.NSMN.train_batch(data1_batch,data2_batch,label_batch)
                print("Epoch %d batch %d: loss %s" % (epoch,i,float(loss)))
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

            data1_batch_embedding = [self.embedding_class.cla_veri_embed(data) for data in data1_batch]
            data2_batch_embedding = [self.embedding_class.cla_veri_embed(data) for data in data2_batch]

            pad_data1_batch_embedding = np.array(pad_batch(data1_batch_embedding, pad))
            pad_data2_batch_embedding = np.array(pad_batch(data2_batch_embedding, pad))
            label_batch = np.array(label_batch)

            yield pad_data1_batch_embedding, pad_data2_batch_embedding, label_batch


class cla_veri_infer():
    def __init__(self, wikis, embedding_class, model, embedding_size=50):
        self.wikis = wikis
        self.embedding_class = embedding_class
        self.NSMN = SemanticNet.NSMN(embedding_size,3,3,3,hidden_dim=3,lstm_layers=3,classify_num=3)
        self.load_model(model)


    def load_model(self, model):
        self.NSMN.load_model(model)

    def use_model(self, data1, data2):
        input1 = np.array(self.embedding_class.doc_retr_embed(data1))
        input2 = np.array(self.embedding_class.doc_retr_embed(data2))
        if len(input1) == 0 or len(input2) == 0:
            return 0,0,1
        res = self.NSMN.inference_once(input1, input2)
        su=res[0]
        re=res[1]
        no=res[2]
        return su,re,no

    def infer(self, claim, sens):
        if sens == []:
            return (2,1)
        su,re,no = self.use_model(claim, " ".join([sen for sen,score in sens]))
        return heapq.nlargest(1,[(0,su),(1,re),(2,no)],key=lambda x: x[1])[0]



