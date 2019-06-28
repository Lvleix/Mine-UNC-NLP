import os
import sys
from phases import doc_retr
from phases import sen_sele
from phases import cla_veri
import Net
import Net.SemanticNet
import embedding_util
import embedding_util.glove_embedding as glove_embedding

def pipeline(mode="adversarial improve", argv=None):
    if mode not in ["train","infer","test","adversarial improve","debug","debug no ad"]:
        print("Invalid mode")
        return

    # if programme exits with error code 0xC00000FD on windows system,
    # it means stack overflow
    # then you can use the code below to fix it
    #sys.setrecursionlimit(100000)

    spath = os.path.dirname(__file__)
    mpath = os.path.dirname(spath)
    wpath = mpath + "/wiki-pages"
    epath = os.path.dirname(mpath)+"/lab2/glove.6B/glove.6B.50d.txt"
    embedding_class = glove_embedding.gloveModel(epath)

    print("read in wikis ... ")
    files = os.listdir(wpath)
    wikis = {}
    if mode in ["debug","debug no ad"]:
        # using small amount of data
        for file in files:

            #if file not in ["wiki-071.jsonl","wiki-072.jsonl","wiki-073.jsonl","wiki-074.jsonl","wiki-075.jsonl",
            #                "wiki-076.jsonl","wiki-077.jsonl","wiki-078.jsonl","wiki-079.jsonl","wiki-080.jsonl"]:
            #    continue
            if file not in ["wiki-071.jsonl"]:
                continue

            print(file)
            f = open(wpath+"/"+file, "r")
            for line in f:
                wiki = eval(line)
                if wiki.get("id", "") == "":
                    continue
                lines = wiki.get("lines", "")
                if lines == "":
                    wikis[wiki["id"]] = [""]
                else:
                    wikis[wiki["id"]] = [sen.split("\t")[1] if len(sen.split("\t")) >= 2 else "" for sen in lines.split("\n")]
            f.close()
        print("wiki dict completed. ")
        # " " == "_" and "(" == "-LRB-" and ")" == "-RRB-"
        # "Savages (band)" == "Savages_-LRB-band-RRB-"

        print("read in data")
        train_data = []
        ccc=0
        with open(mpath+"/train.jsonl","r") as f:
            for line in f:
                td = eval(line.replace("null","''"))
                if ccc < 500 and td["verifiable"] == "NOT VERIFIABLE":
                    train_data.append(td)
                    ccc += 1
                    continue
                evids = td["evidence"][0]
                flag = 0
                for evid in evids:
                    if evid[2] not in wikis:
                        flag = 1
                if flag == 1:
                    continue
                train_data.append(td)
                #train_data = [eval(line.replace("null","''")) for line in f if eval(line.replace("null","''"))[]]
        print(len(train_data))

        with open(mpath+"/shared_task_dev.jsonl","r") as f:
        #with open(spath + "/new_dev_set.jsonl", "r") as f:
            dev_data = [eval(line.replace("null","''")) for line in f]
    else:
        for file in files:
            print("reading file",file)
            f = open(wpath + "/" + file, "r")
            for line in f:
                wiki = eval(line)
                if wiki.get("id", "") == "":
                    continue
                lines = wiki.get("lines", "")
                if lines == "":
                    wikis[wiki["id"]] = [""]
                else:
                    wikis[wiki["id"]] = [sen.split("\t")[1] if len(sen.split("\t")) >= 2 else "" for sen in
                                         lines.split("\n")]
            f.close()
        print("wiki dict completed. ")
        # " " == "_" and "(" == "-LRB-" and ")" == "-RRB-"
        # "Savages (band)" == "Savages_-LRB-band-RRB-"

        print("read in data")
        train_data = []
        ccc = 0
        with open(mpath + "/train.jsonl", "r") as f:
            train_data = [eval(line.replace("null","''")) for line in f]

        with open(mpath + "/shared_task_dev.jsonl", "r") as f:
            dev_data = [eval(line.replace("null", "''")) for line in f]

    model_path = spath+"/model/"

    if mode in ["train","debug no ad"]:
        doc_retr_model = model_path+"doc_retr_model/doc_retr_model.pyt"
        sen_sele_model = model_path+"sen_sele_model/sen_sele_model.pyt"
        cla_veri_model = model_path+"cla_veri_model/cla_veri_model.pyt"
        print("doc retr")
        doc_retr_class = doc_retr.doc_retr_train(train_data, wikis, embedding_class, doc_retr_model, 1)
        doc_retr_class.train()
        del doc_retr_class
        print("sen sele")
        sen_sele_class = sen_sele.sen_sele_train(train_data,wikis, embedding_class, sen_sele_model, 1)
        sen_sele_class.train()
        del sen_sele_class
        print("cla veri")
        cla_veri_class = cla_veri.cla_veri_train(train_data,wikis,embedding_class,doc_retr_model,sen_sele_model,cla_veri_model,1)
        cla_veri_class.train()
        del cla_veri_class
    elif mode == "infer":
        print("Please type in ... ")
        claim = input()
        doc_retr_model = model_path+"doc_retr_model/doc_retr_model.pyt"
        sen_sele_model = model_path+"sen_sele_model/sen_sele_model.pyt"
        cla_veri_model = model_path+"cla_veri_model/cla_veri_model.pyt"
        doc_retr_class = doc_retr.doc_retr_infer(wikis,embedding_class,doc_retr_model)
        sen_sele_class = sen_sele.sen_sele_infer(wikis,embedding_class,sen_sele_model)
        cla_veri_class = cla_veri.cla_veri_infer(wikis,embedding_class,cla_veri_model)
        docs = doc_retr_class.infer(claim)
        sens = sen_sele_class.infer(claim,docs)
        result, score = cla_veri_class.infer(claim,sens)
        res = {0:"SUPPORTS",1:"REFUTES",2:"NOT ENOUGH INFO"}[result]
        print(res)
        if result in [0,1]:
            for i in range(len(sens)):
                print(i+1,sens[i][0])
    elif mode == "test":
        doc_retr_model = model_path+"doc_retr_model/doc_retr_model.pyt"
        sen_sele_model = model_path+"sen_sele_model/sen_sele_model.pyt"
        cla_veri_model = model_path+"cla_veri_model/cla_veri_model.pyt"
        doc_retr_class = doc_retr.doc_retr_infer(wikis, embedding_class, doc_retr_model)
        sen_sele_class = sen_sele.sen_sele_infer(wikis, embedding_class, sen_sele_model)
        cla_veri_class = cla_veri.cla_veri_infer(wikis, embedding_class, cla_veri_model)
        counts = 0
        rights = 0
        for test in dev_data:
            counts += 1
            claim = test["claim"]
            docs = doc_retr_class.infer(claim)
            sens = sen_sele_class.infer(claim, docs)
            result, score = cla_veri_class.infer(claim, sens)
            label = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}[result]
            if label != test["label"]:
                continue
            if test["label"] == "NOT ENOUGH INFO" and result == 2:
                rights += 1
                continue
            flag = 1
            for evid in test["evidence"][0]:
                if wikis[evid[2]][evid[3]] not in sens:
                    flag = 0
            if flag == 1:
                rights += 1
        print("Acc:",rights/counts)
    elif mode in ["adversarial improve","debug"]:
        default_argv = {"iters":1}

        doc_retr_model = model_path + "doc_retr_model/doc_retr_model.pyt"
        sen_sele_model = model_path + "sen_sele_model/sen_sele_model.pyt"
        cla_veri_model = model_path + "cla_veri_model/cla_veri_model.pyt"
        print("doc retr")
        doc_retr_class = doc_retr.doc_retr_train(train_data, wikis, embedding_class, doc_retr_model, 1)
        doc_retr_class.train()
        print("sen sele")
        sen_sele_class = sen_sele.sen_sele_train(train_data, wikis, embedding_class, sen_sele_model, 1)
        sen_sele_class.train()
        print("cla veri")
        cla_veri_class = cla_veri.cla_veri_train(train_data, wikis, embedding_class, doc_retr_model, sen_sele_model,
                                                 cla_veri_model, 1)
        cla_veri_class.train()
        for iter in range(argv.get("iters",default_argv["iters"]) if argv is not None else default_argv["iters"]):
            print("%dth iteration of adversarial sample training" % iter)
            print("Generating doc retr adversarial samples ...")
            doc_retr_class.add_adversarial_data()
            print("Generating sen sele adversarial samples ...")
            sen_sele_class.add_adversarial_data()
            print("Generating cla veri adversarial samples ...")
            cla_veri_class.add_adversarial_data()
            print("Adversarial sample adding complete")
            print("Retrain ...")
            print("doc retr")
            doc_retr_class.train(adversarial_train=True)
            print("sen sele")
            sen_sele_class.train(adversarial_train=True)
            print("cla veri")
            cla_veri_class.train(adversarial_train=True)



if __name__ == "__main__":
    pipeline("debug")



