import os
from phases import doc_retr
from phases import sen_sele
from phases import cla_veri

mode = "test"  # "train","infer","test"
spath = os.path.dirname(__file__)
mpath = os.path.dirname(spath)
wpath = mpath + "/wiki-pages"

print("read in wikis ... ")
files = os.listdir(wpath)
wikis = {}
for file in files:
    """
    if file not in ["wiki-001.jsonl"]:
        continue
    """
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

with open(mpath+"/train.jsonl","r") as f:
#with open(spath + "/new_train_set.jsonl", "r") as f:
    train_data = [eval(line.replace("null","''")) for line in f]

with open(mpath+"/shared_task_dev.jsonl","r") as f:
#with open(spath + "/new_dev_set.jsonl", "r") as f:
    dev_data = [eval(line.replace("null","''")) for line in f]

if mode == "train":
    doc_retr_model = ""
    sen_sele_model = ""
    doc_retr_class = doc_retr.doc_retr_train(train_data, wikis)
    doc_retr_class.train()
    del doc_retr_class
    sen_sele_class = sen_sele.sen_sele_train(train_data,wikis)
    sen_sele_class.train()
    del sen_sele_class
    cla_veri_class = cla_veri.cla_veri_train(train_data,wikis,doc_retr_model,sen_sele_model)
    cla_veri_class.train()
    del cla_veri_class
elif mode == "infer":
    claim = input()
    doc_retr_model = ""
    sen_sele_model = ""
    cla_veri_model = ""
    doc_retr_class = doc_retr.doc_retr_infer(wikis,doc_retr_model)
    sen_sele_class = sen_sele.sen_sele_infer(wikis,sen_sele_model)
    cla_veri_class = cla_veri.cla_veri_infer(wikis,cla_veri_model)
    docs = doc_retr_class.infer(claim)
    sens = sen_sele_class.infer(claim,docs)
    result = cla_veri_class.infer(claim,sens)
    print({0:"SUPPORTS",1:"REFUTES",2:"NOT ENOUGH INFO"}[result])
    print(sens)
elif mode == "test":
    doc_retr_model = ""
    sen_sele_model = ""
    cla_veri_model = ""
    doc_retr_class = doc_retr.doc_retr_infer(wikis, doc_retr_model)
    sen_sele_class = sen_sele.sen_sele_infer(wikis, sen_sele_model)
    cla_veri_class = cla_veri.cla_veri_infer(wikis, cla_veri_model)
    counts = 0
    rights = 0
    for test in dev_data:
        counts += 1
        claim = test["claim"]
        docs = doc_retr_class.infer(claim)
        sens = sen_sele_class.infer(claim, docs)
        result = cla_veri_class.infer(claim, sens)
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



