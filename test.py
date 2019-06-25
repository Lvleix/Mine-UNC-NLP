import os

spath = os.path.dirname(__file__)
mpath = os.path.dirname(spath)
wpath = mpath + "/wiki-pages"
print(spath)
print(mpath)
print(wpath)

if __name__ == "__main__":

    files = os.listdir(wpath)
    wikis = {}
    for file in files:
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
                wikis[wiki["id"]] = [sen.split("\t")[1] if len(sen.split("\t"))>=2 else "" for sen in lines.split("\n")]
        f.close()
    print("wiki dict completed. ")
    #os.system("pause")
    #with open(mpath+"/wiki.json", "w") as f:
    #    f.write(str(wikis))




