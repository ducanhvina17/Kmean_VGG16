import glob
import random
folder=(glob.glob("images/*"))
traindata=open("database/train.txt","w")
testdata=open("database/test.txt","w")
text=0
for img in folder:
    text+=1
    listimg=glob.glob(img+"/*")
    lenlist=len(listimg)
    trainlist=random.sample(listimg,int(lenlist*0.7))
    testlist= list(set(listimg) - set(trainlist)) 
    for item in trainlist:
        traindata.write(str(item)+"\n")
    for item in testlist:
        testdata.write(str(item)+"\n")
traindata.close()         
testdata.close()         
print(text)
