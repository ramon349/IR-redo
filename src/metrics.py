import pdb 

def acc_all(labels): 
    acc = list()
    for i in range(1,31):
        acc.append(acc_k(labels,i))
    return acc 

def acc_k(labels,k):
    return sum(labels[0:k])/k

def recall_k(labels,key,counter,k):
    return sum(labels[0:k])/counter[key]
def recall_all(labels,key,counter): 
    rec = list()
    for i in range(1,31):
        rec.append(recall_k(labels,key,counter,i) )
    return rec
