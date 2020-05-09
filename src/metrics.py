


def acc_10(labels): 
    acc = list()
    for i in range(1,31):
        acc.append(acc_k(labels,i))
    return acc 

def acc_k(labels,k):
    return sum(labels[0:k])/k
