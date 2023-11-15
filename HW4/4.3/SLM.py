import numpy as np
from operator import itemgetter
import math
import matplotlib.pyplot as plt

with open('4.3\hw4_unigram.txt','r') as file:
    uniCount = file.readlines()
    file.close()
with open('4.3\hw4_vocab.txt','r') as file:
    vocab = file.readlines()
    file.close()
with open('4.3\hw4_bigram.txt','r') as file:
    biCount = file.readlines()
    file.close()

for i in range (len(vocab)):
    word = vocab[i].strip()
    word_count = int(uniCount[i].strip())
    vocab[i],uniCount[i] = word, word_count
uniCount = np.array(uniCount)
prob = uniCount/np.sum(uniCount)

#(a)
prob_dic = {}
for i in range(len(vocab)):
    prob_dic[vocab[i]] = float(prob[i])
    if vocab[i][0] == 'M' or vocab[i][0] == 'm':
        print(vocab[i]," ",str(prob[i]))

#b)
biArr = []
for line in biCount:
    parts = line.strip().split()
    biArr.append(parts)
bi_dict_the = {}
for line in biArr:
    if(line[0] == '4'):
        spot = int(line[1]) -1 
        bi_dict_the[vocab[spot]] = float(float(line[2])/ float(uniCount[3]))
res = dict(sorted(bi_dict_the.items(), key=itemgetter(1), reverse=True)[:10])


# c)
def getSentenceIndex(sentence):
    sentence = sentence.split()
    index_sentence = []
    for word in sentence:
        index_sentence.append(vocab.index(word))
    return index_sentence
sentenceC = getSentenceIndex("THE STOCK MARKET FELL BY ONE HUNDRED POINTS LAST WEEK")
def getUnigram(arr):
    answer = 1
    for num in arr:
        answer *= float(prob[num])
    return answer
print(math.log(getUnigram(sentenceC)))

def getBiGramFixed(arr):
    denom = uniCount[arr[0]]
    for line in biArr:
            if int(line[0]) == arr[0] +1 and int(line[1]) == arr[1] + 1:
                num = int(line[2])
                return num/denom
    return 0
def getBiGram(arr):
    answer = 1
    count = 0
    for i in range(len(arr)):
        
        index = 0
        denom = 0
        num = 0
        if i == 0:
            index = vocab.index("<s>")
            denom = int(uniCount[index+1])
        else:
            index = arr[i-1] 
            denom = int(uniCount[index])
        for line in biArr:
            if int(line[0]) == index+1 and int(line[1]) == arr[count] + 1:
                num = int(line[2])
                break
        answer *= (num/denom)
        count +=1
    return answer
print(math.log(getBiGram(sentenceC)))

# d)
sentenceD = getSentenceIndex("THE SIXTEEN OFFICIALS SOLD FIRE INSURANCE")
print(math.log(getUnigram(sentenceD)))
#getBiGram(sentenceD) is undefined, because of the pairs that are not observed

#e 

def getMixProb(lam, w_prime, w):
    p1 = lam* (getUnigram(w_prime))
    p2 = (1-lam)*(getBiGramFixed(w))
    return  p1 + p2

mixList =[]
lambdas = np.linspace(0.00001, 0.99999, 1000)
mix = {}
for lam in lambdas:
    probs = 1
    for i in range(len(sentenceD)):
        if i == 0:
            probs *= getMixProb(lam, [sentenceD[i]], [1,sentenceD[i]])
        else:
            probs *= getMixProb(lam, [sentenceD[i]], [sentenceD[i-1], sentenceD[i]])
    mixList.append((math.log(probs)))
    mix[lam] = math.log(probs)
print(mixList)

plt.plot(lambdas,mixList,'b')
plt.xlabel('lamda', fontdict={'family' : 'Times New Roman', 'size'   : 12})
plt.ylabel('log likihood', fontdict={'family' : 'Times New Roman', 'size'   : 12})
plt.show()

print(getBiGram())

