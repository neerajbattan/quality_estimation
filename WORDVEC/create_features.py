import numpy as np
from collections import Counter
import sys
import copy

src_sents = open(sys.argv[1]).read()
mt_sents = open(sys.argv[3]).read()
src_vecs = open(sys.argv[2]).read()
mt_vecs = open(sys.argv[4]).read()

src_vecs = src_vecs.split('\n')
mt_vecs = mt_vecs.split('\n')

def createVecs(text):
    vecs = {}
    for i in text:
        word = i.split()
        if len(word) == 0:
            continue 
        temp = word[1:]
        temp = np.array(temp,dtype='float64')
        word = word[0]
        vecs[word] = temp
    return vecs

def sentVecs(sent, vecs):
    siz = vecs[next(iter(vecs))].shape
    sum = np.zeros(siz,dtype='float64')
    if len(sent) == 0:
        return sum
    
    for i in sent:
        if i in vecs:
            sum += vecs[i]
    sum = sum / len(sent)
    return sum

def createSent(text, vecs):
    temp = []
    for i in text.split('\n'):
        if len(i)==0:
            continue
        retVec = sentVecs(i.split(), vecs)
        temp.append(retVec)
        '''
        if temp.shape!=(0,):
            temp.append(retVec)
        else:
            print "hi"
            temp = copy.copy(retVec)
        '''
    temp = np.array(temp)
    return temp

src_vecs = createVecs(src_vecs)
mt_vecs = createVecs(mt_vecs)
mt_sentvecs = createSent(mt_sents,mt_vecs)
src_sentvecs = createSent(src_sents,src_vecs) 

#mt_sentvecs.dump('mt_vecs')
#src_sentvecs.dump('src_vecs')



final = np.concatenate([src_sentvecs,mt_sentvecs],axis=1)
final.dump(sys.argv[5])
