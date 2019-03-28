import numpy as np
import codecs
word2idx = np.load("x2_w2i.npy")
word2idx = word2idx.item()
id2word = {k: v for v, k in zip(word2idx.keys(), word2idx.values())}


def translate(word_indexs):
    words = []
    word_indexs=word_indexs.strip()
    for idx in word_indexs.split(' '):
        idx = int(idx)
        word = id2word.get(idx)
        
        if word != '0':
            words.append(id2word.get(idx))
        # else:
        #     words.append("<UNK>")
    return "".join(words)

with codecs.open('_Nucleus.txt','w',encoding='utf8') as f,open('Nucleus.txt') as g:
    for line in g:
        lines=translate(line)
        f.write(lines)
        f.write('\n')

