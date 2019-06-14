from data.Vocab import *
import numpy as np
import torch
from torch.autograd import Variable
import random

def read_corpus(file_path, vocab=None):
    data = []
    tgt_type = parse_my_conll(file_path)
    with open(file_path, 'r') as infile:
        for sentence in readDepTree(infile,tgt_type, vocab):
            data.append(sentence)
    return data

def sentences_numberize(sentences, vocab):
    for sentence in sentences:
        yield sentence2id(sentence, vocab)

def sentence2id(sentence, vocab):
    result = []
    for dep in sentence.sentence:
        wordid = vocab.word2id(dep.form)
        extwordid = vocab.extword2id(dep.form)
        tagid = vocab.tag2id(dep.tag)
        head = dep.head
        if head == -1:
            relid = -1
        elif head >= 0:
            relid = vocab.rel2id(dep.rel)
        result.append([wordid, extwordid, tagid, head, relid])

    return result



def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def data_corpus_weighting(data1, data2, domain_num_corpus_weighting,cdt_domain_ration):
    batch_data1=random.sample(data1,domain_num_corpus_weighting)
    data2_size=domain_num_corpus_weighting*cdt_domain_ration
    batch_data2 = random.sample(data2, data2_size)
    return batch_data1,batch_data2

def data_iter_from_two_data(data1, data2, batch_size, shuffle=True):
    type1 = data1[0].tgt_type
    type2 = data2[0].tgt_type
    assert(type1 != type2)
    data = data1 + data2
    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        batch1 = []
        batch2 = []
        for instance in batch:
            if instance.tgt_type == type1:
                batch1.append(instance)
            elif instance.tgt_type == type2:
                batch2.append(instance)
            else:
                print("there is error in corpus")
                exit(0)
        yield batch1, batch2
def batch_data_variable(batch, vocab):
    length = len(batch[0])
    batch_size = len(batch)
    for b in range(1, batch_size):
        if len(batch[b]) > length: length = len(batch[b])

    words = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    extwords = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    tags = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    masks = Variable(torch.Tensor(batch_size, length).zero_(), requires_grad=False)
    heads = []
    rels = []
    lengths = []

    b = 0
    for sentence in sentences_numberize(batch, vocab):
        index = 0
        length = len(sentence)
        lengths.append(length)
        head = np.zeros((length), dtype=np.int32)
        rel = np.zeros((length), dtype=np.int32)
        for dep in sentence:
            words[b, index] = dep[0]
            extwords[b, index] = dep[1]
            tags[b, index] = dep[2]
            head[index] = dep[3]
            rel[index] = dep[4]
            masks[b, index] = 1
            index += 1
        b += 1
        heads.append(head)
        rels.append(rel)

    return words, extwords, tags, heads, rels, lengths, masks

def batch_variable_depTree(trees, heads, rels, lengths, vocab):
    for Tree, head, rel, length in zip(trees, heads, rels, lengths):
        tree = Tree.sentence
        sentence = []
        for idx in range(length):
            sentence.append(Dependency(idx, tree[idx].org_form, tree[idx].tag, head[idx], vocab.id2rel(rel[idx])))
        yield sentence



