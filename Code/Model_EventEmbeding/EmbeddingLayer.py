#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 23:41:51 2017

@author: red-sky
"""


import numpy as np
import theano
from theano import tensor as T


class EmbeddingLayer(object):
    def __init__(self, num_vocab, word_dim, rng, embedding_w=None):
        '''
        word_dim :: dimension of the word embeddings
        num_vocab :: number of word embeddings in the vocabulary
        embedding_w :: pre-train word vector
        '''

        if embedding_w is None:
            word_vectors = rng.uniform(-1.0, 1.0, (num_vocab, word_dim))
            self.embedding_w = theano.shared(word_vectors,
                                             name="EmbeddingLayer_W") \
                .astype(theano.config.floatX)
        else:
            self.embedding_w = theano.shared(embedding_w,
                                             name="EmbeddingLayer_W") \
                .astype(theano.config.floatX)

        self.params = [self.embedding_w]
        self.infor = [num_vocab, word_dim]

    def words_ind_2vec(self, index):
        map_word_vectors = self.embedding_w[index]
        output = T.mean(map_word_vectors,  axis=0)
        return output, map_word_vectors


if __name__ == "__main__":
    rng = np.random.RandomState(220495)
    arrWords = T.ivector("words")
    EMBD = EmbeddingLayer(100, 150, rng=rng)
    Word2Vec = theano.function(
        inputs=[arrWords],
        outputs=EMBD.words_ind_2vec(arrWords)
    )
    Vec = Word2Vec([1, 2, 3, 4])
    Vec = Word2Vec([2, 3, 4])
    print("Dim: ", Vec.shape)
    print("Val: ", Vec)
