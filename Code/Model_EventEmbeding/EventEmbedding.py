#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:36:33 2017

@author: red-sky
"""
import sys
import json
import theano
import pickle
import os.path
import numpy as np
import theano.tensor as T
from SmallUtils import createShareVar, ADAM_OPTIMIZER
from EmbeddingLayer import EmbeddingLayer
from RoleDependentLayer import RoleDependentLayer


class Input(object):
    def __init__(self, object1, object1_fake, action, object2, rng,
                 vovab_length=4000, wordDim=100, trainedWordsVectors=None,):
        # Init Embeding layer, input vector of index and ouput average
        # of word vector as ref Ding et al 2014
        self.EMBD = EmbeddingLayer(vovab_length, wordDim, rng=rng,
                                   embedding_w=trainedWordsVectors)

        object1_vector, _ = self.EMBD.words_ind_2vec(object1)
        action_vector, _ = self.EMBD.words_ind_2vec(action)
        object2_vector, _ = self.EMBD.words_ind_2vec(object2)
        object1_vector_fake, _ = self.EMBD.words_ind_2vec(object1_fake)

        self.output = [object1_vector, object1_vector_fake,
                       action_vector, object2_vector]
        self.params = self.EMBD.params

    def get_params(self):
        trainParams = {
            "WordWvec": self.EMBD.embedding_w.get_value()
        }
        return(trainParams)


class ModelBody(object):
    def __init__(self, vectorObjects, rng, n_out, n_in,
                 trainedModelParams=None):
        if trainedModelParams is None:
            trainedModelParams = {
                "roleDependentLayer1_": {
                    "T": None, "W1": None, "W2": None, "b": None
                },
                "roleDependentLayer2_": {
                    "T": None, "W1": None, "W2": None, "b": None
                },
                "roleDependentLayer3_": {
                    "T": None, "W1": None, "W2": None, "b": None
                }
            }

        Obj1, Ob1_fake, Act, Obj2 = vectorObjects

        self.RoleDepen1 = RoleDependentLayer(
            left_dependent=T.stack([Obj1, Ob1_fake], axis=0),
            right_dependent=Act,
            n_in=n_in, n_out=n_out, rng=rng,
            trainedParams=trainedModelParams,
            name="roleDependentLayer1_"
        )
        self.RoleDepen1_output = self.RoleDepen1.output

        self.RoleDepen2 = RoleDependentLayer(
            left_dependent=Obj2,
            right_dependent=Act,
            n_in=n_in, n_out=n_out, rng=rng,
            trainedParams=trainedModelParams,
            name="roleDependentLayer2_"
        )
        self.RoleDepen2_output = T.flatten(self.RoleDepen2.output, outdim=1)

        self.RoleDepen3 = RoleDependentLayer(
            left_dependent=self.RoleDepen1_output,
            right_dependent=self.RoleDepen2_output,
            n_in=n_out, n_out=n_out, rng=rng,
            trainedParams=trainedModelParams,
            name="roleDependentLayer3_"
        )

        self.params = self.RoleDepen1.params + self.RoleDepen2.params + \
            self.RoleDepen3.params

        self.L2 = (
            self.RoleDepen1.L2 +
            self.RoleDepen2.L2 +
            self.RoleDepen3.L2
        )
        self.output = self.RoleDepen3.output

    def get_params(self):
        trainedModelParams = {
            "roleDependentLayer1_": self.RoleDepen1.get_params(),
            "roleDependentLayer2_": self.RoleDepen2.get_params(),
            "roleDependentLayer3_": self.RoleDepen3.get_params()
        }
        return(trainedModelParams)


class LogisticRegression(object):

    def __init__(self, rng, layerInput, n_in, n_out,
                 paramsLayer=None,
                 name="LogisticRegression_"):

        self.layerInput = layerInput
        if paramsLayer is None:
            self.W = createShareVar(rng=rng, name=name+"W",
                                    factor_for_init=n_out + n_in,
                                    dim=(n_in, n_out))
        else:
            self.W = theano.shared(value=paramsLayer["W"],
                                   name=name+"W", borrow=True)

        if paramsLayer is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values,
                                   name=name+"b", borrow=True)
        else:
            self.b = theano.shared(value=paramsLayer["b"],
                                   name=name+"b", borrow=True)

        step1 = T.dot(self.layerInput, self.W)
        self.prob_givenX = T.tanh(step1 + self.b)
        self.y_predict = T.argmax(self.prob_givenX, axis=1)

        self.params = [self.W, self.b]
        self.L2 = sum([(param**2).sum() for param in self.params])

    def get_params(self):
        trainedParams = {
            "W": self.W.get_value(), "b": self.b.get_value()
        }
        return(trainedParams)

    def neg_log_likelihood(self, y_true):
        y_true = T.cast(y_true, "int32")
        log_prob = T.log(self.prob_givenX)
        nll = -T.mean(log_prob[T.arange(y_true.shape[0]), y_true])
        return nll

    def margin_loss(self):
        loss = T.max([0, 1 - self.prob_givenX[0, 0] + self.prob_givenX[1, 0]])
        return loss

    def cal_errors(self, y_true):
        if y_true.ndim != self.y_predict.ndim:
            raise TypeError(
                "y should have the same shape as self.y_pred",
                ("y_true", y_true.ndim, "y_pred", self.y_predict.ndim)
            )
        if y_true.dtype.startswith("int"):
            return T.mean(T.neq(self.y_predict, y_true))
        else:
            raise TypeError(
                "y_true should have type int ...",
                ("y_true", y_true.type, "y_pred", self.y_predict.type)
            )


def main(dataPath, trainedParamsPath="modelTrained.pickle",
         outputVectorPath="resultEmbeding.pickle",
         learning_rate=0.005, L2_reg=0.0001,
         n_epochs=500, num_K=150, word_dim=150):
    # CONSTANT VARIABLES
    RNG = np.random.RandomState(220495 + 280295 + 1)
    LABEL_NUM = 2
    if os.path.isfile(trainedParamsPath):
        with open(trainedParamsPath, 'rb') as handle:
            trainedParams = pickle.load(handle)
    else:
        print("No Trained Model, create new")
        trainedParams = {
            "Input": {"WordWvec": None}, "Body": None, "Output": None
        }

    OPTIMIZER = ADAM_OPTIMIZER
    # INPUT DATA
    data_indexed_events = np.load(dataPath)
    N_sample = len(data_indexed_events)
#    N_sample = 1
    all_index = list(set(np.hstack(data_indexed_events.flat)))
#    all_train_index = list(set(np.hstack(data_indexed_events[0:NNN].flat)))
    # Snip tensor at begin
    object1 = T.ivector("object1")
    object1_fake = T.ivector("object1_fake")
    action = T.ivector("action")
    object2 = T.ivector("object2")

    constainY = theano.shared(
        np.asarray([1, 0], dtype=theano.config.floatX),
        borrow=True
    )

    # WORDS EMBEDING VECTOR
    wordsEmbedLayer = Input(
        object1=object1, object1_fake=object1_fake,
        action=action, object2=object2, rng=RNG,
        wordDim=word_dim, vovab_length=len(all_index),
        trainedWordsVectors=trainedParams["Input"]["WordWvec"]
    )

    Obj1, Ob1_fake, Act, Obj2 = wordsEmbedLayer.output

    # EVENTS EMBEDING LAYER - THREE ROLE DEPENTDENT LAYER
    eventsEmbedingLayer = ModelBody(
        vectorObjects=wordsEmbedLayer.output,
        n_out=num_K, n_in=word_dim, rng=RNG,
        trainedModelParams=trainedParams["Body"]
    )

    # CLASSIFY LAYER
    predict_layers = LogisticRegression(
        layerInput=eventsEmbedingLayer.output,
        rng=RNG, n_in=num_K, n_out=1,
        paramsLayer=trainedParams["Output"]
    )

    # COST FUNCTION
    COST = (
        predict_layers.margin_loss() +
        L2_reg * predict_layers.L2 +
        L2_reg * eventsEmbedingLayer.L2
    )

    # GRADIENT CALCULATION and UPDATE
    all_params = wordsEmbedLayer.params + \
        eventsEmbedingLayer.params + predict_layers.params
    print("TRAIN: ", all_params)

    UPDATE = OPTIMIZER(COST, all_params, learning_rate=learning_rate)

    # TRAIN MODEL
    GET_COST = theano.function(
        inputs=[object1, object1_fake, action, object2],
        outputs=[predict_layers.margin_loss(),
                 predict_layers.prob_givenX],
    )

#    TEST = theano.function(
#        inputs=[object1, object1_fake, action, object2],
#        outputs=eventsEmbedingLayer.RoleDepen2.test,
#        on_unused_input='warn'
#    )

    TRAIN = theano.function(
        inputs=[object1, object1_fake, action, object2],
        outputs=[predict_layers.margin_loss()],
        updates=UPDATE
    )

    GET_EVENT_VECTOR = theano.function(
        inputs=[object1, object1_fake, action, object2],
        outputs=[predict_layers.margin_loss(),
                 eventsEmbedingLayer.output],
    )

    def generate_fake_object(all_index, RNG, obj):
        fake_obj = list(RNG.choice(all_index, len(obj)))
        while sorted(fake_obj) == sorted(obj):
            print("WRONG faking object 1", obj)
            fake_obj = list(RNG.choice(all_index, len(obj)))
        return(fake_obj)

    def generate_list_object(data_indexed_events, all_index, RNG):
        list_fake_object1 = [
            generate_fake_object(all_index, RNG, events[0])
            for events in data_indexed_events
        ]
        list_real_object = set([
            "_".join([str(a) for a in sorted(events[0])])
            for events in data_indexed_events
        ])
        wrong = 0
        while True:
            valid = True
            wrong += 1
            for i, obj in enumerate(list_fake_object1):
                s = "_".join([str(a) for a in sorted(obj)])
                if s in list_real_object:
                    valid = valid and False
                    list_fake_object1[i] = \
                        generate_fake_object(all_index, RNG, s)
                else:
                    valid = valid and True
            if valid:
                break
        print("There are %d wrong random loops" % wrong)
        return(list_fake_object1)

    print("*"*72)
    print("Begin Training process")

    for epoch in range(n_epochs):
        # create false label
        print("Begin new epoch: %d" % epoch)

        list_fake_object1 = generate_list_object(data_indexed_events,
                                                 all_index, RNG)
        cost_of_epoch = []
        set_index = set(range(N_sample))
        temp_variable = N_sample
        print("*" * 72+"\n")
        print("*" * 72+"\n")
        # train
        model_train = {
            "Input": wordsEmbedLayer.get_params(),
            "Body": eventsEmbedingLayer.get_params(),
            "Output": predict_layers.get_params()
        }
        RESULT = {}
        outCOST = []
        Max_inter = len(set_index)*2
        iter_num = 0
        while len(set_index) > 0 and iter_num <= Max_inter:
            iter_num += 1
            index = set_index.pop()
            ob1_real, act, obj2 = data_indexed_events[index]
            ob1_fake = list_fake_object1[index]
            cost, probY = GET_COST(ob1_real, ob1_fake, act, obj2)
            outCOST.append(cost)
#            test = TEST(ob1_real, ob1_fake, act, obj2)
#            for a in test:
#                print(a, a.shape)

            if cost > 0:
                set_index.add(index)
                c = TRAIN(ob1_real, ob1_fake, act, obj2)
            else:
                RESULT[index] = GET_EVENT_VECTOR(ob1_real, ob1_fake, act, obj2)

            if (len(set_index) % 50 == 0 and
                    temp_variable != len(set_index)):
                temp_variable = len(set_index)
                print("There are %f %% left in this %d "
                      "epoch with average cost %f"
                      % (len(set_index)/float(N_sample)*100,
                         epoch, np.mean(outCOST[-50:])))
            if iter_num > Max_inter - 5:
                print(set_index, ob1_real, ob1_fake, act, obj2)

        with open(trainedParamsPath, 'wb') as handle:
            pickle.dump(model_train, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        with open(outputVectorPath, 'wb') as handle:
            pickle.dump(RESULT, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
#    arg = ["", "Data/Query_Apple/2005-2010/IndexedEvents.npy",
#           "Data/Query_Apple/2005-2010/linhtinh/", "20"]
    arg = sys.argv
    main(dataPath=arg[1], trainedParamsPath=arg[2]+"TrainedParams.pickle",
         outputVectorPath=arg[2]+"resultEmbeding.pickle", n_epochs=int(arg[3]))

