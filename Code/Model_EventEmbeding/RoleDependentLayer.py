#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:13:18 2017

@author: red-sky
"""

import theano
import numpy as np
import theano.tensor as T
from SmallUtils import createShareVar


class RoleDependentLayer(object):
    def __init__(self, left_dependent, right_dependent, rng,
                 n_in=100, n_out=4, trainedParams=None,
                 name="RoleDependentEmbedding_"):
        if trainedParams is None:
            trainedParams = {
                name: {
                    "T": None, "W1": None, "W2": None, "b": None
                }
            }

        if trainedParams[name]["T"] is not None:
            assert trainedParams[name]["T"].shape == (n_out, n_in, n_in)
            self.T = theano.shared(value=trainedParams[name]["T"],
                                   name=name+"T", borrow=True)
        else:
            self.T = createShareVar(rng=rng, name=name+"T",
                                    factor_for_init=n_out + n_in,
                                    dim=(n_out, n_in, n_in))

        if trainedParams[name]["W1"] is not None:
            assert trainedParams[name]["W1"].shape == (n_in, n_out)
            self.W1 = theano.shared(value=trainedParams[name]["W1"],
                                    name=name+"W1", borrow=True)
        else:
            self.W1 = createShareVar(rng=rng, name=name+"W1",
                                     factor_for_init=n_out + n_in,
                                     dim=(n_in, n_out))

        if trainedParams[name]["W2"] is not None:
            assert trainedParams[name]["W2"].shape == (n_in, n_out)
            self.W2 = theano.shared(value=trainedParams[name]["W2"],
                                    name=name+"W2", borrow=True)
        else:
            self.W2 = createShareVar(rng=rng, name=name+"W2",
                                     factor_for_init=n_out + n_in,
                                     dim=(n_in, n_out))

        if trainedParams[name]["b"] is not None:
            assert trainedParams[name]["b"].shape == (n_out,)
            self.b = theano.shared(value=trainedParams[name]["b"],
                                   name=name+"b", borrow=True)
        else:
            b_values = np.zeros(shape=(n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name=name+"b", borrow=True)

        # list of layer params
        self.params = [self.T, self.W1, self.W2, self.b]

        # L2 regulation
        self.L2 = sum([(param**2).sum() for param in self.params])

        # Bi-linear step
        def one_kernel(Tk, left, right):
            first_bi_libear = theano.dot(left, Tk)
            seccon_bi_linear = theano.dot(first_bi_libear, right)
            return(seccon_bi_linear.flatten())

        bi_1, _ = theano.scan(
            fn=one_kernel,
            sequences=[self.T],
            non_sequences=[left_dependent, right_dependent],
            n_steps=n_out
        )

        # Feed forward network step
        feedforward_step1 = theano.dot(left_dependent, self.W1)
        feedforward_step2 = theano.dot(right_dependent, self.W2)
        feedforward_step3 = (feedforward_step1 +
                             feedforward_step2.dimshuffle("x", 0) +
                             self.b.dimshuffle("x", 0))
        feedforward_step4 = bi_1.dimshuffle(1, 0) + feedforward_step3
        self.output = theano.tensor.tanh(feedforward_step4)
        self.test = [feedforward_step3]

    def output_(self, left_dependent, right_dependent):

        def one_kernel(Tk, left, right):
            first_bi_libear = theano.dot(left, Tk)
            seccon_bi_linear = theano.dot(first_bi_libear, right)
            return(seccon_bi_linear.flatten())

        bi_linear_tensor, _ = theano.scan(
            fn=one_kernel,
            sequences=[self.T],
            non_sequences=[left_dependent, right_dependent],
            n_steps=n_out
        )

        bi_linear_tensor = bi_linear_tensor.dimshuffle(1, 0)
        feedforward_step1 = theano.dot(left_dependent, self.W1)
        feedforward_step2 = theano.dot(right_dependent, self.W2)
        feedforward_step3 = (feedforward_step1 +
                             feedforward_step2.dimshuffle("x", 0) +
                             self.b.dimshuffle("x", 0))
        feedforward_step4 = bi_linear_tensor + feedforward_step3
        output = theano.tensor.tanh(feedforward_step4)
        return(output)

    def get_params(self):
        trainedParams = {
            "T": self.T.get_value(), "W1": self.W1.get_value(),
            "W2": self.W2.get_value(), "b": self.b.get_value()
        }
        return(trainedParams)
