#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:55:14 2017

@author: red-sky
"""
import theano
import theano.tensor as T
import numpy as np

def createShareVar(rng, dim, name, factor_for_init):
    var_values = np.asarray(
        rng.uniform(
            low=-np.sqrt(6.0 / factor_for_init),
            high=np.sqrt(6.0 / factor_for_init),
            size=dim,
        )
    )
    Var = theano.shared(value=var_values, name=name, borrow=True)
    return Var


def adadelta(lr, tparams, cost, grads, listInput):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres

    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """
    np_float = np.asarray(0., dtype=theano.config.floatX)
    zipped_grads = [theano.shared(p.get_value() * np_float,
                                  name='%s_grad' % k)
                    for k, p in enumerate(tparams)]
    running_up2 = [theano.shared(p.get_value() * np_float,
                                 name='%s_rup2' % k)
                   for k, p in enumerate(tparams)]
    running_grads2 = [theano.shared(p.get_value() * np_float,
                                    name='%s_rgrad2' % k)
                      for k, p in enumerate(tparams)]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inputs=listInput,
                                    outputs=cost,
                                    updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams, updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def ADAM_OPTIMIZER(loss, all_params, learning_rate=0.001,
                   b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8):
    """
    CITE: http://sebastianruder.com/optimizing-gradient-descent/index.html#adam
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    # (Decay the first moment running average coefficient)
    b1_t = b1*gamma**(t-1)

    for params_previous, g in zip(all_params, all_grads):
        init_moment = np.zeros(params_previous.get_value().shape,
                               dtype=theano.config.floatX)
        # (the mean)
        first_moment = theano.shared(init_moment)
        # (the uncentered variance)
        second_moment = theano.shared(init_moment)

        # (Update biased first moment estimate)
        bias_m = b1_t*first_moment + (1 - b1_t)*g

        # (Update biased second raw moment estimate)
        bias_v = b2*second_moment + (1 - b2)*g**2

        # (Compute bias-corrected first moment estimate)
        unbias_m = bias_m / (1-b1**t)

        # (Compute bias-corrected second raw moment estimate)
        unbias_v = bias_v / (1-b2**t)

        # (Update parameters)
        update_term = (alpha * unbias_m) / (T.sqrt(unbias_v) + e)
        params_new = params_previous - update_term

        updates.append((first_moment, bias_m))
        updates.append((second_moment, bias_v))
        updates.append((params_previous, params_new))
    updates.append((t, t + 1.))
    return updates