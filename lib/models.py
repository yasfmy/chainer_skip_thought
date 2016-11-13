import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer import link
from chainer.links.connection import linear

import numpy as np

class BaseModel(chainer.Chain):
    def use_gpu(self, gpu_id):
        cuda.get_device(gpu_id).use()
        self.to_gpu()

class ConditionalStatefulGRU(link.Chain):
    def __init__(self, n_units, n_inputs=None, n_conditions=None, init=None,
                inner_init=None, bias_init=0):
        if n_inputs is None:
            n_inputs = n_units
            n_conditions = n_units
        super().__init__(
                W_r=linear.Linear(n_inputs, n_units,
                    initialW=init, initial_bias=bias_init),
                U_r=linear.Linear(n_units, n_units,
                    initialW=inner_init, initial_bias=bias_init),
                C_r=linear.Linear(n_conditions, n_units,
                    initialW=None, initial_bias=bias_init),
                W_z=linear.Linear(n_inputs, n_units,
                    initialW=init, initial_bias=bias_init),
                U_z=linear.Linear(n_units, n_units,
                    initialW=inner_init, initial_bias=bias_init),
                C_z=linear.Linear(n_conditions, n_units,
                    initialW=None, initial_bias=bias_init),
                W=linear.Linear(n_inputs, n_units,
                    initialW=init, initial_bias=bias_init),
                U=linear.Linear(n_units, n_units,
                    initialW=inner_init, initial_bias=bias_init),
                C=linear.Linear(n_conditions, n_units,
                    initialW=None, initial_bias=bias_init),
                )
        self.reset_state()

    def to_cpu(self):
        super().to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super().to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, chainer.Variable)
        h_ = h
        if self.xp == np:
            h_.to_cpu()
        else:
            h_.to_gpu()
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x, condition):
        z = self.W_z(x)
        h_bar = self.W(x)
        if self.h is not None:
            r = sigmoid.sigmoid(self.W_r(x) + self.U_r(self.h) + self.C_r(condition))
            z += self.U_z(self.h)
            h_bar += self.U(r * self.h)
        z = sigmoid.sigmoid(z + self.C_z(condition))
        h_bar = tanh.tanh(h_bar + self.C(condition))

        h_new = z * h_bar
        if self.h is not None:
            h_new += (1 - z) * self.h
        self.h = h_new
        return self.h

class SkipThought(BaseModel):
    def __init__(self,
                 source_vocabulary_size,
                 embed_size,
                 hidden_size,
                 target_vocabulary_size=None):
        if target_vocabulary_size is None:
            target_vocabulary_size = source_vocabulary_size
        super().__init__(
            source_embed = L.EmbedID(source_vocabulary_size, embed_size),
            encoder=L.StatefulGRU(embed_size, hidden_size),
            target_embed=L.EmbedID(target_vocabulary_size, embed_size),
            previous_decoder=ConditionalStatefulGRU(embed_size, hidden_size, hidden_size),
            next_decoder=ConditionalStatefulGRU(embed_size, hidden_size, hidden_size),
            previous_output_weight=L.Linear(hidden_size, target_vocabulary_size),
            next_output_weight=L.Linear(hidden_size, target_vocabulary_size),
        )

    def encode(self, source):
        for x in source:
            word_embedding = self.source_embed(x)
            condition = self.encoder(word_embedding)
        return condition

    def decode_once(self, previous_y, next_y, condition):
        previous_word_embedding = self.target_embed(previous_y)
        next_word_embedding = self.target_embed(next_y)
        previous_h = self.previous_decoder(previous_word_embedding, condition)
        next_h = self.next_decoder(next_word_embedding, condition)
        previous_y = self.previous_output_weight(previous_h)
        next_y = self.next_output_weight(next_h)
        return previous_y, next_y

    @staticmethod
    def loss(previous_y, previous_t, next_y, next_t):
        return (F.softmax_cross_entropy(previous_y, previous_t) +
                F.softmax_cross_entropy(next_y, next_t))

    def forward_train(self, source, previous_target, next_target):
        condition = self.encode(source)
        loss = 0
        for previous_y, previous_t, next_y, next_t in zip(
            previous_target, previous_target[1:], next_target, next_target[1:]):
            previous_y, next_y = self.decode_once(previous_y, next_y, condition)
            loss += self.loss(previous_y, previous_t, next_y, next_t)
        return loss
