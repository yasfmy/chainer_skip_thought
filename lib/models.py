from itertools import chain

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer import link
from chainer.links.connection import linear

class BaseModel(chainer.Chain):
    def use_gpu(self, gpu_id):
        cuda.get_device(gpu_id).use()
        self.to_gpu()

    def _prepare_input(self, x, dtype=None):
        return chainer.Variable(self.xp.array(x, dtype=dtype))

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
            previous_decoder=ConditionalStatefulGRU(hidden_size, embed_size, hidden_size),
            next_decoder=ConditionalStatefulGRU(hidden_size, embed_size, hidden_size),
            previous_output_weight=L.Linear(hidden_size, target_vocabulary_size),
            next_output_weight=L.Linear(hidden_size, target_vocabulary_size),
        )

    def reset_state(self):
        self.encoder.reset_state()
        self.previous_decoder.reset_state()
        self.next_decoder.reset_state()

    def encode(self, source):
        source = self._prepare_input(source, self.xp.int32)
        for x in source:
            word_embedding = self.source_embed(x)
            condition = self.encoder(word_embedding)
        return condition

    def decode_once(self, y, condition, position):
        y = self._prepare_input(y, self.xp.int32)
        word_embedding = self.target_embed(y)
        h = self['{}_decoder'.format(position)](word_embedding, condition)
        y = self['{}_output_weight'.format(position)](h)
        return y

    def loss(self, prediction, t):
        t = self._prepare_input(t, self.xp.int32)
        return F.softmax_cross_entropy(prediction, t)

    def forward_train(self, source_sentence, previous_sentence, next_sentence):
        self.reset_state()
        condition = self.encode(source_sentence)
        loss = 0
        for y, t in zip(previous_sentence, previous_sentence[1:]):
            prediction = self.decode_once(y, condition, 'previous')
            loss += self.loss(prediction, t)
        for y, t in zip(next_sentence, next_sentence[1:]):
            prediction = self.decode_once(y, condition, 'next')
            loss += self.loss(prediction, t)
        return loss

    def forward_test(self, source, limit, bos_id, eos_id):
        self.reset_state()
        batch_size = len(source[0])
        condition = self.encode(source)
        previous_prediction = []
        next_prediction = []
        y = start_y = [bos_id for _ in range(batch_size)]
        while True:
            y = self.decode_once(y, condition, 'previous')
            p = [int(w) for w in y.data.argmax(1)]
            previous_prediction.append(p)
            if all(w == eos_id for w in p):
                break
            elif len(previous_prediction) >= limit:
                previous_prediction.append([eos_id for _ in range(batch_size)])
                break
            y = p
        y = start_y
        while True:
            y = self.decode_once(y, condition, 'next')
            p = [int(w) for w in y.data.argmax(1)]
            next_prediction.append(p)
            if all(w == eos_id for w in p):
                break
            elif len(next_prediction) >= limit:
                next_prediction.append([eos_id for _ in range(batch_size)])
                break
            y = p
        return previous_prediction, next_prediction

    def inference(self):
        pass
