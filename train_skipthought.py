import argparse

from chainer import optimizers
from chainer.optimizer import GradientClipping
import chainer.functions as F
import numpy as np

from lib.tools.text.vocabulary import build_vocabulary
from lib.tools.text.preprocessing import text_to_word_sequence
from lib.tools.flatten import flatten
from lib.models import SkipThought

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--embed', type=int, default=1200)
    parser.add_argument('--vocabulary', type=int, default=20000)
    parser.add_argument('--hidden', type=int, default=1200)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--gradient-clipping', type=int, default=10)
    return parser.parse_args()

def main(args):
    text = ['We describe an approach for unsupervised learning of a generic distributed sentence encoder',
            'Using the continuity of text from books we train an encoder-decoder model that tries to reconstruct the surrounding sentences of an encoded passage',
            'Sentences that share semnatic and syntactic properties are thus mapped to similar vector representations']
    words = [text_to_word_sequence(sentence.lower()) for sentence in text]
    vocab, id2word = build_vocabulary(flatten(words), args.vocabulary)
    words = [['<s>'] + word + ['</s>'] for word in words]

    skip_thought = SkipThought(len(vocab), args.embed, args.hidden)
    skip_thought.use_gpu(args.gpu)
    optimizer = optimizers.AdaGrad(lr=0.01)
    optimizer.setup(skip_thought)
    optimizer.add_hook(GradientClipping(args.gradient_clipping))

    xp = skip_thought.xp

    n_epoch = args.epoch
    batch_size = args.batch

    previous_sentence = F.transpose_sequence([xp.array([vocab[word] for word in words[0]], dtype=np.int32)])
    source_sentence = F.transpose_sequence([xp.array([vocab[word] for word in words[1]], dtype=np.int32)])
    next_sentence = F.transpose_sequence([xp.array([vocab[word] for word in words[2]], dtype=np.int32)])

    print(vocab['</s>'])
    print(vocab['<s>'])

    for epoch in range(n_epoch):
        skip_thought.cleargrads()
        loss = skip_thought.forward_train(source_sentence, previous_sentence, next_sentence)
        print(loss.data)
        loss.backward()
        optimizer.update()
        previous_prediction, next_prediction = skip_thought.forward_test(source_sentence, 50, vocab['<s>'], vocab['</s>'])
        print(' '.join([id2word[id_[0]] for id_ in previous_prediction]))
        print(' '.join([id2word[id_[0]] for id_ in next_prediction]))


if __name__ == '__main__':
    args = parse_args()
    main(args)
