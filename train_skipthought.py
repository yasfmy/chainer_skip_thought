import argparse

from chainer import optimizers
from chainer.optimizer import GradientClipping
import chainer.functions as F
import numpy as np

from lib.tools.text.vocabulary import build_vocabulary
from lib.tools.text.preprocessing import text_to_word_sequence
from lib.tools.flatten import flatten
from lib.models import SkipThought
from lib.batch import read_dataset, sentences_to_token_ids, generate_batch, generate_pair_batch

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
    train = read_dataset('ptb.train.txt')
    vocab, id2word = build_vocabulary(flatten(train))
    train_sentences = sentences_to_token_ids(train, vocab)

    skip_thought = SkipThought(len(vocab), args.embed, args.hidden)
    skip_thought.use_gpu(args.gpu)
    optimizer = optimizers.AdaGrad(lr=0.01)
    optimizer.setup(skip_thought)
    optimizer.add_hook(GradientClipping(args.gradient_clipping))

    n_epoch = args.epoch
    batch_size = args.batch
    N = len(train)

    for epoch in range(n_epoch):
        sum_loss = 0
        for prev_batch, source_batch, next_batch in generate_pair_batch(train_sentences, batch_size):
            skip_thought.cleargrads()
            loss = skip_thought.forward_train(source_batch, prev_batch, next_batch)
            loss.backward()
            optimizer.update()
            sum_loss += loss.data * batch_size
        print(loss.data / N)
        #previous_prediction, next_prediction = skip_thought.forward_test(source_sentence, 50, vocab['<s>'], vocab['</s>'])
        #print(' '.join([id2word[id_[0]] for id_ in previous_prediction]))
        #print(' '.join([id2word[id_[0]] for id_ in next_prediction]))


if __name__ == '__main__':
    args = parse_args()
    main(args)
