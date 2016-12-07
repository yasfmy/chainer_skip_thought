from itertools import islice, tee
from operator import length_hint

def read_dataset(filename, bos_token='<s>', eos_token='</s>'):
    with open(filename) as f:
        return [[bos_token] + line.strip().split() + [eos_token] for line in f]

def sentences_to_token_ids(sentences, vocab):
    return [[vocab[word] for word in sentence] for sentence in sentences]

def generate_batch(sentences, batch_size, pad_id=-1):
    sentences = iter(sentences)
    while length_hint(sentences) > 0:
        batch = [sentence for sentence in islice(sentences, batch_size)]
        max_length = max(len(b) for b in batch)
        filled_batch = [b + [pad_id] * (max_length - len(b) + 1) for b in batch]
        yield list(zip(*filled_batch))

def generate_pair_batch(sentences):
    iters = tee(sentences, 3)
    next(iters[1])
    next(iters[2])
    next(iters[2])
    return zip(iters[0], iters[1], iters[2])

