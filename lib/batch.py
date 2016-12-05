from itertools import islice
from operator import length_hint

def read_dataset(filename, eos_token):
    with open(filename) as f:
        return [line.strip().split() + [eos_token] for line in f]

def generate_batch(lines, vocab, batch_size, pad_id=-1):
    sentences = iter([[vocab[w] for w in line] for line in lines])
    while length_hint(sentences) > 0:
        batch = [sentence for sentence in islice(sentences, batch_size)]
        max_length = max(len(b) for b in batch)
        filled_batch = [b + [pad_id] * (max_length - len(b) + 1) for b in batch]
        yield list(zip(*filled_batch))
