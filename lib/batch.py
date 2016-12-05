from itertools import islice
from operator import length_hint

def read_dataset(filename, eos_token):
    with open(filename) as f:
        return [line.strip().split() + [eos_token] for line in f]

def generate_batch(lines, vocab, batch_size, pad_id=-1):
    max_length = max(len(line) for line in lines)
    sentences = iter([[vocab[w] for w in line] for line in lines])
    while length_hint(sentences) > 0:
        filled_batch = [sentence + [pad_id] * (max_length - len(sentence) + 1)
                            for sentence in islice(sentences, batch_size)]
        yield list(zip(*filled_batch))
