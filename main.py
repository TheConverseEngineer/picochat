import native
import time

print(native.train_tokenizer('/Users/vparikh/Desktop/picochat/datasets/bpe/TinyStories-valid.txt', 'vocab.txt', 4096))
native.tokenize("vocab.txt", "datasets/bpe/TinyStories-valid.txt", "output.txt")