import native
import time

# print(native.train_tokenizer('/Users/vparikh/Desktop/picochat/datasets/bpe/TinyStories-train.txt', 'vocab.txt', 50_000))
native.tokenize("vocab.txt", "datasets/bpe/TinyStories-train.txt", "datasets/bpe/train_tokens.bin")