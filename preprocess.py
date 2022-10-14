from collections import Counter
import numpy as np
import argparse
import random
import re


class Preprocess:
    def __init__(self, dataset_path):
        self.path = dataset_path
        self.pattern = "sin|cos|tan|\d|\w|\(|\)|\+|-|\*+"
        self.vocab_size = 35  # this is counted in data_summary.ipynb, 32 unique words+3 signals
        self.max_length = 32    # max_length=29, +3 signals
        self.cur_index = 3  # the current index for new word
        # <bos>: begin of sequence, <eos>: end of sequence
        self.word_index = {"<pad>": 0, "<bos>": 1, "<eos>": 2}  # word to index dict
        self.index_word = {0: "<pad>", 1: "<bos>", 2: "<eos>"}  # index to word dict

    # map each word to their unique index
    def build_vocab(self):
        with open(self.path, 'r') as f:
            for line in f:
                if self.cur_index >= self.vocab_size:
                    break
                words = Counter(re.findall(self.pattern, line.strip()))
                for word in words:
                    if word not in self.word_index:
                        self.word_index[word] = self.cur_index
                        self.index_word[self.cur_index] = word
                        self.cur_index += 1

    # generate encoder_input, decoder_input and decoder_output
    def generate_data(self, file_path):
        encoder_input, decoder_input = [], []
        with open(file_path, 'r') as f:
            for line in f:
                text1, text2 = re.findall(self.pattern, line.strip().split('=')[0]), re.findall(self.pattern, line.strip().split('=')[1])
                encode = [self.word_index[word] for word in text1] + [0] * (self.max_length - len(text1))      # padding
                # adding start and end signal to shift decoder_input
                decode = [self.word_index["<bos>"]]+[+self.word_index[word] for word in text2]+[self.word_index["<eos>"]]+[0] * (self.max_length - len(text2)-2)
                encoder_input.append(encode)
                decoder_input.append(decode)
        decoder_output = self.decoder_output_creater(decoder_input)
        return np.array(encoder_input, dtype="float32"), np.array(decoder_input, dtype="float32"), decoder_output

    # generate decoder_output based on decoder_input
    def decoder_output_creater(self, decoder_input_data):
        decoder_output_data = np.zeros((len(decoder_input_data), self.max_length, self.vocab_size), dtype="float32")
        for i, seqs in enumerate(decoder_input_data):
            for j, seq in enumerate(seqs):
                if j > 0:
                    # skip the first <bos>, decoder_target_data[:, t, :] = decoder_input_data[:, t + 1, :].
                    decoder_output_data[i][j-1][seq] = 1.
        return decoder_output_data

    # split the dataset into train.txt, test.txt, valid.txt and save them
    def split(self, train_ratio, train_path, valid_ratio, valid_path, test_ratio, test_path, seed):
        pairs = open(self.path, 'r').readlines()
        random.seed(seed)
        random.shuffle(pairs)
        train = pairs[:int(train_ratio * len(pairs))]
        valid = pairs[int(train_ratio * len(pairs)):int((1 - test_ratio) * len(pairs))]
        test = pairs[int((1 - test_ratio) * len(pairs)):]
        with open(train_path, 'w') as f:
            f.write(''.join(train))
        with open(valid_path, 'w') as f:
            f.write(''.join(valid))
        with open(test_path, 'w') as f:
            f.write(''.join(test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocessing data: .")
    parser.add_argument("--dataset_path", type=str, default="dataset.txt")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--train_path", type=str, default="train.txt")
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--valid_path", type=str, default="valid.txt")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--test_path", type=str, default="test.txt")
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    a = Preprocess(args.dataset_path)
    a.split(args.train_ratio, args.train_path, args.valid_ratio, args.valid_path, args.test_ratio, args.test_path, args.seed)