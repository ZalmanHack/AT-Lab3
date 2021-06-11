import glob
import os
import re
import numpy as np

class SimHash:
    def __init__(self, ngram_len: int, path_file_sought: str, path_dir_famous: str):
        self.unique_ngrams = []
        self.documents_list_ngrams = []

        self.ngram_len = ngram_len
        self.path_file_sought: str = path_file_sought
        self.path_dir_famous: str = path_dir_famous

        self.DTM = []

    def open(self):
        for path in glob.glob(os.path.join(self.path_dir_famous, '*.txt')):
            with open(path, 'r', encoding='utf-8') as file:
                self.documents_list_ngrams.append(self.get_unique(self.get_ngrams(self.get_words(file.read()))))
                self.add_unique_ngrams(self.documents_list_ngrams[-1])


    def add_unique_ngrams(self, text_ngrams: list):
        self.unique_ngrams = self._get_unique(text_ngrams, self.unique_ngrams)

    @staticmethod
    def get_words(text: str):
        return re.findall('[а-яёa-z]+', text.lower())

    def get_ngrams(self, words: list):
        n_gram_count = len(words) - self.ngram_len + 1
        return [words[pos:pos + self.ngram_len] for pos in range(n_gram_count)]

    def _get_unique(self, words: list, result: list):
        for ngram in words:
            if ngram not in result:
               result.append(ngram)
        return result

    def get_unique(self, words: list):
        return self._get_unique(words, [])

    def create_DTM(self):
        self.DTM = np.zeros((len(self.unique_ngrams), len(self.documents_list_ngrams)), dtype=int)
        for col, document in enumerate(self.documents_list_ngrams):
            for ngram in document:
                if ngram in self.unique_ngrams:
                    row = self.unique_ngrams.index(ngram)
                    self.DTM[row, col] = 1
        print(self.DTM)


if __name__ == '__main__':
    simhash = SimHash(2, 'sought', 'famous')
    simhash.open()
    simhash.create_DTM()
