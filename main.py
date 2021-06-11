import glob
import math
import os
import random
import re
import numpy as np

class SimHash:
    def __init__(self, ngram_len: int, path_dir_famous: str):
        self.unique_ngrams = []
        self.documents_list_ngrams = []

        self.ngram_len = ngram_len
        self.path_dir_famous: str = path_dir_famous

        self.document_names = []
        self.DTM = np.array(0)

    def open(self):
        for path in glob.glob(os.path.join(self.path_dir_famous, '*.txt')):
            self.document_names.append(path.split('\\')[-1])
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
        print("Матрица DTM:")
        print(self.DTM)

    def create_min_hash(self, permutations: int):
        # создаем массив в котором будем хранить массив с индексами для перестановок
        perm_list = [list(range(len(self.unique_ngrams))) for p in range(permutations)]
        # перемешиваем полученные индексы
        for i, _ in enumerate(perm_list):
            random.shuffle(perm_list[i])
        # создаем таблицу мин хеш где: строки - кол-во возможных перестоновок; кол-во столбцов = кол-ву документов 
        min_hash_table = np.zeros((permutations, self.DTM.shape[1]), dtype=int)
        # перебираем каждый из созданных масивов перестановок
        for perm_index, perm in enumerate(perm_list):
            # перебор перестановок для каждого документа
            for doc_index, _ in enumerate(range(self.DTM.shape[1])):
                # последовательно проходимся по списку перестановок [0..N]
                for index in range(len(perm)):
                    # получаем значение индекса для исходного списка относительно полученных рандомных индексов
                    rand_index = perm.index(index)
                    # ищем перрвое включение '1' в матрице DTM
                    if self.DTM[rand_index, doc_index] == 1:
                        # после нахождения сохраняем сгенерированный индекс в созданную матрицу мин хеш и выходим и дока 
                        min_hash_table[perm_index, doc_index] = rand_index
                        break

        print("Матрица мин хеш:")
        print(min_hash_table)
        print("Массив перестановок:")
        [print(perm) for perm in perm_list]
        return min_hash_table

    @staticmethod
    def split_list(data_list, size_bit):
        return [data_list[i:i+size_bit] for i in range(0, len(data_list), size_bit)]
    
    # создает матрицу симхеш и сразу заполняет матриуц бакетов, и отдает их по завершению
    def get_backets_matrix(self, min_hash_table: np.ndarray, vector_len, backets_count):
        # получаем кол-во строк разбитых на вектора (с учетом дробной части, округляем вверх)
        vectors_count = math.ceil(min_hash_table.shape[0] / vector_len)
        backets_list = [[[] for j in range(backets_count)] for i in range(vectors_count)]
        # проход по столбцам матрицы мин хеш
        for col_index in range(min_hash_table.shape[1]):
            # получаем полный столбец матрицы
            column = [row[col_index] for row in min_hash_table]
            # получаем разбитый на части (размером с vector_len) столбец
            column = self.split_list(column, vector_len)
            # перебор полученных векторов столбца и получение хеша с внесением в бакеты
            for row_bit_index, col_bit in enumerate(column):
                # проеобразовываем каждую часть вектора в строку, получаем хеш и номер бакета по остатку от деления
                backet_index = hash(','.join([str(x) for x in col_bit])) % backets_count
                # добавляем полученный индекс документа в нужную ячейку списка бакетов
                backets_list[row_bit_index][backet_index].append(col_index)

        print("матрица сим хеш:")
        [print(backet) for backet in backets_list]
        return backets_list

    def get_similar_docs(self, backets_matrix):
        # список индексов документов, похожих друг на друга
        similar_docs_indexes = []
        for backet in backets_matrix:
            for item in backet:
                if len(item) > 1 and item not in similar_docs_indexes:
                    similar_docs_indexes.append(item)

        for docs_index, docs in enumerate(similar_docs_indexes):
            docs_names = []
            for doc_index in docs:
                docs_names.append(self.document_names[doc_index])
            similar_docs_indexes[docs_index] = docs_names

        print("список документов имеющие повторения между собой:")
        [print(x) for x in similar_docs_indexes]



if __name__ == '__main__':
    sh = SimHash(
        ngram_len=2,
        path_dir_famous='famous')
    sh.open()
    sh.create_DTM()
    minhash = sh.create_min_hash(3)
    backets_matrix = sh.get_backets_matrix(minhash, 2, 5)
    sh.get_similar_docs(backets_matrix)
