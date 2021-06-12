import glob
import math
import os
import random
import re
import numpy as np
import tqdm as tqdm

from custom_console import custom_console

class SimHash:
    def __init__(self):
        # список всех уникальных N-грамм
        self.unique_ngrams = []
        # список уникальных N-грамм каждого документа
        self.documents_list_ngrams = []
        # именя документов, которые оцениваются
        self.document_names = []

        self.DTM = np.array(0)

        self._ngram_len = 2
        # кол-во перестановок в мин хеш
        self._permutations = 5
        # длина вектора при формировании сим хеш
        self._vector_len = 3
        # кол-во бакетов, на которые будут распределяться хеши во время формирования сим хеш
        self._backets_count = 5
        # путь к файлам, которые необходимо оценить
        self._path_dir_famous: str = ""
        # отображение матриц для дебага
        self._show_debug = False



    @property
    def ngram_len(self) -> int:
        return self._ngram_len

    @property
    def permutations(self) -> int:
        return self._permutations

    @property
    def vector_len(self) -> int:
        return self._vector_len

    @property
    def backets_count(self) -> int:
        return self._backets_count

    @property
    def path_dir_famous(self) -> str:
        return self._path_dir_famous

    @property
    def show_debug(self) -> bool:
        return self._show_debug

    @ngram_len.setter
    def ngram_len(self, value: int) -> None:
        if self._ngram_len == value:
            return
        self._ngram_len = value

    @permutations.setter
    def permutations(self, value: int) -> None:
        if self._permutations == value:
            return
        self._permutations = value

    @vector_len.setter
    def vector_len(self, value: int) -> None:
        if self._vector_len == value:
            return
        self._vector_len = value

    @backets_count.setter
    def backets_count(self, value: int) -> None:
        if self._backets_count == value:
            return
        self._backets_count = value

    @path_dir_famous.setter
    def path_dir_famous(self, value: str) -> None:
        if self._path_dir_famous == value:
            return
        self._path_dir_famous = value

    @show_debug.setter
    def show_debug(self, value: bool) -> None:
        if self._show_debug == value:
            return
        self._show_debug = value

    @ngram_len.deleter
    def ngram_len(self) -> None:
        del self._ngram_len

    @permutations.deleter
    def permutations(self) -> None:
        del self._permutations

    @vector_len.deleter
    def vector_len(self) -> None:
        del self._vector_len

    @backets_count.deleter
    def backets_count(self) -> None:
        del self._backets_count

    @path_dir_famous.deleter
    def path_dir_famous(self) -> None:
        del self._path_dir_famous

    @show_debug.deleter
    def show_debug(self) -> None:
        del self._show_debug


    @staticmethod
    def get_words(text: str):
        return re.findall('[а-яёa-z]+', text.lower())

    def get_ngrams(self, words):
        return self._get_ngrams(words, self.ngram_len)

    @staticmethod
    def _get_ngrams(words: list, ngram_len):
        n_gram_count = len(words) - ngram_len + 1
        return [words[pos:pos + ngram_len] for pos in range(n_gram_count)]

    def _get_unique(self, words: list, result: list):
        for ngram in words:
            if ngram not in result:
                result.append(ngram)
        return result

    def get_unique(self, words: list):
        return self._get_unique(words, [])

    def add_unique_ngrams(self, text_ngrams: list):
        self.unique_ngrams = self._get_unique(text_ngrams, self.unique_ngrams)

    def open(self):
        for path in tqdm.tqdm(glob.glob(os.path.join(self.path_dir_famous, '*.txt')), desc="Чтение документов"):
            self.document_names.append(path.split('\\')[-1])
            with open(path, 'r', encoding='utf-8') as file:
                self.documents_list_ngrams.append(self.get_unique(self.get_ngrams(self.get_words(file.read()))))
                self.add_unique_ngrams(self.documents_list_ngrams[-1])

    def create_DTM(self):
        self.DTM = np.zeros((len(self.unique_ngrams), len(self.documents_list_ngrams)), dtype=int)
        for col, document in tqdm.tqdm(enumerate(self.documents_list_ngrams), desc="Создание дет.матр", total=len(self.documents_list_ngrams)):
            for ngram in document:
                if ngram in self.unique_ngrams:
                    row = self.unique_ngrams.index(ngram)
                    self.DTM[row, col] = 1
        if self.show_debug:
            print("Матрица DTM:")
            print(self.DTM)

    def create_min_hash(self):
        # создаем массив в котором будем хранить массив с индексами для перестановок
        perm_list = [list(range(len(self.unique_ngrams))) for p in range(self.permutations)]
        # перемешиваем полученные индексы
        for i, _ in enumerate(perm_list):
            random.shuffle(perm_list[i])
        # создаем таблицу мин хеш где: строки - кол-во возможных перестоновок; кол-во столбцов = кол-ву документов
        min_hash_table = np.zeros((self.permutations, self.DTM.shape[1]), dtype=int)
        # перебираем каждый из созданных масивов перестановок
        for perm_index, perm in tqdm.tqdm(enumerate(perm_list), desc="Создание min hash", total=len(perm_list)):
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
        if self.show_debug:
            print("Матрица мин хеш:")
            print(min_hash_table)
            print("Массив перестановок:")
            [print(perm) for perm in perm_list]
        return min_hash_table

    @staticmethod
    def split_list(data_list, size_bit):
        return [data_list[i:i + size_bit] for i in range(0, len(data_list), size_bit)]

    # создает матрицу симхеш и сразу заполняет матриуц бакетов, и отдает их по завершению
    def create_backets_matrix(self, min_hash_table: np.ndarray):
        # получаем кол-во строк разбитых на вектора (с учетом дробной части, округляем вверх)
        vectors_count = math.ceil(min_hash_table.shape[0] / self.vector_len)
        backets_list = [[[] for j in range(self.backets_count)] for i in range(vectors_count)]
        # проход по столбцам матрицы мин хеш
        for col_index in tqdm.tqdm(range(min_hash_table.shape[1]), desc="Создание sim hash"):
            # получаем полный столбец матрицы
            column = [row[col_index] for row in min_hash_table]
            # получаем разбитый на части (размером с vector_len) столбец
            column = self.split_list(column, self.vector_len)
            # перебор полученных векторов столбца и получение хеша с внесением в бакеты
            for row_bit_index, col_bit in enumerate(column):
                # проеобразовываем каждую часть вектора в строку, получаем хеш и номер бакета по остатку от деления
                backet_index = hash(','.join([str(x) for x in col_bit])) % self.backets_count
                # добавляем полученный индекс документа в нужную ячейку списка бакетов
                backets_list[row_bit_index][backet_index].append(col_index)
        if self.show_debug:
            print("матрица сим хеш:")
            [print(backet) for backet in backets_list]
        return backets_list

    def show_marks(self, backets):
        # создаем матрицу с названиями документов
        result_matrix = [[name] for name in self.document_names]
        result_matrix.insert(0, [''])
        result_matrix[0].extend(self.document_names)
        for i, rm in enumerate(result_matrix):
            if i > 0:
                result_matrix[i].extend([0 for _ in range(len(self.document_names))])

        # преобразовываем матрицу бакетов в список, сдержащий строго по два повторения
        # (все что больше разбиваются на 2 и добавляются)
        similar_docs_indexes = []
        for backet in backets:
            for item in backet:
                if len(item) > 1:
                    item = self._get_ngrams(item, 2)
                    for i in item:
                        i.sort()
                        similar_docs_indexes.append(i)

        # заполняем итоговую матрицу с вероятностями
        for row in range(len(result_matrix) - 1):
            for col in range(len(result_matrix) - 1):
                counter = 0
                if row == col:
                    result_matrix[row + 1][col + 1] = ''
                else:
                    pattern = [row, col]
                    pattern.sort()
                    for sdi in similar_docs_indexes:
                        if pattern == sdi:
                            counter += 1
                    result_matrix[row + 1][col + 1] = counter/len(backets)*100
        if self.show_debug:
            print("Индексы всех совпавшиз документов (с повторениями)")
            print(similar_docs_indexes)
        print("Вероятностная оценка документов на плагиат:")
        max_len = 0
        for i, row in enumerate(result_matrix):
            if i == 0:
                max_len = custom_console.get_max_len(row)
            print(custom_console.build_row(row, max_len))


    def get_marks(self, show: bool):
        self.create_DTM()
        min_hash = self.create_min_hash()
        backets = self.create_backets_matrix(min_hash)
        self.show_marks(backets)


