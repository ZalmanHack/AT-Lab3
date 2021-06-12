from SimHash import SimHash
from SimHashBuilder import SimHashBuilder

if __name__ == '__main__':
    simhash = SimHashBuilder() \
        .set_ngram_len(2) \
        .set_permutations(30) \
        .set_vector_len(5) \
        .set_backets_count(10) \
        .set_show_debug(False) \
        .open('famous') \
        .build()

    simhash.get_marks(show=True)
