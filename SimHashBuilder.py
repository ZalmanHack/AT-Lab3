from __future__ import annotations

from SimHash import SimHash


class SimHashBuilder:
    def __init__(self):
        self.sim_hash = SimHash()

    def set_ngram_len(self, value: int) -> SimHashBuilder:
        self.sim_hash.ngram_len = value
        return self

    def set_permutations(self, value: int) -> SimHashBuilder:
        self.sim_hash.permutations = value
        return self

    def set_vector_len(self, value: int) -> SimHashBuilder:
        self.sim_hash.vector_len = value
        return self

    def set_backets_count(self, value: int) -> SimHashBuilder:
        self.sim_hash.backets_count = value
        return self

    def set_show_debug(self, value: bool) -> SimHashBuilder:
        self.sim_hash.show_debug = value
        return self


    def open(self, path_dir_famous) -> SimHashBuilder:
        self.sim_hash.path_dir_famous = path_dir_famous
        self.sim_hash.open()
        return self

    def build(self) -> SimHash:
        return self.sim_hash
