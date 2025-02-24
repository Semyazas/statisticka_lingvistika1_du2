import sys
import math
from prob import Probability, read_file, count_words
from classes import Word_Classes_Distribution, actual_count_words
import networkx as nx
import matplotlib.pyplot as plt


def get_15_classes(w_cl_distr: Word_Classes_Distribution):
    w_cl_distr.GA_get_classes(14)
    classes = []
    for cl in w_cl_distr.classes:
        classes.append(tuple(sorted(list(cl))))
    for cl,i in enumerate(list(sorted(classes))):
        print(f"{i}: {cl}")

def get_class_hierarchy(w_cl_distr: Word_Classes_Distribution):
    w_cl_distr.GA_classes(len(w_cl_distr.classes) - 1)  # Removed `-1`, unless it's intentional
    print(f"class history: {w_cl_distr.history_of_merges}")


if __name__ == '__main__':

    lines = read_file('input_classes/TEXTEN1.ptg')
    word_counts, word_tuple_counts, word_triple_counts, characters, last_bigram_unigram = actual_count_words(lines[:8000])

    w_cl_distr = Word_Classes_Distribution(word_counts, word_tuple_counts, characters, last_bigram_unigram)

    if len(sys.argv) > 1 and sys.argv[1] == "h":
        get_15_classes(w_cl_distr)
    else:
        get_class_hierarchy(w_cl_distr)
