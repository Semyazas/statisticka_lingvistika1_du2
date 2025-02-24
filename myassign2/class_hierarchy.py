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
    our_tuples = []
    for cl,i in enumerate(list(sorted(classes))):
        print(f"{i}: {cl}")
        our_tuples.append(i)

    print(our_tuples)
    word_tuples = [
        ('(', 'and', 'as', 'but', 'for', 'if', 'or', 'than', 'when'),
        (')', 'animals', 'breeds', 'case', 'cases', 'conditions', 'facts', 'individuals', 'nature', 'plants', 'races', 'species', 'state', 'structure', 'subject', 'variation', 'varieties'),
        (',',),
        ('.',),
        (':', 'at', 'between', 'from', 'in', 'me', 'on', 'only', 'under'),
        (';', 'In', 'The', 'are'),
        ('I', 'It', 'it', 'there', 'they', 'we'),
        ('a', 'any', 'each', 'some', 'their', 'these', 'very'),
        ('all', 'by', 'differ', 'how', 'much', 'nearly', 'that', 'this', 'what', 'which', 'with'),
        ('an', 'certain', 'different', 'distinct', 'domestic', 'domesticated', 'even', 'great', 'its', 'less', 'long', 'manner', 'many', 'more', 'most', 'my', 'one', 'other', 'our', 'same', 'several', 'short', 'slight', 'such', 'wild'),
        ('be', 'been', 'not', 'often', 'so'),
        ('believe', 'can', 'cannot', 'could', 'do', 'has', 'have', 'is', 'may', 'must', 'see', 'shall', 'will', 'would'),
        ('of',),
        ('the',),
        ('to',)
    ]
    
    assert sorted(word_tuples) == sorted(our_tuples)
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
