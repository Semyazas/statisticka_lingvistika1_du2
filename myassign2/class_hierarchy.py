from prob import Probability, read_file, count_words
from classes import Word_Classes_Distribution, actual_count_words
import math


def get_class_hierarchy( w_cl_distr :  Word_Classes_Distribution):
    
    w_cl_distr.GA_classes(7999)
    
    print(f"class hierarchy {w_cl_distr.history_of_merges}")
    print(f"class hierarchy {w_cl_distr.classes}")

if __name__ == '__main__':

    lines = read_file('input_classes\\TEXTEN1.ptg')
    word_counts, word_tuple_counts, word_triple_counts, characters, last_bigram_unigram = actual_count_words(lines[:8000])
   # print(word_counts)

    w_cl_distr = Word_Classes_Distribution(word_counts, word_tuple_counts,characters, last_bigram_unigram)

    get_class_hierarchy(w_cl_distr)
    #w_cl_distr.GA_classes(5)
