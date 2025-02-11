from prob import Probability, read_file, count_words
import math

def actual_count_words(file : list[str]) -> tuple:
    file = [line.split('/')[0] for line in file]
    return count_words(file = file)

class Word_Classes_Distribution(Probability):

    def __init__(self,word_counts : dict[str,int],bigram_counts : dict[tuple[str,str],int],
                  characters : list[str], last_bigram_unigram : list[tuple])-> None:
        Probability.__init__(self, word_counts, bigram_counts,characters, last_bigram_unigram)

        
        self.word_to_class = {}
        self.classes_counts = {}
        self.classes_bigram_counts = {}
        self.classes_bigram_distributions = {}

        self.num_all_bigrams = 0
   
    
    def get_class_distribution(self, history : str, word : str, distr : dict, classes_counts : dict, classes_bigram_counts) -> dict:
        w_class = self.word_to_class[word]
        h_class = self.word_to_class[history]
        for history, word in self.word_tuple_counts.keys():
            distr[(history,word)] = self.word_counts(word) / classes_counts(w_class) *  classes_bigram_counts((h_class, w_class)) / classes_counts(h_class) 
        return distr

    def single_class_count(self,cl,bigram_counts,context = False):
        count = 0
        i =1
        if context:
            i = 0
        for bigram in bigram_counts.keys():
            if bigram[i] == cl:
                count += bigram_counts[bigram[i]] 
        return count

    def get_q_k(self,left_class, right_class,bigram_counts):
        return bigram_counts[(left_class, right_class)] / self.num_all_bigrams * math.log2( 
                                                                    self.num_all_bigrams * 
                                                                    bigram_counts[(left_class, right_class)] / (
                                                                        self.single_class_counts(left_class, bigram_counts, True)*
                                                                        self.single_class_counts(right_class, bigram_counts, True)
                                                                    )                   
                                                                )

    def get_s_k(self,cl, bigram_counts):
        sum_left = sum([for ]

        for 
        pass

    def merge_bigram_class_counts(self,left_class : tuple[str],right_class : tuple[str]) -> dict:
        new_bigram_counts = self.classes_bigram_counts.copy()

        new_class = tuple(left_class + right_class)

        print(new_class)
        new_bigram_counts[(new_class,new_class)] = (
                                                            new_bigram_counts[(left_class,right_class)] + 
                                                            new_bigram_counts[(right_class,left_class)] + 
                                                            new_bigram_counts[(left_class,left_class)]  + 
                                                            new_bigram_counts[(right_class,right_class)]
                                                        )
        to_remove = [(left_class,right_class),(left_class,left_class),(right_class,right_class),(right_class,left_class)]

        for history,word in self.classes_bigram_counts.keys():
            if (history,word) not in to_remove:
                if (history == left_class and word != right_class) or (history == right_class and word != left_class):
                    new_bigram_counts[(new_class,word)] = new_bigram_counts[(left_class,word)] + new_bigram_counts[(right_class,word)]
                    to_remove.append((history,word))

                elif  (history != left_class and word == right_class) or (history != right_class and word == left_class):
                    new_bigram_counts[(history,new_class)] = new_bigram_counts[(history,left_class)] + new_bigram_counts[(history,right_class)]
                    to_remove.append((history,word))

        for t in to_remove:
            new_bigram_counts.pop(t)

        return new_bigram_counts


def assert_dicts_equal(dict1, dict2):
    assert dict1 == dict2, f"Dictionaries are not equal! Differences: {set(dict1.items()) ^ set(dict2.items())}"

def test_merge():
    print("Testing merge")
    wc = Word_Classes_Distribution({}, {}, {}, {})
    
    wc_classes_bigram_counts = {
        (("c1",), ("c1",)): 10, (("c1",), ("c2",)): 2, (("c1",), ("c3",)): 0, (("c1",), ("c4",)): 1,
        (("c2",), ("c1",)): 0,  (("c2",), ("c2",)): 0, (("c2",), ("c3",)): 5, (("c2",), ("c4",)): 2,
        (("c3",), ("c1",)): 0,  (("c3",), ("c2",)): 2, (("c3",), ("c3",)): 0, (("c3",), ("c4",)): 3,
        (("c4",), ("c1",)): 2,  (("c4",), ("c2",)): 3, (("c4",), ("c3",)): 0, (("c4",), ("c4",)): 0
    }

    bigram_counts = {
        (("c1",), ("c1",)): 10, (("c1",), ("c2", "c4")): 3, (("c1",), ("c3",)): 0,
        (("c2", "c4"), ("c1",)): 2, (("c2", "c4"), ("c2", "c4")): 5, (("c2", "c4"), ("c3",)): 5,
        (("c3",), ("c1",)): 0, (("c3",), ("c2", "c4")): 5, (("c3",), ("c3",)): 0
    }
    wc.classes_bigram_counts = wc_classes_bigram_counts
    
    tuple_counts = wc.merge_bigram_class_counts(("c2",),("c4",))
    print(tuple_counts)
    assert_dicts_equal(tuple_counts, bigram_counts)
    print("Test successful !")

if __name__ == '__main__':
    test_merge()

    lines = read_file('input_classes\\TEXTEN1.ptg')
    word_counts, word_tuple_counts, word_triple_counts, characters, last_bigram_unigram = actual_count_words(lines[:8000])
#    print(word_counts)

    w_cl_distr = Word_Classes_Distribution(word_counts, word_tuple_counts,characters, last_bigram_unigram)
    w_cl_distr.greedy_A_classes()


    