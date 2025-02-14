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
        self.classes_bigram_counts = self.word_tuple_counts
        self.classes_bigram_distributions = {}

        self.num_all_bigrams = len(self.classes_bigram_counts.keys())
    
    def get_class_distribution(self, history : str, word : str, distr : dict, classes_counts : dict, classes_bigram_counts) -> dict:
        w_class = self.word_to_class[word]
        h_class = self.word_to_class[history]
        for history, word in self.word_tuple_counts.keys():
            distr[(history,word)] = self.word_counts(word) / classes_counts(w_class) *  classes_bigram_counts((h_class, w_class)) / classes_counts(h_class) 
        return distr
    
    def init_single_q_counts(self,q_counts, left = False) -> dict:
        single_q_counts = {}
        i = 1
        if left: i = 0
        for bigram in q_counts.keys():
            if bigram[i] in single_q_counts:
                single_q_counts[bigram[i]] += self.q_k(bigram,q_counts)
            else:
                single_q_counts[bigram[i]] = self.q_k(bigram,q_counts)
        return single_q_counts

    def single_class_count(self,cl,bigram_counts,context = False) -> int:
        count = 0
        i =1
        if context:
            i = 0
        for bigram in bigram_counts.keys():
            if bigram[i] == cl:
                count += bigram_counts[bigram] 
        return count

    def get_q_k(self,left_class, right_class,bigram_counts) -> float:
        #TODO: Napiš hezky
        if (left_class, right_class) in bigram_counts and self.single_class_count(left_class, bigram_counts, True) !=0 and self.single_class_count(right_class, bigram_counts, True) !=0:
            return (bigram_counts[(left_class, right_class)] / self.num_all_bigrams) * math.log2(
                self.num_all_bigrams * bigram_counts[(left_class, right_class)] /
                (self.single_class_count(left_class, bigram_counts, True) *
                self.single_class_count(right_class, bigram_counts, True))
            )
        else:
            return 0
    def q_k(self,cl, q_counts) -> float:
        if cl in q_counts: return q_counts[cl]
        else: return 0

    def s_k(self,cl,s_counts) -> float:
        if cl in s_counts: return s_counts[cl]
        else: return 0

    def get_s_k(self,cl,  q_counts):
        sum_left = sum([q_counts[(cont,w)] for cont,w in q_counts.keys() if cont == cl ])
        sum_right = sum([q_counts[(cont,w)] for cont,w in q_counts.keys() if w == cl ])

        return sum_left + sum_right - self.q_k(cl,q_counts)

    def get_sub_k(self,s_counts,q_counts,cl_left,cl_right):
        return s_counts[cl_left] + s_counts[cl_right] - q_counts[(cl_left,cl_right)] - q_counts[(cl_right,cl_left)]

    def get_add_k(self,new_class, q_counts):
        sum_left = sum([q_counts[(cont,w)] for cont,w in q_counts.keys() if cont == new_class ])
        sum_right = sum([q_counts[(cont,w)] for cont,w in q_counts.keys() if w == new_class ])
        return sum_right + sum_left + q_counts[(new_class,new_class)]

    def init_q_counts(self, bigram_counts, classes) -> dict:
        q_counts = {}
        for history, word in bigram_counts:
            q_counts[(history,word)] = self.get_q_k(history,word,bigram_counts)
        return q_counts

    def init_s_k(self,q_counts) -> dict:
        s_k_counts = {}
        for cl in q_counts.keys():
            s_k_counts[cl] = self.get_s_k(cl,q_counts)
        return s_k_counts 

    def init_Losses(self, q_counts, s_counts, classes) -> dict:
        L = {}
        for l_class,r_class in q_counts.keys():
            new_class = (l_class,r_class)
            new_q_counts = self.init_q_counts(self.merge_bigram_class_counts((l_class,),(r_class,)),classes)
            L[new_class] = ( self.s_k(l_class,s_counts) + self.s_k(r_class,s_counts) -  self.q_k((new_class),q_counts)
                           - self.q_k((r_class,l_class),q_counts) - self.q_k((new_class,new_class),q_counts)
                           - sum([new_q_counts[(l_cl,r_cl)] for l_cl,r_cl in new_q_counts.keys() if r_cl == new_class and l_cl != new_class])
                           - sum([new_q_counts[(l_cl,r_cl)] for l_cl,r_cl in new_q_counts.keys() if r_cl != new_class and l_cl == new_class])
            )
        return L
    
    def merge_bigram_class_counts(self,left_class : tuple[str],right_class : tuple[str]) -> dict:
        new_bigram_counts = self.classes_bigram_counts.copy()

        new_class = tuple(left_class + right_class)

        print(new_class)
        to_remove = []
        new_bigram_counts[(new_class,new_class)] = 0
        for item in [(left_class, right_class), (left_class,left_class), (right_class,left_class), (right_class,right_class)]:
            if item in new_bigram_counts.keys():
                new_bigram_counts[(new_class,new_class)] += new_bigram_counts[item]
                to_remove.append(item)

        for history,word in self.classes_bigram_counts.keys():
            if (history,word) not in to_remove:
                if (history == left_class and word != right_class) or (history == right_class and word != left_class):
                    new_bigram_counts[(new_class,word)] = new_bigram_counts[(left_class,word)] + new_bigram_counts[(right_class,word)]
                    to_remove.append((history,word))

                elif  (history != left_class and word == right_class) or (history != right_class and word == left_class):
                    new_bigram_counts[(history,new_class)] = new_bigram_counts[(history,left_class)] + new_bigram_counts[(history,right_class)]
                    to_remove.append((history,word))
       # print(new_bigram_counts)
        for t in to_remove:
       #     print(t)
            
            new_bigram_counts.pop(t)

        return new_bigram_counts

    def GA_classes(self, number_of_iterations : int):
        classes = [cl for cl in self.word_counts.keys()]
        q_counts = self.init_q_counts(self.word_tuple_counts,classes)
        s_counts = self.init_s_k(q_counts)
    #    print(q_counts)
    #    return
        losses = self.init_Losses(q_counts, s_counts,classes)
      #  print(losses)
        min_cl = None
        min = 10000000
        for merged_cl in losses.keys():
            if min > losses[merged_cl]:
                min = losses[merged_cl]
                min_cl = merged_cl
        print(f"Best merged class: {min_cl}, Loss: {min}")


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
    assert_dicts_equal(tuple_counts, bigram_counts)
    print("Test successful !")

if __name__ == '__main__':
    test_merge()

    lines = read_file('input_classes\\TEXTEN1.ptg')
    word_counts, word_tuple_counts, word_triple_counts, characters, last_bigram_unigram = actual_count_words(lines[:8000])
#    print(word_counts)

    w_cl_distr = Word_Classes_Distribution(word_counts, word_tuple_counts,characters, last_bigram_unigram)

    w_cl_distr.GA_classes(10)

    