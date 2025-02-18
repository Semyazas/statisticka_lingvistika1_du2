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
        self.classes = [cl for cl in self.word_counts.keys() if self.word_counts[cl] ]

        self.num_all_bigrams = sum(self.classes_bigram_counts.values())
    
    def init_single_counts(self, bigram_counts : dict ,left = False) -> dict:
        single_counts = {}
        i = 1
        if left: i = 0
        for bigram in bigram_counts.keys():
            if bigram[i] in single_counts:
                single_counts[bigram[i]] += bigram_counts[bigram]
            else:
                single_counts[bigram[i]] = bigram_counts[bigram]
        return single_counts

    def single_class_count(self,cl,single_q_counts : dict) -> int:
        if cl in single_q_counts.keys(): return single_q_counts[cl]
        else: return 0

    def class_count(self,cl) -> int:
        if cl in self.classes_bigram_counts.keys(): return self.classes_bigram_counts[cl]
        else: return 0

    def get_q_k(self,left_class, right_class,bigram_counts,
                single_counts_left : dict, single_counts_right : dict) -> float:

        if (left_class, right_class) in bigram_counts and (self.single_class_count(left_class, single_counts_left) !=0 
            and self.single_class_count(right_class, single_counts_right) !=0) and bigram_counts[(left_class, right_class)] != 0:
            
            return (bigram_counts[(left_class, right_class)] / self.num_all_bigrams) * math.log2(
                self.num_all_bigrams * bigram_counts[(left_class, right_class)] /
                (self.single_class_count(left_class, single_counts_left) *
                self.single_class_count(right_class, single_counts_right))
            )
        else:
            return 0
    def q_k(self,cl, q_counts) -> float:
        if cl in q_counts: return q_counts[cl]
        else: return 0

    def a_plus_b_q_k(self,cl,a,b,s_cl_left,s_cl_right,left = False) -> float:
        denominator = 0
        numerator= 0
        a = (a,)
        b = (b,)
        cl = (cl,)

        if not left:
            denominator = (self.class_count((a,cl)) + self.class_count((b,cl))) 
            numerator = (self.single_class_count(a,s_cl_left) + self.single_class_count(b,s_cl_left)) * self.single_class_count(cl,s_cl_right)
        else: 
            denominator = (self.class_count((cl,a)) + self.class_count((cl,b))) 
            numerator = (self.single_class_count(b,s_cl_right) + self.single_class_count(a,s_cl_right)) * self.single_class_count(cl,s_cl_left)
        if denominator != 0 and numerator != 0:

            return denominator / self.num_all_bigrams * math.log2(denominator * self.num_all_bigrams / numerator)
        return 0
    def s_k(self,cl,s_counts) -> float:
        if cl in s_counts: return s_counts[cl]
        else: return 0

    def get_s_k(self,cl,  q_counts):
        sum_left = sum([q_counts[(cont,w)] for cont,w in q_counts.keys() if cont == (cl,) ])
        sum_right = sum([q_counts[(cont,w)] for cont,w in q_counts.keys() if w == (cl,) ])

        return sum_left + sum_right - self.q_k(((cl,),(cl,)),q_counts) 

    def init_q_counts(self, bigram_counts,  single_counts_left, single_counts_right) -> dict:
        q_counts = {}
        for history, word in bigram_counts:
            q_counts[(history,word)] = self.get_q_k(history,word,bigram_counts,single_counts_left , single_counts_right)
        return q_counts

    def init_s_k(self,classes,q_counts) -> dict:
        s_k_counts = {}
        for cl in classes:
            s_k_counts[cl] = self.get_s_k(cl,q_counts)
        return s_k_counts 

    #TODO: zkontrolovat formule + opravit merge/q_counts
    def init_Losses(self, q_counts, s_counts, classes,l_cl_counts,r_cl_counts) -> dict:
        L = {}
        class_list = list(classes)  # Convert to list to avoid multiple set/dict iterations
        for i, l_class in enumerate(class_list):
            print(f"{l_class} ; number: {i + 1}")
            s_l = self.s_k(l_class, s_counts)  # Precompute s_k values for efficiency

            for r_class in class_list: 
                if (l_class != r_class):
                    #TODO: oprav county
                    new_class = ((l_class,), (r_class,))
                    q_new_class = self.q_k(new_class, q_counts)

                    q_l_r = self.q_k(((r_class,), (l_class,)), q_counts)

                    denom_new_new = 2* (self.single_class_count((l_class,),l_cl_counts) + self.single_class_count((r_class,),r_cl_counts))
                    numerator_new_new = (self.single_class_count((l_class,),l_cl_counts) + self.single_class_count((r_class,),l_cl_counts)) * (
                                         self.single_class_count((l_class,),r_cl_counts) + self.single_class_count((r_class,),r_cl_counts))
                    q_new_new = 0

                    if denom_new_new != 0 and numerator_new_new != 0:
                        q_new_new = (denom_new_new / self.num_all_bigrams) * math.log2(denom_new_new * 
                                                            self.num_all_bigrams / numerator_new_new) 
                    # Optimized sum calculations using dictionary lookups
                    sum1 = sum(self.a_plus_b_q_k(l_cl,l_class,r_class,l_cl_counts,r_cl_counts,left = True) 
                              for l_cl in classes if  l_cl != l_class and l_cl != r_class)

                    sum2 = sum(self.a_plus_b_q_k(r_cl,l_class,r_class,l_cl_counts,r_cl_counts,left = False)
                                for r_cl in classes if r_cl != l_class and r_cl != r_class)

                    L[new_class] = s_l + self.s_k(r_class, s_counts) - q_new_class - q_l_r - q_new_new - sum1 - sum2
                    """
                    print(s_l)
                    print(self.s_k(r_class, s_counts))
                    print(q_new_class)
                    print(q_l_r)
                    print(q_new_new)
                    print(sum1)
                    print(sum2)
                    print(L[new_class])
                    print("--------------------------------")"""
        return L
    
    def merge_bigram_class_counts(self,left_class : tuple[str],right_class : tuple[str]) -> dict:
        new_bigram_counts = self.classes_bigram_counts.copy()

        new_class = tuple(left_class + right_class)
        to_remove = []
        new_bigram_counts[(new_class,new_class)] = 0
        for item in [(left_class, right_class), (left_class,left_class), (right_class,left_class), (right_class,right_class)]:
            if item in new_bigram_counts.keys():
                new_bigram_counts[(new_class,new_class)] += self.class_count(item)
                to_remove.append(item)
        for history,word in self.classes_bigram_counts.keys():
            if (history,word) not in to_remove:
                if (history == left_class and word != right_class) or (history == right_class and word != left_class):
                    new_bigram_counts[(new_class,word)] = self.class_count((left_class,word)) + self.class_count((right_class,word))              
                    to_remove.append((history,word))

                elif  (history != left_class and word == right_class) or (history != right_class and word == left_class):
                    new_bigram_counts[(history,new_class)] = self.class_count((history,left_class)) + self.class_count((history,right_class))
                    to_remove.append((history,word))
        for t in to_remove:
            new_bigram_counts.pop(t)

        return new_bigram_counts

    def GA_classes(self, number_of_iterations : int):

        classes = [cl for cl in self.word_counts.keys() if self.word_counts[cl] >= 10]
        classes_bigrams = {cl:self.word_tuple_counts[cl] for cl in self.word_tuple_counts.keys() if cl[0] in classes and cl[1] in classes}
        print(len(classes))

        left_cl_counts = self.init_single_counts(classes_bigrams,left=True)
        right_cl_counts = self.init_single_counts(classes_bigrams,left=False)
        q_counts = self.init_q_counts(classes_bigrams,left_cl_counts,right_cl_counts)
        
        s_counts = self.init_s_k(classes,q_counts)

        losses = self.init_Losses(q_counts, s_counts,classes,left_cl_counts,right_cl_counts)
        print(losses)
        max_cl = None
        max = -10000000
        for merged_cl in losses.keys():
            if max < losses[merged_cl]:
                max = losses[merged_cl]
                max_cl = merged_cl
                print(merged_cl)
        print(f"Best merged class: {max_cl}, Loss: {max}")

    def mutual_information(self)-> float:
        left_cl_counts = self.init_single_counts(self.classes_bigram_counts,left=True)
        right_cl_counts = self.init_single_counts(self.classes_bigram_counts,left=False)
        mi = 0
        for cl1 in self.classes:
            for cl2 in self.classes:
                if cl1!= cl2:
                    p_cl1 = self.single_class_count((cl1,),left_cl_counts) / (self.num_all_bigrams) 
                    p_cl2 = self.single_class_count((cl2,),right_cl_counts) / (self.num_all_bigrams)

                    if p_cl1 != 0 and p_cl2 != 0 and  self.class_count(((cl1,),(cl2,))) > 0:
                        joint_p = self.class_count(((cl1,),(cl2,))) / self.num_all_bigrams
                        mi += joint_p * math.log2(joint_p / (p_cl1 * p_cl2))
        return mi


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

    print(w_cl_distr.mutual_information())

    w_cl_distr.GA_classes(10)

    