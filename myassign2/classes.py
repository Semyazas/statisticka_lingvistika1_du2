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

        self.left_cl_counts =self.init_single_counts(self.classes_bigram_counts,left=True)
        self.right_cl_counts = self.init_single_counts(self.classes_bigram_counts,left=False)
        self.classes = [(cl,) for cl in self.word_counts.keys() if self.word_counts[cl] >= 10]
        self.all_classes = [(cl,) for cl in self.word_counts.keys()]

        self.num_all_bigrams = sum(self.classes_bigram_counts.values())

        self.history_of_merges = []
    
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

    def class_count(self,cl, bigram_cl_counts = None) -> int:
        if bigram_cl_counts is None:
            bigram_cl_counts = self.classes_bigram_counts
        if cl in bigram_cl_counts.keys(): return bigram_cl_counts[cl]
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

    def a_plus_b_q_k(self,cl,a,b,s_cl_left,s_cl_right,
        bigr_cl_counts = None, left = False) -> float: #TODO: nešlo by přepsat nějak hezky ? 
        denominator = 0
        numerator= 0

        if not left: # (a+b,cl)
            denominator = (self.class_count((a,cl),bigr_cl_counts) + self.class_count((b,cl),bigr_cl_counts)) 
            numerator = (self.single_class_count(a,s_cl_left) + self.single_class_count(b,s_cl_left)) * self.single_class_count(cl,s_cl_right)
        else: 
            denominator = (self.class_count((cl,a),bigr_cl_counts) + self.class_count((cl,b),bigr_cl_counts)) 
            numerator = (self.single_class_count(b,s_cl_right) + self.single_class_count(a,s_cl_right)) * self.single_class_count(cl,s_cl_left)
        if denominator != 0 and numerator != 0:

            return denominator / self.num_all_bigrams * math.log2(denominator * self.num_all_bigrams / numerator)
        return 0
    def s_k(self,cl,s_counts) -> float:
        if cl in s_counts: return s_counts[cl]
        else: return 0

    def get_s_k(self,cl,  q_counts, all_classes) -> float:
        sum_left = sum([self.q_k((cl,cl1),q_counts) for cl1 in all_classes ])
        sum_right = sum([self.q_k((cl2,cl),q_counts) for cl2 in all_classes ])

        return sum_left + sum_right - self.q_k((cl,cl),q_counts) 

    def init_q_counts(self, bigram_counts) -> dict:
        q_counts = {}
        for history, word in bigram_counts:
            q_counts[(history,word)] = self.get_q_k(history,word,bigram_counts,self.left_cl_counts , self.right_cl_counts)
        return q_counts

    def init_s_k(self,q_counts) -> dict:
        s_k_counts = {}
        for cl in self.all_classes:
            s_k_counts[cl] = self.get_s_k(cl,q_counts, self.all_classes)
        return s_k_counts 


    def compute_loss_2_classes(self, r_class, l_class, 
        q_counts, s_counts, s_l, bigram_counts = None ,debug=False):
        # Compute initial q values
        q_new_class = self.q_k((l_class, r_class), q_counts)
        q_l_r = self.q_k((r_class, l_class), q_counts)

        # Compute the denominator and numerator for q_new_new
        denom_new_new = (
            self.class_count((l_class, r_class),bigram_counts)
            + self.class_count((l_class, l_class),bigram_counts)
            + self.class_count((r_class, r_class),bigram_counts)
            + self.class_count((r_class, l_class),bigram_counts)
        )

        numerator_new_new = (
            self.single_class_count(l_class, self.right_cl_counts) 
            + self.single_class_count(r_class, self.left_cl_counts)
        ) * (
            self.single_class_count(l_class, self.right_cl_counts) 
            + self.single_class_count(r_class, self.right_cl_counts)
        )

        # Compute q_new_new if denominator and numerator are nonzero
        q_new_new = 0
        if denom_new_new and numerator_new_new:
            q_new_new = (denom_new_new / self.num_all_bigrams) * math.log2(
                (denom_new_new * self.num_all_bigrams) / numerator_new_new
            )

        # Compute summations over all classes except l_class and r_class
        sum1 = sum(
            self.a_plus_b_q_k(l_cl, l_class, r_class, self.left_cl_counts, self.right_cl_counts, bigram_counts,left=True)
            for l_cl in self.all_classes if l_cl not in {l_class, r_class}
        )

        sum2 = sum(
            self.a_plus_b_q_k(r_cl, l_class, r_class, self.left_cl_counts, self.right_cl_counts, bigram_counts, left=False)
            for r_cl in self.all_classes if r_cl not in {l_class, r_class}
        )

        # Debug output if enabled
        if debug:
            debug_info = {
                "l_class": l_class,
                "r_class": r_class,
                "q_new_class": q_new_class,
                "q_l_r": q_l_r,
                "q_new_new": q_new_new,
                "sum1": sum1,
                "sum2": sum2,
                "s_l": s_l,
                "s_r": self.s_k(r_class, s_counts),
                "loss": s_l + self.s_k(r_class, s_counts) - q_new_class - q_l_r - q_new_new - sum1 - sum2
            }
            
            for key, value in debug_info.items():
                print(f"{key}: {value}")
            print("-" * 50)

        # Compute and return final loss
        return s_l + self.s_k(r_class, s_counts) - q_new_class - q_l_r - q_new_new - sum1 - sum2



    def init_Losses(self, q_counts, s_counts,bigram_counts) -> dict:
        L = {}
        class_list = list(self.classes)  # Convert to list to avoid multiple set/dict iterations
        for i, l_class in enumerate(class_list):
            print(f"{l_class} ; number: {i + 1}")
            s_l = self.s_k(l_class, s_counts)  # Precompute s_k values for efficiency

            for r_class in class_list: 
                if (l_class != r_class):
                    new_class =(l_class, r_class)
                    L[new_class] = self.compute_loss_2_classes(r_class, l_class,
                                     q_counts, s_counts, s_l, bigram_counts, False)
        return L


    def update_loss_table(self,loss_table : dict, new_classes, new_s_counts,q_counts, 
                    s_counts, merged_class, a,b,sc_old_left, sc_old_right, old_bigram_counts) -> dict:
        new_loss_table = {}
        for cl1 in new_classes:
            for cl2 in new_classes:
                if (cl1 , cl2) in loss_table:
            
                    new_loss_table[(cl1, cl2)] = (loss_table[(cl1, cl2)] - 
                                                    self.s_k(cl1,s_counts) + self.s_k(cl1,new_s_counts) - 
                                                    self.s_k(cl2,s_counts) + self.s_k(cl2,new_s_counts) +
                                                    self.a_plus_b_q_k(a,cl1,cl2,sc_old_left,sc_old_right,old_bigram_counts) +
                                                    self.a_plus_b_q_k(a,cl1,cl2,sc_old_left,sc_old_right,old_bigram_counts, left = True) +
                                                    self.a_plus_b_q_k(b,cl1,cl2,sc_old_left,sc_old_right,old_bigram_counts) +
                                                    self.a_plus_b_q_k(b,cl1,cl2,sc_old_left,sc_old_right,old_bigram_counts, left = True) -
                                                    self.a_plus_b_q_k(merged_class,cl1,cl2,self.left_cl_counts,self.right_cl_counts) -
                                                    self.a_plus_b_q_k(merged_class,cl1,cl2,self.left_cl_counts,self.right_cl_counts,left = True) 
                    ) # do q_k chceš propagovat bigram county (takhle nesedí unigram county a bigram county v čase)

                    if  new_loss_table[(cl1, cl2)] <0:
                        print(merged_class)
                        
                        print((cl1,cl2))
                        print(f"loss_table: {loss_table[(cl1, cl2)]}")
                        print(f"s_k(i) {self.s_k(cl1,s_counts)}")
                        print(f"s_k(j) {self.s_k(cl2,s_counts)}")
                        print(f"s_k-1(i) {self.s_k(cl1,s_counts)}")
                        print(f"s_k-1(j) {self.s_k(cl2,s_counts)}")

                        print(f"{self.a_plus_b_q_k(merged_class[0],cl1,cl2,sc_old_left,sc_old_right,old_bigram_counts)}")
                        print(f"{self.a_plus_b_q_k(merged_class[0],cl1,cl2,sc_old_left,sc_old_right,old_bigram_counts, left=True)}")
                        print(f"{self.a_plus_b_q_k(merged_class[1],cl1,cl2,sc_old_left,sc_old_right,old_bigram_counts)}")
                        print(f"{self.a_plus_b_q_k(merged_class[1],cl1,cl2,sc_old_left,sc_old_right,old_bigram_counts, left=True)}")
                        print(f"{self.a_plus_b_q_k(merged_class,cl1,cl2,self.left_cl_counts,self.right_cl_counts)}")
                        print(f"{self.a_plus_b_q_k(merged_class,cl1,cl2,self.left_cl_counts,self.right_cl_counts, left=True)}")
                        print(f"partial sum s {loss_table[(cl1, cl2)] - self.s_k(cl1,s_counts) + self.s_k(cl1,new_s_counts) - self.s_k(cl2,s_counts) + self.s_k(cl2,new_s_counts)}")
                        print(f"partial sum of merge: { self.a_plus_b_q_k(merged_class[0],cl1,cl2,sc_old_left,sc_old_right,old_bigram_counts) +self.a_plus_b_q_k(merged_class[0],cl1,cl2,sc_old_left,sc_old_right,old_bigram_counts, left = True) +self.a_plus_b_q_k(merged_class[1],cl1,cl2,sc_old_left,sc_old_right,old_bigram_counts) +  self.a_plus_b_q_k(merged_class[1],cl1,cl2,sc_old_left,sc_old_right,old_bigram_counts, left = True) -  self.a_plus_b_q_k(merged_class,cl1,cl2,self.left_cl_counts,self.right_cl_counts) -  self.a_plus_b_q_k(merged_class,cl1,cl2,self.left_cl_counts,self.right_cl_counts,left = True)}")
                        exit(0)
                if (merged_class == cl1 and merged_class != cl2) or (merged_class == cl2 and merged_class != cl1):
                    s_l = self.s_k(cl1, new_s_counts)
                    new_loss_table[(cl1 , cl2)] = self.compute_loss_2_classes(cl2,cl1,q_counts,new_s_counts,s_l)

        return new_loss_table

    def update_classes(self,left_class,right_class) -> None:
        new_class = tuple(left_class + right_class)

        self.classes.remove(left_class)
        self.classes.remove(right_class)
        self.classes.append(new_class)

        self.all_classes.remove(left_class)
        self.all_classes.remove(right_class)
        self.all_classes.append(new_class)

    def merge_bigram_class_counts(self,left_class : tuple[str],right_class : tuple[str]) -> dict: #TODO: napiš to hezčejc, hlavně ty updaty
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

        self.update_classes(left_class,right_class)

        return new_bigram_counts
    def GA_get_classes(self, number_of_classes):
        self.GA_classes(len(self.classes) - number_of_classes - 1)

    def GA_classes(self, number_of_iterations : int):
        new_q_counts = self.init_q_counts(self.classes_bigram_counts)
        new_s_counts = self.init_s_k(new_q_counts)

        losses = self.init_Losses(new_q_counts, new_s_counts, self.classes_bigram_counts)
        old_losses = {}
        old_left_counts={}
        old_right_counts={}
        old_s_counts = {}
        old_bigram_counts={}

        for i in range(number_of_iterations):
            old_bigram_counts = self.classes_bigram_counts.copy()
            old_losses = losses.copy()
            old_s_counts = new_s_counts.copy()
            
            min_cl = None
            min = 10000000
            for merged_cl in old_losses.keys():
                if min > old_losses[merged_cl]:
                    min = old_losses[merged_cl]
                    min_cl = merged_cl
              #      print(merged_cl)
            print(f"Best merged class: {min_cl}, Loss: {min}")
          #  print(list(set(losses.keys())))
            self.history_of_merges.append(min_cl)

            old_left_counts = self.left_cl_counts.copy()
            old_right_counts = self.right_cl_counts.copy()

            self.classes_bigram_counts = self.merge_bigram_class_counts(min_cl[0],min_cl[1])
            self.left_cl_counts = self.init_single_counts(self.classes_bigram_counts,True) # chyba
            self.right_cl_counts = self.init_single_counts(self.classes_bigram_counts)

            new_q_counts = self.init_q_counts(self.classes_bigram_counts) # chyba
            new_s_counts = self.init_s_k(new_q_counts)

            losses = self.update_loss_table(old_losses, self.classes,new_s_counts,new_q_counts,
                                    old_s_counts,tuple(min_cl[0] + min_cl[1]),min_cl[0],min_cl[1],old_left_counts,old_right_counts,old_bigram_counts)

    def mutual_information(self)-> float:
        mi = 0
        for cl1 in self.all_classes:
            for cl2 in self.all_classes:
                if cl1!= cl2:
                    p_cl1 = self.single_class_count((cl1,),self.left_cl_counts) / (self.num_all_bigrams) 
                    p_cl2 = self.single_class_count((cl2,),self.right_cl_counts) / (self.num_all_bigrams)

                    if p_cl1 != 0 and p_cl2 != 0 and  self.class_count(((cl1,),(cl2,))) > 0:
                        joint_p = self.class_count(((cl1,),(cl2,))) / self.num_all_bigrams
                        mi += joint_p * math.log2(joint_p / (p_cl1 * p_cl2))
        return mi
    
    def mutual_information_using_q(self,q_counts)-> float:
        mi = 0
        for cl1 in self.all_classes:
            for cl2 in self.all_classes:
                if cl1!= cl2:
                    mi += self.q_k(((cl1,),(cl2,)),q_counts)
        return mi

def assert_dicts_equal(dict1, dict2):
    assert dict1 == dict2, f"Dictionaries are not equal! Differences: {set(dict1.items()) ^ set(dict2.items())}"

def test_merge():
    print("Testing merge")
    wc = Word_Classes_Distribution({}, {}, {}, {})
    wc.classes = [("c1",),("c2",),("c3",),("c4",)]
    wc.all_classes = [("c1",),("c2",),("c3",),("c4",)]
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
  #  print(wc.classes)
    assert_dicts_equal(tuple_counts, bigram_counts)
    print("Test successful !")

if __name__ == '__main__':
    test_merge()
 #   exit(0)
    lines = read_file('input_classes\\TEXTEN1.ptg')
    word_counts, word_tuple_counts, word_triple_counts, characters, last_bigram_unigram = actual_count_words(lines[:8000])
   # print(word_counts)

    w_cl_distr = Word_Classes_Distribution(word_counts, word_tuple_counts,characters, last_bigram_unigram)

    w_cl_distr.GA_classes(5)

    