from prob import Probability, read_file, count_words

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
    
    def merge_classes(self) -> dict[str,int]:
        new_classes = {}
        new_word_to_class = {}
        new_class_bigram_distribution = {}
        new_class_bigram_counts = {}


        for left_class, right_class in self.classes_bigram_counts.keys():
            new_classes = self.word_to_class.copy()         
            new_class = tuple(left_class + right_class)
            new_classes[new_class] = new_classes[left_class] + new_classes[right_class] 
            new_classes.remove(left_class)
            new_classes.remove(right_class)

            for history, word in self.word_tuple_counts.keys():
                h_class = list(self.word_to_class[history])
                w_class = list(self.word_to_class[word])

                if self.word_to_class[history] == left_class:
                    h_class = new_class
                if self.word_to_class[word] == right_class:
                    w_class = new_class

                if tuple(h_class + w_class) not in new_class_bigram_counts:
                    new_class_bigram_counts[tuple(h_class + w_class)] = 1 
                else:
                    new_class_bigram_counts[tuple(h_class + w_class)] += 1

                if self.word_to_class[history] == left_class and self.word_to_class[word] == right_class:
                    new_word_to_class[word] = new_class
                    new_word_to_class[history] = new_class
                else:
                    new_word_to_class[word] = self.word_to_class[history]
                    new_word_to_class[history] = self.word_to_class[word]

            new_class_bigram_distribution = self.get_class_distribution(history,word,new_class,new_class_bigram_counts, new_class_bigram_counts)

            prob = 0
            for word in word_counts:
                for history in word_counts:
                    prob += self._get_probability(word, history, new_class_bigram_distribution)
                    
            assert prob == 1 , "Probability does not sum to 1"
                
    def _get_probability(self,word,history, distr) -> float:

        if (history, word) in distr:
            return distr[(history, word)]
        return 0

    def greedy_A_classes(self) -> dict[str,int]:
        self.compute_distributions()

        self.word_classes_count = self.word_counts.copy()
        self.word_to_classes       = {key : [key] for key in self.word_counts.keys()}
        self.classes_bigram_counts = self.word_tuple_counts.copy()
        self.classes_bigram_distributions = self.bigram_conditional_distribution.copy()
        self.merge_classes()

        return self.word_classes_count
    
    def get_class_distribution(self, history : str, word : str, distr : dict, classes_counts : dict, classes_bigram_counts) -> dict:
        w_class = self.word_to_class[word]
        h_class = self.word_to_class[history]
        for history, word in self.word_tuple_counts.keys():
            distr[(history,word)] = self.word_counts(word) / classes_counts(w_class) *  classes_bigram_counts((h_class, w_class)) / classes_counts(h_class) 
        return distr


if __name__ == '__main__':
    lines = read_file('input_classes\\TEXTEN1.ptg')
    word_counts, word_tuple_counts, word_triple_counts, characters, last_bigram_unigram = actual_count_words(lines[:8000])
#    print(word_counts)

    w_cl_distr = Word_Classes_Distribution(word_counts, word_tuple_counts,characters, last_bigram_unigram)
    w_cl_distr.greedy_A_classes()


    