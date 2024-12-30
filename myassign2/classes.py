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
            new_classes = self.word_to_classs.copy()
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

                new_class_bigram_counts[tuple(h_class + w_class)] =

                if self.word_to_class[history] == left_class and self.word_to_class[word] == right_class:
                    new_word_to_class[word] = new_class
                    new_word_to_class[history] = new_class
                else:
                    new_word_to_class[word] = self.word_to_class[history]
                    new_word_to_class[history] = self.word_to_class[word]

                
                

    def greedy_A_classes(self) -> dict[str,int]:
        self.word_classes_count = self.word_counts
        self.word_to_classes       = {key : [key] for key in self.word_counts.keys()}


        return self.word_classes_count
    

    def probability_of_bigram(self, history : str, word : str, distr : dict, classes_counts : dict, classes_bigram_counts) -> dict:
        w_class = self.word_to_class[word]
        h_class = self.word_to_class[history]
        for history, word in self.word_tuple_counts.keys():
            distr[(h_class,w_class)] = self.word_counts(word) / classes_counts(w_class) *  classes_bigram_counts((h_class, w_class)) / classes_counts(h_class) 
        return distr
if __name__ == '__main__':
    lines = read_file('input_classes\\TEXTEN1.ptg')
    word_counts, word_tuple_counts, word_triple_counts, characters, last_bigram_unigram = actual_count_words(lines)
    print(word_counts)


    