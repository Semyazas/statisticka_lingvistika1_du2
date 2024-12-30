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
    
    def merge_classes(self) -> dict[str,int]:
        new_classes = {}
    

        for left_class, right_class in self.word_tuple_counts.items():
            new_classes = self.word_to_classs.copy()
            new_classes[tuple(left_class + right_class)] = new_classes[left_class] + new_classes[right_class] 
            new_classes.remove(left_class)
            new_classes.remove(right_class)


    def greedy_A_classes(self) -> dict[str,int]:
        self.word_classes_count = self.word_counts
        self.word_to_classes       = {key : [key] for key in self.word_counts.keys()}


        return self.word_classes_count
    
    def probability_of_bigram(self, history : str, word : str) -> float:
        return self.classes_bigram_counts()

if __name__ == '__main__':
    lines = read_file('input_classes\\TEXTEN1.ptg')
    word_counts, word_tuple_counts, word_triple_counts, characters, last_bigram_unigram = actual_count_words(lines)
    print(word_counts)


    