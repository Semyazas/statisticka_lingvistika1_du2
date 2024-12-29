from prob import Probability, read_file, count_words

def actual_count_words(file : list[str]) -> tuple:
    file = [line.split('/')[0] for line in file]
    return count_words(file = file)

class Word_Classes:

    def __init__(self,word_counts : dict[str,int],
                      bigram_counts : dict[tuple[str,str],int])-> None:
        self.word_counts = word_counts
        self.bigram_counts = bigram_counts
        self.word_classes = {}
        
    def merge_classes(self,word_classses : dict) -> dict[str,int]:
        new_classes = {}

        for left_word, right_word in self.bigram_counts.items():
            new_classes = word_classses.copy()
            new_classes[left_word, right_word] = new_classes[left_word, right_word]
            new_classes.remove(left_word)


    def greedy_A_classes(self,words : dict[str,int]) -> dict[str,int]:
        self.word_classes_count = words


        return word_classes_count

if __name__ == '__main__':
    lines = read_file('input_classes\\TEXTEN1.ptg')
    word_counts, word_tuple_counts, word_triple_counts, characters, last_bigram_unigram = actual_count_words(lines)
    print(word_counts)


    