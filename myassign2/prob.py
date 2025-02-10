import math
import numpy as np

# Reads a text file and returns a list of non-empty lines
def read_file(file_path: str) -> list:
    with open(file_path, 'r', encoding="iso-8859-2") as file:
        lines = [line.strip() for line in file if line.strip()]  # Remove empty lines
    return lines

# Counts word occurrences, word bigram/trigram occurrences, and characters
def count_words(file: list) -> tuple:
    word_counts = {}
    word_tuple_counts = {}
    word_triple_counts = {}
    characters = set()
    last_bigram_unigram = []

    for i, line in enumerate(file):
        word = line.strip()
        word_counts[word] = word_counts.get(word, 0) + 1
        characters.update(word)

        if i > 0:
            previous = file[i - 1].strip()
            word_tuple_counts[(previous, word)] = word_tuple_counts.get((previous, word), 0) + 1

        if i > 1:
            preprevious = file[i - 2].strip()
            word_triple_counts[(preprevious, previous, word)] = word_triple_counts.get(
                (preprevious, previous, word), 0) + 1

        if i == len(file) - 1:
            last_bigram_unigram.append((previous, word))

        if i == len(file) - 1:
            last_bigram_unigram.append(word)

    return word_counts, word_tuple_counts, word_triple_counts, characters, last_bigram_unigram

class Probability:

    def __init__(self, word_counts, word_tuple_counts, characters, last_bigram_unigram):
        """#+
        Initializes a Probability object with word counts, bigram counts, character set, and the last bigram/unigram.#+
        
        Parameters:
        - word_counts (dict): A dictionary where keys are words and values are their counts.#+
        - word_tuple_counts (dict): A dictionary where keys are word tuples (bigrams or trigrams) and values are their counts.#+
        - characters (set): A set of unique characters found in the text.#+
        - last_bigram_unigram (list): A list containing the last bigram and unigram from the text.#+

        Initializes the following attributes:#+
        - unigram_distribution (dict): A dictionary where keys are words and values are their probabilities.#+
        - bigram_joint_distribution (dict): A dictionary where keys are bigrams and values are their joint probabilities.#+
        - bigram_conditional_distribution (dict): A dictionary where keys are bigrams and values are their conditional probabilities.#+
        """
        self.word_counts = word_counts
        self.word_tuple_counts = word_tuple_counts
        self.characters = characters
        self.last_bigram_unigram = last_bigram_unigram

        self.unigram_distribution = {}
        self.bigram_joint_distribution = {}
        self.bigram_conditional_distribution = {}

    def compute_distributions(self):
        sum_of_word_counts = sum(self.word_counts.values())
        sum_of_bigram_counts = sum(self.word_tuple_counts.values())

        self.unigram_distribution = {word: count / sum_of_word_counts for word, count in self.word_counts.items()}

        self.bigram_joint_distribution = {bigram: count /sum_of_bigram_counts for bigram, count in self.word_tuple_counts.items()}

        self.bigram_conditional_distribution = {
            bigram: count / self.word_counts[bigram[0]] if bigram[0] not in self.last_bigram_unigram 
                    else count / (self.word_counts[bigram[0]] - 1)
            for bigram, count in self.word_tuple_counts.items()
        }

    def mutal_information(self, bigram):
        """
        Calculates the mutual information of a given bigram.

        Parameters:
            bigram (tuple): A tuple containing two words (unigrams) for which the mutual information is calculated.#+

        Returns:
            float: The mutual information of the given bigram.

        """
        unigram_a = bigram[0]
        unigram_b = bigram[1]

        joint_probability = self.bigram_joint_distribution.get(bigram, 0)
        unigram_a_probability = self.unigram_distribution.get(unigram_a, 0)
        unigram_b_probability = self.unigram_distribution.get(unigram_b, 0)

        mutual_information =  math.log2(joint_probability / (unigram_a_probability * unigram_b_probability))

        return mutual_information
