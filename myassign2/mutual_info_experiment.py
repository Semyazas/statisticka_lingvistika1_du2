from prob import Probability, read_file, count_words


def consequtive_bigrams_experiment(lines : list[str]) -> list[float]:

    word_counts,tuple_counts, word_triple_counts, chars, last_bigram_unigram = count_words(lines)
    prob = Probability(word_counts,tuple_counts,word_triple_counts,chars, last_bigram_unigram)
    prob.compute_distributions()
    pointwise_mutual_information = [(bigram,prob.mutal_information(bigram)) for bigram in tuple_counts.keys() if word_counts[bigram[0]] >= 10 and 
                                    word_counts[bigram[1]] >= 10]

    sorted_pmi = sorted(pointwise_mutual_information, key=lambda x: x[1], reverse=True)

    print(f"Top 10 consequtive bigrams with highest PMI: {sorted_pmi[0:10]}" )

    return sorted_pmi

def nonconsecutive_bigrams_experiment(lines : list[str]) -> list[float]:
    """
    Compute the pointwise mutual information (PMI) for distant word pairs 
    (1 to 50 words apart) in the given text lines, disregarding pairs where one 
    or both words appear less than 10 times in the corpus.
    
    Returns a sorted list of the best 20 bigram pairs based on PMI.
    """
    # Preprocess and count words
    word_counts, tuple_counts, word_triple_counts, chars, last_bigram_unigram = count_words(lines)
    prob = Probability(word_counts, tuple_counts, word_triple_counts, chars, last_bigram_unigram)
    prob.compute_distributions()

    non_consecutive_bigrams = []
    len_lines = len(lines)
    window_size = 49  # As per requirements, we consider distances up to 50 words.

    # Loop through each word and find distant word pairs
    for index, word in enumerate(lines):

        
        # Define left and right bounds of the window
        window_left_index = max(0, index - window_size)
        window_right_index = min(len_lines, index + window_size + 1)

        # Add pairs for words on the left side of the window
        for word_prev in lines[window_left_index:index]:
            if  word_prev != word:
                non_consecutive_bigrams.append((word_prev, word))

        # Add pairs for words on the right side of the window
        for word_next in lines[index + 1:window_right_index]:
            if word_next != word:
                non_consecutive_bigrams.append((word, word_next))


    # Compute PMI for non-consecutive bigrams
    non_consecutive_bigrams = list(set(non_consecutive_bigrams))  # Remove duplicates
    pmi_nonconsecutive_bigrams = [(bigram, prob.mutal_information(bigram)) for bigram in non_consecutive_bigrams 
                                   if word_counts[bigram[0]] >= 10 and word_counts[bigram[1]] >= 10]
    
    print(f"PMI for non-consecutive bigrams: {pmi_nonconsecutive_bigrams[0:10]}")

if __name__ == '__main__':
    lines = read_file('input\\TEXTEN.txt')
    consequtive_bigrams_experiment(lines)

    nonconsecutive_bigrams_experiment(lines)