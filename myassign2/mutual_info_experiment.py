from prob import Probability, read_file, count_words


def consequtive_bigrams_experiment(lines : list[str]) -> list[float]:

    word_counts,tuple_counts, word_triple_counts, chars, last_bigram_unigram = count_words(lines)
    prob = Probability(word_counts,tuple_counts,chars, last_bigram_unigram)
    prob.compute_distributions()
    pointwise_mutual_information = [(bigram,prob.pointwise_mutual_information(bigram)) for bigram in tuple_counts.keys() if prob.word_tupled_counts[bigram[0]] >= 10 and 
                                    prob.word_tupled_counts[bigram[1]] >= 10]

    sorted_pmi = sorted(pointwise_mutual_information, key=lambda x: x[1], reverse=True)

    print("Comsequtive bigrams")
    for i,bigram in enumerate(sorted_pmi[0:20]):
        print(f"{i}: bigram: {bigram}")
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
    prob = Probability(word_counts, tuple_counts,
                         chars, last_bigram_unigram)
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
        l_index = 0
        if index  > 0:
            l_index = index - 1
        for word_prev in lines[window_left_index:l_index]:
            if  word_prev != word:
                non_consecutive_bigrams.append(((word_prev,), (word,)))

        r_index = len(lines) -1
        if index < r_index:
            r_index = index + 2

        # Add pairs for words on the right side of the window
        for word_next in lines[r_index:window_right_index]:
            if word_next != word:
                non_consecutive_bigrams.append(((word,), (word_next,)))


    # Compute PMI for non-consecutive bigrams
    non_consecutive_bigrams = list(set(non_consecutive_bigrams))  # Remove duplicates
    pmi_nonconsecutive_bigrams = [(bigram, prob.pointwise_mutual_information(bigram)) for bigram in non_consecutive_bigrams 
                                   if prob.word_tupled_counts[bigram[0]] >= 10 and prob.word_tupled_counts[bigram[1]] >= 10]
    sorted_pmi = sorted(pmi_nonconsecutive_bigrams, key=lambda x: x[1], reverse=True)

    print(f"PMI for non-consecutive bigrams: ")
    for i,bigram in enumerate(sorted_pmi[0:20]):
        print(f"{i}: bigram: {bigram}")

if __name__ == '__main__':
    lines = read_file('input\\TEXTEN.txt')
    consequtive_bigrams_experiment(lines)

    nonconsecutive_bigrams_experiment(lines)