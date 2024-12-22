from prob import Probability, read_file, count_words


def consequtive_bigrams_experiment(lines : list[str]) -> list[float]:

    word_counts,tuple_counts, word_triple_counts, chars, last_bigram_unigram = count_words(lines)
    prob = Probability(word_counts,tuple_counts,word_triple_counts,chars, last_bigram_unigram)
    prob.compute_distributions()
    pointwise_mutual_information = [(bigram,prob.mutal_information(bigram)) for bigram in tuple_counts.keys() if tuple_counts[bigram] >= 10]

    sorted_pmi = sorted(pointwise_mutual_information, key=lambda x: x[1], reverse=True)

    print(f"Top 10 consequtive bigrams with highest PMI: {sorted_pmi[0:10]}" )

    return sorted_pmi

def nonconsecutive_bigrams_experiment(lines : list[str]) -> list[float]:
    word_counts,tuple_counts, word_triple_counts, chars, last_bigram_unigram = count_words(lines)
    prob = Probability(word_counts,tuple_counts,word_triple_counts,chars, last_bigram_unigram)
    prob.compute_distributions()

    non_consequtive_bigrams = []
    len_lines = len(lines) 

    window_size = 3

    for index,word in enumerate(lines):
        window_left_index = 0
        if index > window_size: 
            window_left_index = index - window_size

        if len_lines - index < window_size: 
            window_right_index = index + window_size
        
        for word_prev in lines[window_left_index:index]:
            if word_prev != word:
                non_consequtive_bigrams.append((word_prev,word))

        for word_next in lines[index+1:window_right_index+1]:
            if word_next != word:
                non_consequtive_bigrams.append((word,word_next))

    print(non_consequtive_bigrams)

if __name__ == '__main__':
    lines = read_file('input\\TEXTEN.txt')
    consequtive_bigrams_experiment(lines)

    nonconsecutive_bigrams_experiment(lines)