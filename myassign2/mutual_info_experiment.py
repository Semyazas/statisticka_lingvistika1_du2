from prob import Probability, read_file, count_words


def consequtive_bigrams_experiment(lines : list[str]) -> list[float]:

    word_counts,tuple_counts, word_triple_counts, chars, last_bigram_unigram = count_words(lines)
    prob = Probability(word_counts,tuple_counts,word_triple_counts,chars, last_bigram_unigram)
    prob.compute_distributions()
    pointwise_mutual_information = [(bigram,prob.mutal_information(bigram)) for bigram in tuple_counts.keys() if tuple_counts[bigram] >= 10]

    sorted_pmi = sorted(pointwise_mutual_information, key=lambda x: x[1], reverse=True)

    print(f"Top 10 consequtive bigrams with highest PMI: {sorted_pmi[0:10]}" )

    return sorted_pmi

if __name__ == '__main__':
    lines = read_file('input\\TEXTEN.txt')
    consequtive_bigrams_experiment(lines)