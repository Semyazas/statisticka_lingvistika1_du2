from prob import Probability, read_file, count_words

def actual_count_words(file : list[str]) -> tuple:
    file = [line.split('/')[0] for line in file]
    return count_words(file = file)

if __name__ == '__main__':
    lines = read_file('input_classes\\TEXTEN1.ptg')
    word_counts, word_tuple_counts, word_triple_counts, characters, last_bigram_unigram = actual_count_words(lines)
    print(word_counts)


