from collections import defaultdict
import math
import re


class Assignment1:
    def __init__(self) -> None:
        self.train_data = None
        self.val_data = None
        self.train_tokens = None
        self.val_tokens = None
        self.unigram_counts = None
        self.bigram_counts = None

    # Open and read the file
    def read_file(self, path):
        try:
            with open(path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            print(f"File '{path}' not found.")
            return ""

    # Load the training and validation data
    def load_data(self):
        self.train_data = self.read_file("A1_DATASET/train.txt")
        self.val_data = self.read_file("A1_DATASET/val.txt")

    # Tokenize the input data
    def process_text(self, text_data):
        text = text_data.lower()

        # Split text into sentences
        sentences = text.split("\n")

        processed_sentences = []

        for sentence in sentences:
            # Remove newline characters and non-alphabetic characters
            cleaned_sentence = re.sub("[^a-z]", " ", sentence)

            # Remove extra spaces and check if the sentence is not empty
            cleaned_sentence = re.sub("\s{2,}", " ", cleaned_sentence).strip()

            if cleaned_sentence:
                # Add start and end tokens and tokenize the sentence
                processed_sentence = f"<s> {cleaned_sentence} </s>"
                tokens = processed_sentence.split()
                processed_sentences.extend(tokens)

        return processed_sentences

    # Prepare bigram tokens
    def bigram_tokenize(self):
        bigram_tokens = []
        for token in range(len(self.val_tokens)-1):
            cur = self.val_tokens[token]
            next = self.val_tokens[token+1]
            bigram_tokens.append((cur, next))
        return bigram_tokens

    # Create hashmap of unigram counts
    def prepare_unigrams(self, corpus):
        unigram_counts = defaultdict(int)

        for word in corpus:
            unigram_counts[word] += 1

        return unigram_counts

    # Create hashmap of bigram counts
    def prepare_bigrams(self, corpus):
        bigram_counts = defaultdict(int)

        for i in range(len(corpus) - 1):
            cur_word = corpus[i]
            next_word = corpus[i + 1]
            bigram_counts[(cur_word, next_word)] += 1

        return bigram_counts

    # Load and preprocess data
    def preprocess_data(self):
        self.load_data()

        # Preprocess training and validation data
        self.train_tokens = self.process_text(self.train_data)
        self.val_tokens = self.process_text(self.val_data)

        # Unigram and bigram counts
        self.unigram_counts = self.prepare_unigrams(self.train_tokens)
        self.bigram_counts = self.prepare_bigrams(self.train_tokens)

    # Filter word having frequancy less the threshold k
    def filter_with_threshold(self, word_counts, k, unigram=False):
        new_word_counts = defaultdict(int)
        count_of_filtered = 0

        for word, count in word_counts.items():
            if count > k:
                new_word_counts[word] = count
            else:
                count_of_filtered += count

        if unigram:
            new_word_counts['<UNK/>'] = count_of_filtered
        else:
            new_word_counts[('<UNK/>', '<UNK/>')] = count_of_filtered

        return new_word_counts

    # Calculating perplexity
    def calc_perplexity(self, probs, N):
        # Calculate the sum of log probabilities
        log_prob_sum = sum([-prob for prob in probs])

        # Calculate perplexity
        perplexity = math.exp(log_prob_sum / N)

        return perplexity

    # Calculate unigram and bigram probabilities, and calculate perplexity on training data
    def calc_probabilities(self, alpha=0, threshold=0):
        self.uni_probs = defaultdict(int)
        self.bi_probs = defaultdict(int)
        uni_prob_ar = []
        bi_prob_ar = []

        word_count = len(self.train_tokens)

        self.unique_cnt = len(self.unigram_counts.keys())

        if alpha > 0:
            self.unigram_counts = self.filter_with_threshold(
                self.unigram_counts, threshold, unigram=True)
            self.bigram_counts = self.filter_with_threshold(
                self.bigram_counts, threshold)

        alpha_v = alpha * self.unique_cnt

        for word, count in self.unigram_counts.items():
            self.uni_probs[word] = (
                (count + alpha)/(word_count + alpha_v))
            uni_prob_ar.append(count*self.uni_probs[word])

        for word_pair, count in self.bigram_counts.items():
            self.bi_probs[word_pair] = (
                (count + alpha)/(self.unigram_counts[word_pair[0]] + alpha_v))
            bi_prob_ar.append(count*self.bi_probs[word_pair])

        return self.calc_perplexity(uni_prob_ar, len(self.train_tokens)), self.calc_perplexity(bi_prob_ar, len(self.train_tokens))

    # Calculate unigram and bigram perplexities on validation data
    def validate_models(self, alpha=0):
        # For unigram
        uni_probs = []
        for word in self.val_tokens:
            word_prob = 0
            if word not in self.uni_probs:
                if self.uni_probs['<UNK/>'] <= 0:
                    continue

                uni_probs.append(math.log(self.uni_probs['<UNK/>']))
                continue

            uni_probs.append(math.log(self.uni_probs[word]))

        # For bigram
        bigram_tokens = self.bigram_tokenize()
        bigram_probs = []
        for word_pair in bigram_tokens:
            if word_pair not in self.bi_probs:
                if self.bi_probs[('<UNK/>', '<UNK/>')] <= 0:
                    continue

                bigram_probs.append(
                    math.log(self.bi_probs[('<UNK/>', '<UNK/>')]))
                continue

            bigram_probs.append(math.log(self.bi_probs[word_pair]))

        return self.calc_perplexity(uni_probs, len(self.val_tokens)), self.calc_perplexity(bigram_probs, len(self.val_tokens))


alpha = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 3]
threshold = [1, 2, 3, 4, 5]
for i in alpha:
    for j in threshold:
        model = Assignment1()
        model.preprocess_data()
        print(f"Perplexities for alpha={i} and threshold = {j}")
        print()

        unigram_perplixity, bigram_perplexity = model.calc_probabilities(
            alpha=i, threshold=j)
        print("On Training data")
        print("Unigram: ", unigram_perplixity)
        print("Bigram: ", bigram_perplexity)
        print()

        unigram_val_preplexity, bigram_val_preplexity = model.validate_models(
            i)
        print("On Validation data")
        print("Unigram: ", unigram_val_preplexity)
        print("Bigram: ", bigram_val_preplexity)
        print()
        print()
