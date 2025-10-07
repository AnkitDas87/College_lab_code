import nltk
import re
import string
import inflect
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt

# Initialize resources
p = inflect.engine()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
english_vocab = set(w.lower() for w in words.words())

# 1. Tokenization (both sentences and words)
def tokenize_text(text):
    sentences = sent_tokenize(text)
    words_ = word_tokenize(text)
    return sentences, words_

# 2. Lowercase conversion
def text_lowercase(text):
    return text.lower()

# 3. Remove numbers (digits)
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

# 4. Convert numbers to words
def convert_numbers_to_words(text):
    temp_str = text.split()
    new_string = []
    for word in temp_str:
        if word.isdigit():
            new_string.append(p.number_to_words(word))
        else:
            new_string.append(word)
    return ' '.join(new_string)

# 5. Remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# 6. Remove extra whitespaces
def remove_whitespace(text):
    return " ".join(text.split())

# 7. Remove stopwords
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return filtered_text

# 8. Stemming
def stem_words(text):
    word_tokens = word_tokenize(text)
    return [stemmer.stem(word) for word in word_tokens]

# 9. Lemmatization
def lemmatize_words(text):
    word_tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]

# 10. POS tagging
def pos_tagging(text):
    word_tokens = word_tokenize(text)
    return nltk.pos_tag(word_tokens)

# 11. Detect words matching regex patterns (example: words ending with 'ed')
def detect_word_patterns(pattern):
    wordlist = [w for w in words.words('en') if w.islower()]
    matched_words = [w for w in wordlist if re.search(pattern, w)]
    return matched_words

# Demonstration function running all steps in order on input text
def preprocess_pipeline(text):
    print("Original Text:\n", text, "\n")

    # Tokenization
    sentences, tokens = tokenize_text(text)
    print("Sentences:\n", sentences)
    print("Tokens:\n", tokens, "\n")

    # Lowercase
    text = text_lowercase(text)
    print("Lowercase Text:\n", text, "\n")

    # Remove numbers
    no_numbers = remove_numbers(text)
    print("Text without numbers:\n", no_numbers, "\n")

    # Convert numbers to words
    text_with_num_words = convert_numbers_to_words(text)
    print("Text with numbers converted to words:\n", text_with_num_words, "\n")

    # Remove punctuation
    no_punct = remove_punctuation(text)
    print("Text without punctuation:\n", no_punct, "\n")

    # Remove extra whitespaces
    no_whitespace = remove_whitespace(no_punct)
    print("Text without extra whitespaces:\n", no_whitespace, "\n")

    # Remove stopwords
    no_stopwords = remove_stopwords(no_whitespace)
    print("Tokens after removing stopwords:\n", no_stopwords, "\n")

    # Stemming
    stemmed = stem_words(no_whitespace)
    print("Stemmed tokens:\n", stemmed, "\n")

    # Lemmatization
    lemmatized = lemmatize_words(no_whitespace)
    print("Lemmatized tokens:\n", lemmatized, "\n")

    # POS tagging
    pos_tags = pos_tagging(no_whitespace)
    print("POS Tags:\n", pos_tags, "\n")

    # Example word pattern detection: words ending with 'ed'
    pattern = r'ed$'
    matched = detect_word_patterns(pattern)
    print(f"Words ending with 'ed' (sample): {matched[:20]} ...\n")

    # Plot frequency of filtered tokens (no stopwords)
    freq_dist = nltk.FreqDist(no_stopwords)
    print("Plotting frequency distribution of tokens (excluding stopwords)...")
    freq_dist.plot(20, cumulative=False)

def main():
    input_text = input("Enter text to preprocess:\n")
    preprocess_pipeline(input_text)

if __name__ == "__main__":
    main()
