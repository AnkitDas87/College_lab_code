import nltk
from nltk.util import ngrams
from textblob import TextBlob

# Download required NLTK data
nltk.download('punkt', quiet=True)

def extract_ngrams_nltk(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [' '.join(grams) for grams in n_grams]

def extract_ngrams_textblob(data, num):
    n_grams = TextBlob(data).ngrams(num)
    return [' '.join(grams) for grams in n_grams]

def main():
    print("N-Gram Model using NLTK and TextBlob\n")
    sentence = input("Enter a sentence: ").strip()

    if not sentence:
        print("Empty input. Please enter a valid sentence.")
        return

    print("\nUsing NLTK:")
    for n in range(1, 5):
        ngrams_list = extract_ngrams_nltk(sentence, n)
        print(f"{n}-gram:", ngrams_list)

    print("\nUsing TextBlob:")
    for n in range(1, 5):
        ngrams_list = extract_ngrams_textblob(sentence, n)
        print(f"{n}-gram:", ngrams_list)

if __name__ == "__main__":
    main()
