from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import RegexpStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

# Download required NLTK data if not already done
nltk.download('wordnet', quiet=True)

def morphological_analysis(word):
    print(f"\nMorphological analysis for the word: '{word}'\n")

    # 1. Lancaster Stemmer
    lancaster_stemmer = LancasterStemmer()
    print("Lancaster Stemmer:", lancaster_stemmer.stem(word))

    # 2. Porter Stemmer
    porter_stemmer = PorterStemmer()
    print("Porter Stemmer:", porter_stemmer.stem(word))

    # 3. Snowball Stemmer (English)
    snowball_stemmer = EnglishStemmer()
    print("English Snowball Stemmer:", snowball_stemmer.stem(word))

    # 4. Regexp Stemmer (remove 'ing', 's', or 'e' endings if word length >= 4)
    regexp_stemmer = RegexpStemmer('ing$|s$|e$', min=4)
    print("Regexp Stemmer:", regexp_stemmer.stem(word))

    # 5. WordNet Lemmatizer (verb context)
    lemmatizer = WordNetLemmatizer()
    print("WordNet Lemmatizer (verb):", lemmatizer.lemmatize(word, 'v'))


def main():
    while True:
        user_input = input("Enter a word for morphological analysis (or type 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        elif user_input == "":
            print("Please enter a valid word.")
            continue
        else:
            morphological_analysis(user_input)

if __name__ == "__main__":
    main()
