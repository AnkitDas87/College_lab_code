import nltk
from nltk.tag import tnt

# Ensure necessary NLTK packages are downloaded
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('indian')

def english_pos_tagger(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    return tagged

def hindi_pos_tagger(text):
    train_data = nltk.corpus.indian.tagged_sents('hindi.pos')
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    tokens = nltk.word_tokenize(text)
    tagged = tnt_pos_tagger.tag(tokens)
    return tagged

def marathi_pos_tagger(text):
    train_data = nltk.corpus.indian.tagged_sents('marathi.pos')
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    tokens = nltk.word_tokenize(text)
    tagged = tnt_pos_tagger.tag(tokens)
    return tagged

def main():
    print("POS Tagging Program")
    print("Choose language:")
    print("1. English")
    print("2. Hindi")
    print("3. Marathi")
    choice = input("Enter choice (1/2/3): ").strip()

    if choice not in ('1', '2', '3'):
        print("Invalid choice. Exiting.")
        return

    text = input("Enter your sentence: ").strip()
    if not text:
        print("Empty input. Exiting.")
        return

    if choice == '1':
        tagged = english_pos_tagger(text)
    elif choice == '2':
        tagged = hindi_pos_tagger(text)
    else:
        tagged = marathi_pos_tagger(text)

    print("\nTagged output:")
    for word, tag in tagged:
        print(f"{word}\t{tag}")

if __name__ == "__main__":
    main()

# | Tag | Meaning                               | Example       |
# | --- | ------------------------------------- | ------------- |
# | NN  | Noun, singular                        | dog, permit   |
# | NNS | Noun, plural                          | dogs, permits |
# | PRP | Personal pronoun                      | he, she       |
# | VB  | Verb, base form                       | permit, run   |
# | VBP | Verb, non-3rd person singular present | run, refuse   |
# | DT  | Determiner                            | the, a        |
# | TO  | to (infinitive marker)                | to            |
