# Experiment No. 6 - Name Entity Recognition using SpaCy

# Importing SpaCy library
import spacy

# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

# Ask user to input a sentence
sentence = input("Enter a sentence for Named Entity Recognition:\n")

# Process the sentence
doc = nlp(sentence)

# Display output
print("\n--- Named Entities, Positions, and Labels ---")
if doc.ents:
    for ent in doc.ents:
        print(f"Entity: {ent.text}")
        print(f"Start Char: {ent.start_char}, End Char: {ent.end_char}")
        print(f"Label: {ent.label_}")
        print("-" * 40)
else:
    print("No named entities found in the given sentence.")
