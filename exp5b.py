import nltk
from nltk import pos_tag, RegexpParser
from nltk.chunk.util import tree2conlltags, conlltags2tree
from nltk.tokenize import word_tokenize

def chunk_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    print("\nTokens:", tokens)

    # POS tagging
    tagged = pos_tag(tokens)
    print("\nPOS Tags:", tagged)

    # Convert to flat tree (no chunks yet)
    flat_tree = conlltags2tree([(word, tag, 'O') for word, tag in tagged])
    print("\nFlat Tree:")
    print(flat_tree.pformat(margin=70))  # Pretty printed tree

    # Chunk string before applying rules
    conlltags = tree2conlltags(flat_tree)
    chunk_string_before = " ".join([f"{w}/{t}/{c}" for w, t, c in conlltags])
    print("\nChunk String Before Chunking:\n", chunk_string_before)

    # Define grammar for NP and VP
    grammar = r"""
      NP: {<DT>?<JJ>*<NN.*>+}      # Noun phrase
      VP: {<VB.*><NP|PP>*}         # Verb phrase
    """

    # Create RegexpChunkParser from grammar
    chunk_parser = RegexpParser(grammar)
    print("\nCreated RegexpChunkParser with Grammar:\n", grammar)

    # Apply the grammar to the tagged tokens
    chunked_tree = chunk_parser.parse(tagged)

    # Display the chunked tree in a readable format
    print("\nChunked Tree:")
    print(chunked_tree.pformat(margin=70))  # Pretty printed tree like the screenshot

    # Convert to manual chunk string (safe for nested chunks)
    def flatten_chunks(tree):
        result = []
        for subtree in tree:
            if isinstance(subtree, nltk.Tree):
                label = subtree.label()
                words = [w for w, t in subtree.leaves()]
                tags = [t for w, t in subtree.leaves()]
                chunked = " ".join(f"{w}/{t}/{label}" for w, t in zip(words, tags))
                result.append(f"[{chunked}]")
            else:
                w, t = subtree
                result.append(f"{w}/{t}/O")
        return " ".join(result)

    chunk_string_after = flatten_chunks(chunked_tree)
    print("\nChunk String After Applying Chunk Rules:\n", chunk_string_after)

    # Display the chunked tree in a visual window (like experiment report)
    print("\nOpening chunk tree visual window...")
    chunked_tree.draw()

    return chunked_tree


def main():
    print("=== EXPERIMENT 5: CHUNKING USING NLTK ===\n")
    text = input("Enter a sentence to chunk: ").strip()
    if not text:
        print("Empty input! Exiting.")
        return

    chunk_tree = chunk_text(text)

    # Extract NP and VP
    print("\nExtracted Noun Phrases (NP):")
    for subtree in chunk_tree.subtrees(filter=lambda t: t.label() == 'NP'):
        phrase = " ".join(word for word, tag in subtree.leaves())
        print(f" - {phrase}")

    print("\nExtracted Verb Phrases (VP):")
    for subtree in chunk_tree.subtrees(filter=lambda t: t.label() == 'VP'):
        phrase = " ".join(word for word, tag in subtree.leaves())
        print(f" - {phrase}")


if __name__ == "__main__":
    main()
