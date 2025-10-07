import nltk
from nltk import pos_tag, RegexpParser
from nltk.chunk.util import tree2conlltags, conlltags2tree
from nltk.tokenize import word_tokenize

def chunk_text(text):
    # Tokenize and tag the sentence
    tokens = word_tokenize(text)
    print("\nTokens:", tokens)

    tagged = pos_tag(tokens)
    print("\nPOS Tags:", tagged)

    # Convert to flat tree (no chunks)
    flat_tree = conlltags2tree([(word, tag, 'O') for word, tag in tagged])
    print("\nFlat Tree:\n", flat_tree)

    # Chunk string before chunking
    conlltags = tree2conlltags(flat_tree)
    chunk_string_before = " ".join([f"{w}/{t}/{c}" for w, t, c in conlltags])
    print("\nChunk String Before Chunking:\n", chunk_string_before)

    # Define grammar for NP and VP
    grammar = r"""
      NP: {<DT>?<JJ>*<NN.*>+}      # Noun phrase
      VP: {<VB.*><NP|PP>*}         # Verb phrase
    """

    # Create RegexpChunkParser
    chunk_parser = RegexpParser(grammar)
    print("\nCreated RegexpChunkParser with Grammar:\n", grammar)

    # Apply the grammar to get a chunked tree
    chunked_tree = chunk_parser.parse(tagged)
    print("\nChunked Tree:\n", chunked_tree)

    # --- FIX: Safe manual conversion for chunk string after chunking ---
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

    # Optional: visualize (requires GUI)
    # chunked_tree.draw()

    return chunked_tree


def main():
    print("Chunking with Noun and Verb Phrases using NLTK\n")
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
