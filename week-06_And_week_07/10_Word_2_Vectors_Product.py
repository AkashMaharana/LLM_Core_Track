import numpy as np

from gensim.models import Word2Vec


def main():

    sentences = [
        ["premium", "wireless", "headphones"],
        ["wireless", "bluetooth", "headphones"],
        ["premium", "bluetooth", "earbuds"],
        ["wireless", "portable", "speaker"],
        ["bluetooth", "portable", "speaker"],
        ["coffee", "maker", "stainless", "steel"],
        ["espresso", "coffee", "machine"],
        ["premium", "coffee", "machine"],
    ]

    model = Word2Vec(
        sentences=sentences,
        vector_size=20,
        window=2,
        min_count=1,
        workers=1,
        sg=1,
        epochs=100,
    )

    product_words = [
        "premium",
        "wireless",
        "headphones",
    ]

    # ---------------------------------------------------------
    # 1. Collect vector for each word
    # ---------------------------------------------------------

    word_vectors = []

    for word in product_words:

        if word in model.wv:

            word_vectors.append(
                model.wv[word]
            )

    # ---------------------------------------------------------
    # 2. Average word vectors
    # ---------------------------------------------------------

    product_vector = np.mean(
        word_vectors,
        axis=0,
    )

    print("\nPRODUCT")
    print("=" * 60)

    print(
        " ".join(product_words)
    )

    print("\nPRODUCT VECTOR")
    print("=" * 60)

    print(product_vector)

    print(
        "\nVector dimensions:",
        len(product_vector)
    )


if __name__ == "__main__":
    main()