from gensim.models import Word2Vec


def main():

    # ---------------------------------------------------------
    # 1. Tiny tokenized training corpus
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 2. Train Word2Vec
    # ---------------------------------------------------------

    model = Word2Vec(
        sentences=sentences,
        vector_size=20,
        window=2,
        min_count=1,
        workers=1,
        sg=1,
        epochs=100,
    )

    # ---------------------------------------------------------
    # 3. Inspect one word vector
    # ---------------------------------------------------------

    print("\nVECTOR FOR 'wireless'")
    print("=" * 60)

    print(
        model.wv["wireless"]
    )

    # ---------------------------------------------------------
    # 4. Find similar words
    # ---------------------------------------------------------

    print("\nWORDS SIMILAR TO 'headphones'")
    print("=" * 60)

    similar_words = model.wv.most_similar(
        "headphones",
        topn=3,
    )

    for word, score in similar_words:

        print(
            word,
            round(score, 4)
        )


if __name__ == "__main__":
    main()