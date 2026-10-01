from sklearn.feature_extraction.text import TfidfVectorizer


def main():

    texts = [
        "wireless headphones",
        "wireless speaker",
        "coffee maker",
    ]

    # ---------------------------------------------------------
    # 1. Create TF-IDF converter
    # ---------------------------------------------------------

    vectorizer = TfidfVectorizer()

    # ---------------------------------------------------------
    # 2. Learn vocabulary + TF-IDF statistics
    # ---------------------------------------------------------

    vectors = vectorizer.fit_transform(
        texts
    )

    # ---------------------------------------------------------
    # 3. Show vocabulary
    # ---------------------------------------------------------

    print("\nVOCABULARY")
    print("=" * 60)

    print(
        vectorizer.get_feature_names_out()
    )

    # ---------------------------------------------------------
    # 4. Show TF-IDF values
    # ---------------------------------------------------------

    print("\nTF-IDF MATRIX")
    print("=" * 60)

    print(
        vectors.toarray()
    )


if __name__ == "__main__":
    main()