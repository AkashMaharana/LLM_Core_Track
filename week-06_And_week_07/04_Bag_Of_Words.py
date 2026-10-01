from sklearn.feature_extraction.text import CountVectorizer


def main():

    texts = [
        "wireless headphones",
        "wireless speaker",
        "coffee maker"
    ]

    # ---------------------------------------------------------
    # 1. Create the Bag-of-Words converter
    # ---------------------------------------------------------

    vectorizer = CountVectorizer()

    # ---------------------------------------------------------
    # 2. Learn vocabulary + transform text
    # ---------------------------------------------------------

    vectors = vectorizer.fit_transform(
        texts
    )

    # ---------------------------------------------------------
    # 3. Display vocabulary
    # ---------------------------------------------------------

    print("\nVOCABULARY")
    print("=" * 60)

    print(
        vectorizer.get_feature_names_out()
    )

    # ---------------------------------------------------------
    # 4. Convert sparse matrix to normal array
    # ---------------------------------------------------------

    print("\nVECTORS")
    print("=" * 60)

    print(
        vectors.toarray()
    )


if __name__ == "__main__":
    main()