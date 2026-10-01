from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression


def main():

    # ---------------------------------------------------------
    # Training data
    # ---------------------------------------------------------

    train_texts = [
        "basic usb charging cable",
        "portable desk fan",
        "stainless steel coffee maker",
        "premium wireless headphones",
        "large 4k computer monitor",
        "premium oled television",
    ]

    train_prices = [
        10,
        25,
        130,
        250,
        400,
        800,
    ]

    # ---------------------------------------------------------
    # Test data
    # ---------------------------------------------------------

    test_texts = [
        "wireless headphones",
        "4k monitor",
        "usb cable",
    ]

    # ---------------------------------------------------------
    # 1. Convert training text → TF-IDF features
    # ---------------------------------------------------------

    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(
        train_texts
    )

    # ---------------------------------------------------------
    # 2. Train regression model
    # ---------------------------------------------------------

    model = LinearRegression()

    model.fit(
        X_train,
        train_prices
    )

    # ---------------------------------------------------------
    # 3. Transform test text using SAME vocabulary/statistics
    # ---------------------------------------------------------

    X_test = vectorizer.transform(
        test_texts
    )

    # ---------------------------------------------------------
    # 4. Predict prices
    # ---------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # ---------------------------------------------------------
    # 5. Display
    # ---------------------------------------------------------

    for text, prediction in zip(
        test_texts,
        predictions
    ):

        print(
            text,
            "→ predicted price:",
            round(prediction, 2)
        )


if __name__ == "__main__":
    main()