from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error


def main():

    # ---------------------------------------------------------
    # 1. Training data
    # ---------------------------------------------------------

    train_texts = [
        "basic usb charging cable",
        "cheap usb phone cable",
        "small portable desk fan",
        "stainless steel coffee maker",
        "programmable coffee machine",
        "premium wireless headphones",
        "noise cancelling wireless headphones",
        "large 4k computer monitor",
        "27 inch gaming monitor",
        "premium oled television",
        "large 65 inch 4k oled television",
    ]

    train_prices = [
        10,
        15,
        25,
        130,
        160,
        250,
        300,
        400,
        450,
        800,
        900,
    ]

    # ---------------------------------------------------------
    # 2. Test data
    # ---------------------------------------------------------

    test_texts = [
        "usb charging cable",
        "wireless noise cancelling headphones",
        "4k gaming monitor",
        "65 inch oled television",
    ]

    test_prices = [
        12,
        275,
        425,
        850,
    ]

    # ---------------------------------------------------------
    # 3. Create TF-IDF converter
    # ---------------------------------------------------------

    vectorizer = TfidfVectorizer()

    # Learn vocabulary/statistics from TRAINING text
    X_train = vectorizer.fit_transform(
        train_texts
    )

    # Use same learned vocabulary on TEST text
    X_test = vectorizer.transform(
        test_texts
    )

    # ---------------------------------------------------------
    # 4. Create LinearSVR model
    # ---------------------------------------------------------

    model = LinearSVR(
        C=1.0,
        epsilon=10.0,
        random_state=42,
    )

    # ---------------------------------------------------------
    # 5. Train
    # ---------------------------------------------------------

    model.fit(
        X_train,
        train_prices
    )

    # ---------------------------------------------------------
    # 6. Predict
    # ---------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # ---------------------------------------------------------
    # 7. Display individual predictions
    # ---------------------------------------------------------

    print("\nPREDICTIONS")
    print("=" * 70)

    for text, actual, predicted in zip(
        test_texts,
        test_prices,
        predictions,
    ):

        print("\nProduct:")
        print(text)

        print(
            "Actual:",
            actual
        )

        print(
            "Predicted:",
            round(predicted, 2)
        )

        print(
            "Error:",
            round(
                abs(actual - predicted),
                2
            )
        )

    # ---------------------------------------------------------
    # 8. Evaluate entire model
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        test_prices,
        predictions
    )

    print("\n" + "=" * 70)

    print(
        "Mean Absolute Error:",
        round(mae, 2)
    )


if __name__ == "__main__":
    main()