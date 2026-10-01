from sklearn.ensemble import RandomForestRegressor


def main():

    # ---------------------------------------------------------
    # Training data
    #
    # Features:
    # [weight_kg, description_length]
    # ---------------------------------------------------------

    X_train = [
        [1, 20],
        [2, 25],
        [3, 30],
        [5, 60],
        [8, 100],
        [10, 120],
    ]

    # Actual prices
    y_train = [
        20,
        30,
        50,
        150,
        400,
        500,
    ]

    # ---------------------------------------------------------
    # 1. Create Random Forest
    # ---------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    # ---------------------------------------------------------
    # 2. Train
    # ---------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # 3. New products
    # ---------------------------------------------------------

    X_test = [
        [2.5, 28],
        [9, 110],
    ]

    # ---------------------------------------------------------
    # 4. Predict
    # ---------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    print(
        "Prediction 1:",
        round(predictions[0], 2)
    )

    print(
        "Prediction 2:",
        round(predictions[1], 2)
    )


if __name__ == "__main__":
    main()