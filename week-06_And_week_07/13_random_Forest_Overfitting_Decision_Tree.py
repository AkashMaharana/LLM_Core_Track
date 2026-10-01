from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


def main():

    # ---------------------------------------------------------
    # Training data
    # One feature: some simplified product score
    # ---------------------------------------------------------

    X_train = [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
    ]

    y_train = [
        20,
        35,
        50,
        65,
        85,
        110,
        150,
        220,
    ]

    # ---------------------------------------------------------
    # Test data
    # New unseen points
    # ---------------------------------------------------------

    X_test = [
        [1.5],
        [2.5],
        [4.5],
        [6.5],
        [7.5],
    ]

    y_test = [
        28,
        42,
        75,
        130,
        185,
    ]

    # ---------------------------------------------------------
    # 1. Very deep tree
    # ---------------------------------------------------------

    model = DecisionTreeRegressor(
        max_depth=None,
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
    # 3. Predictions
    # ---------------------------------------------------------

    train_predictions = model.predict(
        X_train
    )

    test_predictions = model.predict(
        X_test
    )

    # ---------------------------------------------------------
    # 4. Evaluate
    # ---------------------------------------------------------

    train_mae = mean_absolute_error(
        y_train,
        train_predictions,
    )

    test_mae = mean_absolute_error(
        y_test,
        test_predictions,
    )

    print(
        "Training MAE:",
        round(train_mae, 2)
    )

    print(
        "Test MAE:",
        round(test_mae, 2)
    )


if __name__ == "__main__":
    main()