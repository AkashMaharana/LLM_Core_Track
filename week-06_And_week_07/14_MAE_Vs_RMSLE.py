import math

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_log_error,
)

# RMSLE : Root Mean Squared Logarithmic Error

def main():

    # ---------------------------------------------------------
    # Actual prices
    # ---------------------------------------------------------

    actual_prices = [
        20,
        900,
    ]

    # ---------------------------------------------------------
    # Predictions
    #
    # Both predictions are wrong by exactly $50.
    # ---------------------------------------------------------

    predicted_prices = [
        70,
        950,
    ]

    # ---------------------------------------------------------
    # 1. MAE
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        actual_prices,
        predicted_prices,
    )

    # ---------------------------------------------------------
    # 2. MSLE
    # ---------------------------------------------------------

    msle = mean_squared_log_error(
        actual_prices,
        predicted_prices,
    )

    # ---------------------------------------------------------
    # 3. RMSLE
    # ---------------------------------------------------------

    rmsle = math.sqrt(
        msle
    )

    print(
        "MAE:",
        round(mae, 4)
    )

    print(
        "RMSLE:",
        round(rmsle, 4)
    )


if __name__ == "__main__":
    main()