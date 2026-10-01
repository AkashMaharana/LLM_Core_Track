import re

from openai import OpenAI
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"


def main():

    # ---------------------------------------------------------
    # 1. OpenAI client
    # ---------------------------------------------------------

    client = OpenAI(
        base_url=endpoint
    )

    # ---------------------------------------------------------
    # 2. Small held-out test set
    #
    # Pretend these products were NOT used for training.
    # ---------------------------------------------------------

    test_products = [
        {
            "description":
                "USB-C fast charging cable, 2 meter length",
            "price": 15,
        },
        {
            "description":
                "Premium wireless noise cancelling headphones "
                "with 40 hour battery life",
            "price": 275,
        },
        {
            "description":
                "65 inch OLED 4K smart television with HDR "
                "and Dolby Atmos",
            "price": 850,
        },
    ]

    predictions = []

    # ---------------------------------------------------------
    # 3. Run zero-shot prediction
    # ---------------------------------------------------------

    for product in test_products:

        prompt = f"""
Estimate the price of the product described below.

Product:
{product["description"]}

Reply only with the estimated price in US dollars.
Example output: $199
""".strip()

        response = client.responses.create(
            model=deployment_name,
            input=prompt,
        )

        output = response.output_text

        # -----------------------------------------------------
        # 4. Extract numeric price from LLM response
        # -----------------------------------------------------

        match = re.search(
            r"\$?(\d+(?:\.\d+)?)",
            output,
        )

        if match:

            predicted_price = float(
                match.group(1)
            )

        else:

            predicted_price = 0.0

        predictions.append(
            predicted_price
        )

        print("\n" + "=" * 70)

        print("PRODUCT")
        print(product["description"])

        print(
            "\nACTUAL PRICE:",
            product["price"]
        )

        print(
            "RAW MODEL RESPONSE:",
            output
        )

        print(
            "EXTRACTED PREDICTION:",
            predicted_price
        )

    # ---------------------------------------------------------
    # 5. Calculate MAE
    # ---------------------------------------------------------

    actual_prices = [
        product["price"]
        for product in test_products
    ]

    mae = mean_absolute_error(
        actual_prices,
        predictions,
    )

    print("\n" + "=" * 70)

    print(
        "GPT-5.1 Zero-Shot MAE:",
        round(mae, 2)
    )


if __name__ == "__main__":
    main()