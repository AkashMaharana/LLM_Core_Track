import re

from openai import OpenAI
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

def main():

    client = OpenAI(
        base_url=endpoint
    )

    # ---------------------------------------------------------
    # 1. Few-shot examples
    #
    # These examples are demonstrations.
    # They are NOT the test products.
    # ---------------------------------------------------------

    examples = [
        {
            "description": "USB-C fast charging cable",
            "price": 15,
        },
        {
            "description":
                "Premium wireless noise cancelling headphones",
            "price": 275,
        },
        {
            "description":
                "65 inch OLED 4K smart television",
            "price": 850,
        },
    ]

    # ---------------------------------------------------------
    # 2. Held-out test set
    # ---------------------------------------------------------

    test_products = [
        {
            "description":
                "2 meter braided USB charging cable",
            "price": 18,
        },
        {
            "description":
                "Bluetooth over-ear headphones with "
                "active noise cancellation",
            "price": 240,
        },
        {
            "description":
                "55 inch OLED 4K HDR smart television",
            "price": 700,
        },
    ]

    predictions = []

    # ---------------------------------------------------------
    # 3. Build reusable examples text
    # ---------------------------------------------------------

    examples_text = ""

    for example in examples:

        examples_text += f"""
Product:
{example["description"]}

Price:
${example["price"]}

"""

    # ---------------------------------------------------------
    # 4. Predict each test product
    # ---------------------------------------------------------

    for product in test_products:

        prompt = f"""
You estimate product prices.

Use the examples below to understand the expected
price scale and output format.

EXAMPLES:

{examples_text}

NOW ESTIMATE:

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
        # 5. Extract numeric price
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

        print("PRODUCT:")
        print(
            product["description"]
        )

        print(
            "ACTUAL:",
            product["price"]
        )

        print(
            "PREDICTED:",
            predicted_price
        )

    # ---------------------------------------------------------
    # 6. Evaluate
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
        "GPT-5.1 Few-Shot MAE:",
        round(mae, 2)
    )


if __name__ == "__main__":
    main()