from transformers import AutoTokenizer


def main():

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct"
    )

    products = [
        {
            "title": "Stainless Steel Coffee Maker",
            "description": (
                "Programmable 12 cup coffee maker with "
                "thermal carafe, automatic shutoff and "
                "adjustable brew strength."
            ),
            "price": 129.99,
        },
        {
            "title": "Wireless Noise Cancelling Headphones",
            "description": (
                "Bluetooth headphones with active noise "
                "cancellation, 40 hour battery life and "
                "USB-C fast charging."
            ),
            "price": 249.00,
        },
        {
            "title": "Portable Desk Fan",
            "description": (
                "Compact USB powered desk fan with "
                "three speed settings."
            ),
            "price": 24.99,
        },
    ]

    max_tokens = 80

    for product in products:

        text = (
            product["title"]
            + "\n"
            + product["description"]
        )

        tokens = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        tokens = tokens[:max_tokens]

        cleaned_text = tokenizer.decode(
            tokens,
            skip_special_tokens=True,
        )

        training_prompt = f"""
How much does this cost to the nearest dollar?

{cleaned_text}

Price is ${round(product["price"]):.2f}
""".strip()

        test_prompt = f"""
How much does this cost to the nearest dollar?

{cleaned_text}
""".strip()

        print("\n" + "=" * 70)

        print("TOKEN COUNT:")
        print(len(tokens))

        print("\nTRAINING EXAMPLE:")
        print(training_prompt)

        print("\nTEST INPUT:")
        print(test_prompt)


if __name__ == "__main__":
    main()