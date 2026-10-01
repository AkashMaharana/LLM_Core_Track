import json


def main():

    # ---------------------------------------------------------
    # 1. Example curated products
    # ---------------------------------------------------------

    products = [
        {
            "description": "USB-C fast charging cable",
            "price": 15,
        },
        {
            "description": "Portable desk fan",
            "price": 25,
        },
        {
            "description": "Stainless steel coffee maker",
            "price": 130,
        },
        {
            "description":
                "Premium wireless noise cancelling headphones",
            "price": 275,
        },
        {
            "description": "27 inch 4K gaming monitor",
            "price": 450,
        },
        {
            "description":
                "65 inch OLED 4K smart television",
            "price": 850,
        },
    ]

    # ---------------------------------------------------------
    # 2. Split training and validation
    # ---------------------------------------------------------

    train_products = products[:5]

    validation_products = products[5:]

    # ---------------------------------------------------------
    # 3. Write files
    # ---------------------------------------------------------

    write_jsonl(
        "fine_tune_train.jsonl",
        train_products,
    )

    write_jsonl(
        "fine_tune_validation.jsonl",
        validation_products,
    )

    print(
        "Created fine_tune_train.jsonl"
    )

    print(
        "Created fine_tune_validation.jsonl"
    )


def write_jsonl(filename, products):

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:

        for product in products:

            training_example = {
                "messages": [
                    {
                        "role": "system",
                        "content":
                            "You estimate product prices. "
                            "Reply only with the price "
                            "in US dollars.",
                    },
                    {
                        "role": "user",
                        "content":
                            product["description"],
                    },
                    {
                        "role": "assistant",
                        "content":
                            f"${product['price']}",
                    },
                ]
            }

            json_line = json.dumps(
                training_example
            )

            file.write(
                json_line + "\n"
            )


if __name__ == "__main__":
    main()