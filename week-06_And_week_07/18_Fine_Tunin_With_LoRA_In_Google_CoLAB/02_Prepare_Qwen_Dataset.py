from datasets import Dataset
from transformers import AutoTokenizer


def main():

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # ---------------------------------------------------------
    # 1. Raw curated products
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
    # 2. Convert products into chat training examples
    # ---------------------------------------------------------

    training_examples = []

    for product in products:

        example = {
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

        training_examples.append(
            example
        )

    # ---------------------------------------------------------
    # 3. Convert Python list → Hugging Face Dataset
    # ---------------------------------------------------------

    dataset = Dataset.from_list(
        training_examples
    )

    print("\nDATASET")
    print("=" * 70)

    print(dataset)

    print(
        "\nNumber of examples:",
        len(dataset)
    )

    # ---------------------------------------------------------
    # 4. Inspect first structured example
    # ---------------------------------------------------------

    first_example = dataset[0]

    print("\nFIRST RAW EXAMPLE")
    print("=" * 70)

    print(
        first_example
    )

    # ---------------------------------------------------------
    # 5. Load Qwen tokenizer
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # ---------------------------------------------------------
    # 6. Apply Qwen's chat template
    # ---------------------------------------------------------

    formatted_text = tokenizer.apply_chat_template(
        first_example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    print("\nFORMATTED CHAT")
    print("=" * 70)

    print(
        formatted_text
    )

    # ---------------------------------------------------------
    # 7. Tokenize using the chat template directly
    # ---------------------------------------------------------

    token_ids = tokenizer.apply_chat_template(
        first_example["messages"],
        tokenize=True,
        add_generation_prompt=False,
    )

    print("\nTOKEN IDS")
    print("=" * 70)

    print(
        token_ids
    )

    print(
        "\nNumber of tokens:",
        len(token_ids)
    )

    # ---------------------------------------------------------
    # 8. Decode tokens back into text
    # ---------------------------------------------------------

    decoded_text = tokenizer.decode(
        token_ids
    )

    print("\nDECODED TOKENS")
    print("=" * 70)

    print(
        decoded_text
    )


if __name__ == "__main__":
    main()