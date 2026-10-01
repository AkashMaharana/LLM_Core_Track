import ssl

import httpx
import torch
import truststore

from huggingface_hub import set_client_factory
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():

    # Corporate SSL setup
    ssl_context = truststore.SSLContext(
        ssl.PROTOCOL_TLS_CLIENT
    )

    set_client_factory(
        lambda: httpx.Client(
            verify=ssl_context
        )
    )

    model_name = (
        "distilbert/"
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
    )

    reviews = [
        "The phone is excellent.",
        "The phone is excellent but the battery life is disappointing."
    ]

    # -----------------------------------------
    # Tokenize both sentences together
    # -----------------------------------------

    inputs = tokenizer(
        reviews,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    print("\nINPUT IDS")
    print(inputs["input_ids"])

    print("\nATTENTION MASK")
    print(inputs["attention_mask"])

    # -----------------------------------------
    # Show readable tokens
    # -----------------------------------------

    print("\nTOKENS")

    for token_ids in inputs["input_ids"]:

        tokens = tokenizer.convert_ids_to_tokens(
            token_ids
        )

        print(tokens)

    # -----------------------------------------
    # Run model
    # -----------------------------------------

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    # -----------------------------------------
    # Show final results
    # -----------------------------------------

    for review, probs in zip(
        reviews,
        probabilities
    ):

        class_id = torch.argmax(
            probs
        ).item()

        label = model.config.id2label[
            class_id
        ]

        print(f"\nReview: {review}")

        print(
            f"NEGATIVE: {probs[0].item():.4f}"
        )

        print(
            f"POSITIVE: {probs[1].item():.4f}"
        )

        print(
            f"Final: {label}"
        )


if __name__ == "__main__":
    main()