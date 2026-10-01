import ssl

import httpx
import torch
import truststore

from huggingface_hub import set_client_factory
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():

    # Corporate SSL certificate support
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

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # 2. Load model + trained weights
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
    )

    reviews = [
        "The battery life is absolutely fantastic.",
        "The camera quality is terrible.",
        "The screen is good but the battery is disappointing."
    ]

    # 3. Convert text into tensors
    inputs = tokenizer(
        reviews,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # 4. Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. Raw model output
    logits = outputs.logits

    # 6. Convert logits → probabilities
    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    # 7. Select highest probability label
    predicted_classes = torch.argmax(
        probabilities,
        dim=-1
    )

    for review, probs, class_id in zip(
        reviews,
        probabilities,
        predicted_classes
    ):
        label = model.config.id2label[
            class_id.item()
        ]

        score = probs[
            class_id.item()
        ].item()

        print(f"\nReview: {review}")
        print(f"Sentiment: {label}")
        print(f"Score: {score:.4f}")


if __name__ == "__main__":
    main()