import ssl

import httpx
import truststore
from huggingface_hub import set_client_factory
from transformers import pipeline


def main():

    # Use macOS system certificates.
    # Useful on corporate networks where the company CA
    # is trusted by macOS but not by Python's default CA bundle.
    ssl_context = truststore.SSLContext(
        ssl.PROTOCOL_TLS_CLIENT
    )

    set_client_factory(
        lambda: httpx.Client(
            verify=ssl_context
        )
    )

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

    reviews = [
        "The battery life is absolutely fantastic.",
        "The camera quality is terrible.",
        "The screen is good but the battery is disappointing."
    ]

    results = classifier(reviews)

    for review, result in zip(reviews, results):
        print(f"\nReview: {review}")
        print(f"Sentiment: {result['label']}")
        print(f"Score: {result['score']:.4f}")


if __name__ == "__main__":
    main()