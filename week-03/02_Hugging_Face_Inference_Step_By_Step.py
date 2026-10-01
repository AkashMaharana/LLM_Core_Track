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

    text = "The phone is excellent."

    # ------------------------------------------------
    # Step 1: Load tokenizer and model
    # ------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
    )

    print("\n1. ORIGINAL TEXT")
    print(text)

    # ------------------------------------------------
    # Step 2: Tokenize
    # ------------------------------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    print("\n2. TOKEN IDs")
    print(inputs["input_ids"])

    # Convert IDs back to readable tokens
    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    print("\n3. TOKENS")
    print(tokens)

    # ------------------------------------------------
    # Step 3: Run model
    # ------------------------------------------------

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    print("\n4. LOGITS - RAW MODEL SCORES")
    print(logits)

    # ------------------------------------------------
    # Step 4: Convert logits to probabilities
    # ------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    print("\n5. PROBABILITIES")
    print(probabilities)

    # ------------------------------------------------
    # Step 5: Pick highest probability
    # ------------------------------------------------

    predicted_class = torch.argmax(
        probabilities,
        dim=-1
    )

    class_id = predicted_class.item()

    label = model.config.id2label[class_id]

    print("\n6. PREDICTED CLASS ID")
    print(class_id)

    print("\n7. FINAL LABEL")
    print(label)

    # Show each class clearly
    print("\n8. CLASS PROBABILITIES")

    for class_id, probability in enumerate(
        probabilities[0]
    ):
        label = model.config.id2label[class_id]

        print(
            f"{label}: {probability.item():.4f}"
        )


if __name__ == "__main__":
    main()