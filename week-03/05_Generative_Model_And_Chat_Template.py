import ssl

import httpx
import torch
import truststore

from huggingface_hub import set_client_factory
from transformers import AutoTokenizer, AutoModelForCausalLM


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

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # -----------------------------------------
    # Load tokenizer
    # -----------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # -----------------------------------------
    # Load generative model
    # -----------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    messages = [
        {
            "role": "system",
            "content": "You are a concise technical assistant."
        },
        {
            "role": "user",
            "content": "Explain Redis in one sentence."
        }
    ]

    # -----------------------------------------
    # Apply the model's chat template
    # -----------------------------------------

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print("\nFORMATTED PROMPT")
    print(formatted_prompt)

    # -----------------------------------------
    # Convert prompt → PyTorch tensors
    # -----------------------------------------

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    # -----------------------------------------
    # Generate response
    # -----------------------------------------

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

    # -----------------------------------------
    # Remove original prompt tokens
    # -----------------------------------------

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    # -----------------------------------------
    # Convert generated token IDs → text
    # -----------------------------------------

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    print("\nMODEL RESPONSE")
    print(response)


if __name__ == "__main__":
    main()