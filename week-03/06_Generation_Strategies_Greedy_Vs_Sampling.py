import ssl

import httpx
import torch
import truststore

from huggingface_hub import set_client_factory
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():

    ssl_context = truststore.SSLContext(
        ssl.PROTOCOL_TLS_CLIENT
    )

    set_client_factory(
        lambda: httpx.Client(
            verify=ssl_context
        )
    )

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    messages = [
        {
            "role": "user",
            "content": "Give me one creative use case for Kafka."
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    # -----------------------------------
    # Greedy generation
    # -----------------------------------

    with torch.no_grad():

        greedy_output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

    greedy_tokens = greedy_output[0][
        inputs["input_ids"].shape[1]:
    ]

    greedy_response = tokenizer.decode(
        greedy_tokens,
        skip_special_tokens=True
    )

    print("\nGREEDY")
    print(greedy_response)

    # -----------------------------------
    # Sampling
    # -----------------------------------

    with torch.no_grad():

        sampled_output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )

    sampled_tokens = sampled_output[0][
        inputs["input_ids"].shape[1]:
    ]

    sampled_response = tokenizer.decode(
        sampled_tokens,
        skip_special_tokens=True
    )

    print("\nSAMPLED")
    print(sampled_response)


if __name__ == "__main__":
    main()