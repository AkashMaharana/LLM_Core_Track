# !pip install -q transformers accelerate bitsandbytes
# This script demonstrates how to transcribe a meeting audio file and generate meeting minutes using an open-source LLM in 4-bit quantization. 
# It needs to be executed in Google Colab Envirobnment with GPU enabled. 
# The script uses the Qwen2.5-7B-Instruct model for demonstration.

# Colab Link : https://colab.research.google.com/drive/1MoOJDvImvt1_Ze729In_NeCfCSQPxfxQ?usp=sharing

import os
import torch

from openai import OpenAI
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
)


def main():

    # -----------------------------------------
    # 1. Transcribe meeting audio
    # -----------------------------------------

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )

    with open("DIALOGUE.ogg", "rb") as audio_file:

        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file
        )

    transcript = transcription.text

    print("\nTRANSCRIPT")
    print("-" * 80)
    print(transcript)

    # -----------------------------------------
    # 2. Load open-source LLM in 4-bit
    # -----------------------------------------

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # -----------------------------------------
    # 3. Build meeting-minutes prompt
    # -----------------------------------------

    messages = [
        {
            "role": "system",
            "content": """
You create concise and accurate meeting minutes.

Return the result in Markdown with:

# Meeting Minutes

## Summary

## Key Discussion Points

## Decisions

## Action Items

For each action item include the owner if one is clearly mentioned.
Do not invent information that is not present in the transcript.
"""
        },
        {
            "role": "user",
            "content": f"""
Create meeting minutes from this transcript:

{transcript}
"""
        }
    ]

    # -----------------------------------------
    # 4. Apply chat template
    # -----------------------------------------

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # -----------------------------------------
    # 5. Stream generated minutes
    # -----------------------------------------

    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    print("\nMEETING MINUTES")
    print("-" * 80)

    with torch.no_grad():

        model.generate(
            **inputs,
            max_new_tokens=800,
            do_sample=False,
            streamer=streamer
        )


if __name__ == "__main__":
    main()