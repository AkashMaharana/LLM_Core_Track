from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv
import base64

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

load_dotenv()


def main():
    client = OpenAI(
        base_url=endpoint
    )

    image_path = "/Users/in46032455/Documents/My Learnings/LLM_0_To_1/Core_Track/week-02/iphone.jpeg"

    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    response = client.responses.create(
        model=deployment_name,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Describe what you see in this image."
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}"
                    }
                ]
            }
        ]
    )

    print(response.output_text)


if __name__ == "__main__":
    main()