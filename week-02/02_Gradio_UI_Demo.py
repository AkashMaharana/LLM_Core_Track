from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

load_dotenv()


def main():
    client = OpenAI(
        base_url=endpoint
    )

    def analyze_review(review):
        response = client.responses.create(
            model=deployment_name,
            instructions="""
            You are a retail product review analyst.

            Identify only the product characteristics explicitly
            mentioned in the review.

            For each characteristic, classify the sentiment as:
            POSITIVE, NEGATIVE, or MIXED.

            Do not infer characteristics that are not mentioned.
            """,
            input=review
        )

        return response.output_text

    demo = gr.Interface(
        fn=analyze_review,
        inputs=gr.Textbox(
            lines=6,
            label="Customer Review"
        ),
        outputs=gr.Textbox(
            lines=8,
            label="Sentiment Analysis"
        ),
        title="Review Sentiment Analyzer"
    )

    demo.launch(share=True)


if __name__ == "__main__":
    main()