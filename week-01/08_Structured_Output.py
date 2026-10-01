from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

INSTRUCTIONS = """
        Analyze the product review.

        Identify only product characteristics explicitly
        mentioned in the review.
        """

REVIEW = """
    The screen is excellent and the camera takes great photos,
    but battery life is very disappointing.
    """

def main():
    client = OpenAI(
        base_url=endpoint
    )

    response = client.responses.create(
                model = deployment_name,
                instructions = INSTRUCTIONS,
                input = REVIEW,
                text={
            "format": {
                "type": "json_schema",
                "name": "review_sentiment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "sentiments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "aspect": {
                                        "type": "string"
                                    },
                                    "sentiment": {
                                        "type": "string",
                                        "enum": [
                                            "Screen Quality",
                                            "Camera",
                                            "Battery Life",
                                            "Build Quality",
                                            "Performance"
                                        ]
                                    }
                                },
                                "required": [
                                    "aspect",
                                    "sentiment"
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["sentiments"],
                    "additionalProperties": False
                }
            }
        }
            )
    result = json.loads(response.output_text)

    print(result)


if __name__ == "__main__":
    main()