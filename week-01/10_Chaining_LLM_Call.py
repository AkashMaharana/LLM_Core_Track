from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

REVIEW = """
    The phone looks fantastic and the camera is excellent.
    Battery life is disappointing though.
    """

INSTRUCTIONS_1 = """
        Extract only the product characteristics explicitly mentioned
        in the review.

        Return only the characteristics, one per line.
        """

INSTRUCTIONS_2 = """
        For each provided product characteristic, identify the sentiment
        as POSITIVE, NEGATIVE, or MIXED.

        Do not add any new characteristics.
        """

def main():
    client = OpenAI(
        base_url=endpoint
    )

    # Step 1: Extract product characteristics
    aspect_response = client.responses.create(
                model = deployment_name,
                instructions = INSTRUCTIONS_1,
                input=REVIEW
            )

    aspects = aspect_response.output_text

    print("Extracted aspects:")
    print(aspects)

    # Step 2: Find sentiment for the extracted characteristics
    sentiment_response = client.responses.create(
                    model = deployment_name,
                    instructions = INSTRUCTIONS_2,
                    input=f"""
        Review:
        {REVIEW}

        Characteristics:
        {aspects}
        """
                )

    print("\nSentiment result:")
    print(sentiment_response.output_text)





if __name__ == "__main__":
    main()