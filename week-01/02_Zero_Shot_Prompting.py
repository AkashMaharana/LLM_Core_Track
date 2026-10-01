# Install OpenAI sdk before executing this script
# pip install -U openai

from openai import OpenAI

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

client = OpenAI(
	base_url=endpoint
)

INSTRUCTIONS = """
Identify product characteristics mentioned by the customer
and classify each as Positive, Negative or Mixed.

Do not infer characteristics that are not explicitly supported
by the review.

Ignore delivery and packaging feedback.
"""

# USER_INPUT
REVIEWS = [
    "The screen is gorgeous but the battery barely lasts until evening.",

    "Excellent camera, excellent screen, excellent battery. Absolutely love it.",

    "The phone looks premium. Delivery was three days late and the box was damaged.",

    "Battery life could be better, although it's not terrible. Camera quality is quite impressive.",

    "Bought it last week. It's a phone."
]

def analyze_review() -> None:

	for sample_review in REVIEWS:
		response = client.responses.create(
			model = deployment_name,
			instructions = INSTRUCTIONS,
			input = sample_review
		)

		print("\n --- RESPONSE --- \n")
		print(response.output_text)

		print("\n --- Next --- \n")

if __name__ == "__main__":
	analyze_review()