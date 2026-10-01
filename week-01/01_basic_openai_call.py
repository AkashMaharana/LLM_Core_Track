# Install OpenAI sdk before executing this script
# pip install -U openai

from openai import OpenAI
from dotenv import load_dotenv

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

client = OpenAI(
	base_url=endpoint
)

load_dotenv()

"""
INSTRUCTIONS = "Analyze this review."
"""

'''
INSTRUCTIONS = """
Identify the positive and negative product characteristics
mentioned in this review.
"""
'''

INSTRUCTIONS = """
You are a retail product review analyst.

Analyze only characteristics explicitly mentioned by the customer.
Do not infer anything that is not present in the review.

Separate the response into:
1. Positive feedback
2. Negative feedback
"""

# USER_INPUT
REVIEW = """
I love the screen quality and the phone feels very premium.
The camera is fantastic during the day, but night photos are disappointing.
Battery life is also much worse than I expected.
"""

def analyze_review(review: str) -> None:
	response = client.responses.create(
		model = deployment_name,
		instructions = INSTRUCTIONS,
		input = review
	)

	print("\n --- RESPONSE --- \n")
	print(response.output_text)

	print("\n --- USAGES --- \n")
	print(f"Input Tokenss : {response.usage.input_tokens}")
	print(f"Output Tokens : {response.usage.output_tokens}")
	print(f"Total Tokens : {response.usage.total_tokens}")

if __name__ == "__main__":
	analyze_review(REVIEW)