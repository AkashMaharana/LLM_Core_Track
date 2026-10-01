from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"


def main():
    client = OpenAI(
        base_url=endpoint
    )

    question = """
Explain Kafka consumer groups to a backend engineer
who understands basic messaging systems.
"""

    answer_a = """
A Kafka consumer group is a collection of consumers
working together to consume records from a topic.

Each partition is assigned to only one consumer within
a consumer group at a time.

If there are more consumers than partitions, some
consumers remain idle.

When consumers join or leave the group, Kafka may
redistribute partition ownership through a rebalance.
"""

    answer_b = """
Kafka consumer groups allow applications to process
messages in parallel.

All consumers can consume messages from Kafka and Kafka
automatically manages everything.

Adding more consumers always increases throughput because
Kafka distributes the messages between all consumers.
"""

    prompt = f"""
You are evaluating two technical answers.

QUESTION:

{question}

ANSWER A:

{answer_a}

ANSWER B:

{answer_b}

Evaluate both answers using these criteria:

1. Technical correctness
2. Completeness
3. Clarity
4. Conciseness

For each answer:
- Give each criterion a score from 1 to 5.
- Briefly explain important strengths or problems.

Then state which answer better satisfies the question.

Do not prefer an answer because it is longer.
Focus on factual correctness first.
"""

    response = client.responses.create(
        model=deployment_name,
        input=prompt
    )

    print(response.output_text)


if __name__ == "__main__":
    main()