from openai import OpenAI
import gradio as gr
import json
from dotenv import load_dotenv
import sqlite3

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

load_dotenv()

def main():
    client = OpenAI(
        base_url=endpoint
    )

    # Create a small SQLite database
    connection = sqlite3.connect("airline.db")
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_prices (
            destination TEXT PRIMARY KEY,
            price INTEGER
        )
    """)

    cursor.executemany(
        """
        INSERT OR REPLACE INTO ticket_prices(destination, price)
        VALUES (?, ?)
        """,
        [
            ("Paris", 149),
            ("Rome", 179),
            ("Berlin", 129)
        ]
    )

    connection.commit()

    def get_ticket_price(destination):
        cursor.execute(
            """
            SELECT price
            FROM ticket_prices
            WHERE LOWER(destination) = LOWER(?)
            """,
            (destination,)
        )

        row = cursor.fetchone()

        if row:
            return f"${row[0]}"

        return "Price not available"

    price_tool = {
        "type": "function",
        "name": "get_ticket_price",
        "description": "Get the current ticket price for a destination.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string"
                }
            },
            "required": ["destination"],
            "additionalProperties": False
        },
        "strict": True
    }

    user_question = "How much is a ticket to Paris?"

    response = client.responses.create(
        model="gpt-5.1",
        input=user_question,
        tools=[price_tool]
    )

    for item in response.output:

        if item.type == "function_call":

            args = json.loads(item.arguments)

            price = get_ticket_price(
                args["destination"]
            )

            print("Tool result:", price)

            response = client.responses.create(
                model=deployment_name,
                previous_response_id=response.id,
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": price
                    }
                ],
                tools=[price_tool]
            )

    print("\nAssistant:")
    print(response.output_text)

    connection.close()


if __name__ == "__main__":
    main()