import os

import requests

from openai import OpenAI


# =============================================================
# CONFIGURATION
# =============================================================

MODEL = "gpt-5.1"

PUSHOVER_URL = (
    "https://api.pushover.net/1/messages.json"
)

# -------------------------------------------------------------
# False:
#   print the notification only
#
# True:
#   actually call Pushover
# -------------------------------------------------------------

ENABLE_PUSH = False


# =============================================================
# MESSENGER AGENT
# =============================================================

class MessengerAgent:

    def __init__(self):

        self.client = OpenAI()

        self.pushover_user = (
            os.environ.get(
                "PUSHOVER_USER"
            )
        )

        self.pushover_token = (
            os.environ.get(
                "PUSHOVER_TOKEN"
            )
        )

    # ---------------------------------------------------------
    # Ask GPT-5.1 to create a concise notification
    # ---------------------------------------------------------

    def craft_message(
        self,
        description,
        deal_price,
        estimated_value,
    ):

        discount = (
            estimated_value
            - deal_price
        )

        discount_percentage = (
            discount
            / estimated_value
            * 100
            if estimated_value > 0
            else 0
        )

        prompt = f"""
Create a concise push notification for a strong product deal.

Product:
{description}

Current selling price:
${deal_price:.2f}

Estimated fair value:
${estimated_value:.2f}

Estimated saving:
${discount:.2f}

Estimated discount:
{discount_percentage:.1f}%

Write 2 short sentences.

Requirements:

- Be factual.
- Mention the selling price.
- Mention the estimated fair value.
- Mention the estimated saving.
- Do not exaggerate beyond the supplied numbers.
- Do not include a URL.
- Return only the notification text.
"""

        response = (
            self.client.responses.create(
                model=MODEL,
                input=prompt,
            )
        )

        return (
            response.output_text
            .strip()
        )

    # ---------------------------------------------------------
    # Send notification through Pushover
    # ---------------------------------------------------------

    def send_push(
        self,
        message,
        url,
    ):

        # -----------------------------------------------------
        # Safe demo mode
        # -----------------------------------------------------

        if not ENABLE_PUSH:

            print()

            print(
                "=" * 70
            )

            print(
                "PUSH NOTIFICATION PREVIEW"
            )

            print(
                "=" * 70
            )

            print(
                message
            )

            print()

            print(
                "URL:",
                url
            )

            return {
                "status":
                    "preview_only"
            }

        # -----------------------------------------------------
        # Actual Pushover mode
        # -----------------------------------------------------

        if not self.pushover_user:

            raise RuntimeError(
                "PUSHOVER_USER is not set."
            )

        if not self.pushover_token:

            raise RuntimeError(
                "PUSHOVER_TOKEN is not set."
            )

        payload = {

            "token":
                self.pushover_token,

            "user":
                self.pushover_user,

            "title":
                "Deal Alert",

            "message":
                message,

            "sound":
                "cashregister",

            "url":
                url,

            "url_title":
                "View Deal",
        }

        response = requests.post(
            PUSHOVER_URL,
            data=payload,
            timeout=15,
        )

        response.raise_for_status()

        return response.json()

    # ---------------------------------------------------------
    # Public operation
    #
    # Planner will eventually call this.
    # ---------------------------------------------------------

    def notify(
        self,
        description,
        deal_price,
        estimated_value,
        url,
    ):

        message = (
            self.craft_message(

                description=description,

                deal_price=deal_price,

                estimated_value=(
                    estimated_value
                ),
            )
        )

        push_result = (
            self.send_push(
                message,
                url,
            )
        )

        return {

            "message":
                message,

            "deal_price":
                deal_price,

            "estimated_value":
                estimated_value,

            "discount":
                estimated_value
                - deal_price,

            "delivery":
                push_result,
        }


# =============================================================
# MAIN
# =============================================================

def main():

    if not os.environ.get(
        "OPENAI_API_KEY"
    ):

        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    messenger = (
        MessengerAgent()
    )

    # ---------------------------------------------------------
    # Imagine Planner already found this opportunity.
    # ---------------------------------------------------------

    description = (
        "55 inch OLED 4K HDR Smart TV "
        "with Dolby Vision, Dolby Atmos "
        "and four HDMI ports"
    )

    deal_price = 299.00

    estimated_value = 650.00

    url = (
        "https://example.com/deal"
    )

    result = (
        messenger.notify(

            description=description,

            deal_price=deal_price,

            estimated_value=(
                estimated_value
            ),

            url=url,
        )
    )

    print()

    print(
        "=" * 70
    )

    print(
        "MESSENGER AGENT RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "Message:",
        result[
            "message"
        ]
    )

    print(
        "Discount:",
        result[
            "discount"
        ]
    )

    print(
        "Delivery:",
        result[
            "delivery"
        ]
    )


if __name__ == "__main__":
    main()