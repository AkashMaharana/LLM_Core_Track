import modal

# modal secret create messenger-secret \
#  OPENAI_API_KEY=... \
#  PUSHOVER_USER=... \
#  PUSHOVER_TOKEN=...

# =============================================================
# CONFIGURATION
# =============================================================

APP_NAME = (
    "week8-messenger-agent"
)

MODEL = (
    "gpt-5.1"
)

PUSHOVER_URL = (
    "https://api.pushover.net/1/messages.json"
)


# =============================================================
# MODAL APP
# =============================================================

app = modal.App(
    APP_NAME
)


# =============================================================
# SECRET
# =============================================================

messenger_secret = (
    modal.Secret.from_name(

        "messenger-secret",

        required_keys=[
            "OPENAI_API_KEY",
            "PUSHOVER_USER",
            "PUSHOVER_TOKEN",
        ],
    )
)


# =============================================================
# IMAGE
# =============================================================

image = (

    modal.Image
    .debian_slim(
        python_version="3.11"
    )

    .uv_pip_install(
        "openai",
        "requests",
    )
)


# =============================================================
# MESSENGER AGENT
# =============================================================

@app.cls(

    image=image,

    secrets=[
        messenger_secret
    ],

    timeout=120,
)
class MessengerAgent:

    # ---------------------------------------------------------
    # Container startup
    # ---------------------------------------------------------

    @modal.enter()
    def initialize(self):

        import os

        import requests

        from openai import OpenAI

        self.requests = requests

        self.client = OpenAI()

        self.pushover_user = (
            os.environ[
                "PUSHOVER_USER"
            ]
        )

        self.pushover_token = (
            os.environ[
                "PUSHOVER_TOKEN"
            ]
        )

        print(
            "MessengerAgent ready."
        )

    # ---------------------------------------------------------
    # GPT message generation
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
- Mention the saving.
- Do not exaggerate.
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
    # Pushover delivery
    # ---------------------------------------------------------

    def send_push(
        self,
        message,
        url,
    ):

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

        response = (
            self.requests.post(

                PUSHOVER_URL,

                data=payload,

                timeout=15,
            )
        )

        response.raise_for_status()

        return (
            response.json()
        )

    # ---------------------------------------------------------
    # Public remote operation
    # ---------------------------------------------------------

    @modal.method()
    def notify(
        self,
        description,
        deal_price,
        estimated_value,
        url,
    ):

        message = (
            self.craft_message(

                description,

                deal_price,

                estimated_value,
            )
        )

        delivery = (
            self.send_push(
                message,
                url,
            )
        )

        return {

            "message":
                message,

            "discount":
                estimated_value
                - deal_price,

            "delivery":
                delivery,
        }


# =============================================================
# LOCAL TEST
# =============================================================

@app.local_entrypoint()
def main():

    messenger = (
        MessengerAgent()
    )

    result = (
        messenger
        .notify
        .remote(

            (
                "55 inch OLED 4K HDR "
                "Smart TV with Dolby Vision"
            ),

            299.0,

            650.0,

            "https://example.com/deal",
        )
    )

    print(
        result
    )


if __name__ == "__main__":
    pass