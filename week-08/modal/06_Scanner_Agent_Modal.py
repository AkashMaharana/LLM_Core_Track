import modal


# =============================================================
# CONFIGURATION
# =============================================================

APP_NAME = (
    "week8-scanner-agent"
)


# =============================================================
# MODAL APP
# =============================================================

app = modal.App(
    APP_NAME
)


# =============================================================
# OPENAI SECRET
# =============================================================

openai_secret = (
    modal.Secret.from_name(
        "openai-secret",
        required_keys=[
            "OPENAI_API_KEY"
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
        "pydantic",
        "feedparser",
        "beautifulsoup4",
        "requests",
    )
)


# =============================================================
# SCANNER AGENT
# =============================================================

@app.cls(

    image=image,

    secrets=[
        openai_secret
    ],

    timeout=300,
)
class ScannerAgent:

    @modal.enter()
    def initialize(self):

        from openai import OpenAI

        self.client = (
            OpenAI()
        )

        print(
            "ScannerAgent ready."
        )


    @modal.method()
    def scan(
        self,
        seen_urls=None,
    ):

        import feedparser

        import requests

        from bs4 import (
            BeautifulSoup
        )

        from pydantic import (
            BaseModel,
            Field,
        )


        MODEL = "gpt-5.1"

        MAX_SELECTED_DEALS = 5

        FEEDS = [

            (
                "https://www.dealnews.com/"
                "c142/Electronics/?rss=1"
            ),

            (
                "https://www.dealnews.com/"
                "c39/Computers/?rss=1"
            ),

            (
                "https://www.dealnews.com/"
                "f1912/Smart-Home/?rss=1"
            ),
        ]


        class Deal(BaseModel):

            product_description: str = (
                Field(
                    description=(
                        "Detailed product "
                        "description."
                    )
                )
            )

            price: float

            url: str


        class DealSelection(BaseModel):

            deals: list[Deal]


        def clean_html(text):

            soup = BeautifulSoup(
                text or "",
                "html.parser",
            )

            return soup.get_text(
                " ",
                strip=True,
            )


        def fetch_details(url):

            try:

                response = requests.get(
                    url,
                    timeout=10,
                    headers={
                        "User-Agent":
                            "Mozilla/5.0"
                    },
                )

                response.raise_for_status()

                soup = BeautifulSoup(
                    response.text,
                    "html.parser",
                )

                return (
                    soup.get_text(
                        " ",
                        strip=True,
                    )[:1500]
                )

            except requests.RequestException:

                return ""


        # -----------------------------------------------------
        # Fetch RSS
        # -----------------------------------------------------

        scraped = []

        for feed_url in FEEDS:

            feed = (
                feedparser.parse(
                    feed_url
                )
            )

            for entry in (
                feed.entries[:10]
            ):

                url = entry.get(
                    "link",
                    "",
                )

                if not url:
                    continue

                scraped.append(
                    {
                        "title":
                            clean_html(
                                entry.get(
                                    "title",
                                    "",
                                )
                            ),

                        "summary":
                            clean_html(
                                entry.get(
                                    "summary",
                                    "",
                                )
                            ),

                        "details":
                            fetch_details(
                                url
                            ),

                        "url":
                            url,
                    }
                )


        # -----------------------------------------------------
        # Memory filtering
        # -----------------------------------------------------

        seen = set(
            seen_urls or []
        )

        scraped = [

            deal

            for deal in scraped

            if deal["url"]
            not in seen
        ]


        if not scraped:

            return {
                "deals": []
            }


        # -----------------------------------------------------
        # Prompt
        # -----------------------------------------------------

        text_blocks = []

        for deal in scraped:

            text_blocks.append(

                f"""
Title: {deal['title']}

Summary:
{deal['summary']}

Details:
{deal['details']}

URL:
{deal['url']}
"""
            )


        prompt = f"""
Select up to 5 product deals.

Only select products that have:

- a clear actual selling price
- a useful product description
- a valid URL

Do NOT confuse "$100 off" or "save $100"
with an actual selling price.

Rewrite each description to describe the
product rather than promotional terms.

Deals:

{"---".join(text_blocks)}
"""


        response = (
            self.client.responses.parse(

                model=MODEL,

                input=[
                    {
                        "role":
                            "system",

                        "content":
                            (
                                "You identify "
                                "high-quality product "
                                "deal candidates."
                            ),
                    },
                    {
                        "role":
                            "user",

                        "content":
                            prompt,
                    },
                ],

                text_format=(
                    DealSelection
                ),
            )
        )


        selection = (
            response.output_parsed
        )


        valid = [

            deal.model_dump()

            for deal
            in selection.deals

            if deal.price > 0
        ]


        return {
            "deals":
                valid[
                    :MAX_SELECTED_DEALS
                ]
        }


# =============================================================
# LOCAL TEST
# =============================================================

@app.local_entrypoint()
def main():

    scanner = (
        ScannerAgent()
    )

    result = (
        scanner
        .scan
        .remote([])
    )

    for deal in result[
        "deals"
    ]:

        print(
            deal
        )


if __name__ == "__main__":
    pass