import os

import feedparser
import requests

from bs4 import BeautifulSoup

from openai import OpenAI

from pydantic import (
    BaseModel,
    Field,
)


# =============================================================
# CONFIGURATION
# =============================================================

MODEL = "gpt-5.1"


FEEDS = [

    "https://www.dealnews.com/c142/Electronics/?rss=1",

    "https://www.dealnews.com/c39/Computers/?rss=1",

    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]


MAX_ITEMS_PER_FEED = 10

MAX_SELECTED_DEALS = 5


# =============================================================
# STRUCTURED OUTPUT MODELS
# =============================================================

class Deal(BaseModel):

    product_description: str = Field(
        description=(
            "Clear description of the product itself. "
            "Focus on features and specifications rather "
            "than discount terms."
        )
    )

    price: float = Field(
        description=(
            "The actual current selling price of the product. "
            "Do not use the amount saved or discount amount."
        )
    )

    url: str = Field(
        description="URL of the deal."
    )


class DealSelection(BaseModel):

    deals: list[Deal]


# =============================================================
# RAW DEAL
# =============================================================

class ScrapedDeal:

    def __init__(
        self,
        title,
        summary,
        url,
        details="",
    ):

        self.title = title

        self.summary = summary

        self.url = url

        self.details = details


    def describe(self):

        return (
            f"Title: {self.title}\n"
            f"Summary: {self.summary}\n"
            f"Details: {self.details}\n"
            f"URL: {self.url}"
        )


# =============================================================
# CLEAN HTML
# =============================================================

def clean_html(text):

    if not text:
        return ""

    soup = BeautifulSoup(
        text,
        "html.parser",
    )

    return (
        soup
        .get_text(
            " ",
            strip=True,
        )
    )


# =============================================================
# OPTIONAL PAGE FETCH
#
# RSS already gives us useful information.
#
# We attempt to retrieve extra page details, but if DealNews
# blocks the request or its HTML changes, the scanner can still
# continue using the RSS data.
# =============================================================

def fetch_page_details(url):

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

        # Keep this deliberately generic because
        # website HTML structures can change.
        text = soup.get_text(
            " ",
            strip=True,
        )

        return text[:1500]

    except requests.RequestException:

        return ""


# =============================================================
# FETCH RSS DEALS
# =============================================================

def fetch_deals():

    deals = []

    for feed_url in FEEDS:

        print(
            "Reading feed:",
            feed_url
        )

        feed = feedparser.parse(
            feed_url
        )

        for entry in feed.entries[
            :MAX_ITEMS_PER_FEED
        ]:

            title = clean_html(
                entry.get(
                    "title",
                    "",
                )
            )

            summary = clean_html(
                entry.get(
                    "summary",
                    "",
                )
            )

            url = ""

            if entry.get("link"):

                url = entry["link"]

            elif entry.get("links"):

                url = (
                    entry["links"][0]
                    .get(
                        "href",
                        "",
                    )
                )

            if not url:
                continue

            details = (
                fetch_page_details(
                    url
                )
            )

            deal = ScrapedDeal(
                title=title,
                summary=summary,
                url=url,
                details=details,
            )

            deals.append(
                deal
            )

    return deals


# =============================================================
# SCANNER AGENT
# =============================================================

class ScannerAgent:

    def __init__(self):

        self.client = OpenAI()


    # ---------------------------------------------------------
    # Build one LLM prompt containing all candidate deals
    # ---------------------------------------------------------

    def build_prompt(
        self,
        scraped_deals,
    ):

        deal_text = "\n\n--- DEAL ---\n\n".join(

            deal.describe()

            for deal in scraped_deals
        )

        return f"""
You are selecting product deals for a price-analysis system.

Choose up to {MAX_SELECTED_DEALS} deals from the candidates.

Select deals where:

1. The actual selling price is clear.
2. The product itself has enough detail to estimate its fair value.
3. The URL is clearly available.

IMPORTANT:

If text says "$300 off", "$100 discount", or "save $200",
that number is NOT the product price.

Only use an actual current selling price.

Rewrite product_description so it describes the PRODUCT itself:
brand, model, specifications, size, capacity, features, etc.

Do not focus on coupons or promotional wording.

Candidate deals:

{deal_text}
"""


    # ---------------------------------------------------------
    # Scan
    #
    # seen_urls will later come from system memory.
    # ---------------------------------------------------------

    def scan(
        self,
        seen_urls=None,
    ):

        if seen_urls is None:

            seen_urls = set()

        else:

            seen_urls = set(
                seen_urls
            )

        print(
            "Fetching current deals..."
        )

        scraped_deals = (
            fetch_deals()
        )

        print(
            "Fetched deals:",
            len(scraped_deals)
        )

        # -----------------------------------------------------
        # Remove deals already handled previously.
        # -----------------------------------------------------

        new_deals = [

            deal

            for deal in scraped_deals

            if deal.url
            not in seen_urls
        ]

        print(
            "New deals:",
            len(new_deals)
        )

        if not new_deals:

            return DealSelection(
                deals=[]
            )

        prompt = (
            self.build_prompt(
                new_deals
            )
        )

        # -----------------------------------------------------
        # Structured Output
        #
        # GPT must return DealSelection rather than arbitrary
        # text that we then have to manually parse.
        # -----------------------------------------------------

        response = (
            self.client.responses.parse(

                model=MODEL,

                input=[
                    {
                        "role":
                            "system",

                        "content":
                            (
                                "You identify product deals "
                                "with clear prices and useful "
                                "product descriptions."
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

        # -----------------------------------------------------
        # Final defensive validation.
        # -----------------------------------------------------

        valid_deals = [

            deal

            for deal
            in selection.deals

            if deal.price > 0
            and deal.url
        ]

        return DealSelection(
            deals=(
                valid_deals[
                    :MAX_SELECTED_DEALS
                ]
            )
        )


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

    scanner = (
        ScannerAgent()
    )

    result = (
        scanner.scan()
    )

    print()

    print(
        "=" * 70
    )

    print(
        "SCANNER RESULTS"
    )

    print(
        "=" * 70
    )

    if not result.deals:

        print(
            "No suitable deals found."
        )

        return

    for (
        position,
        deal,
    ) in enumerate(
        result.deals,
        start=1,
    ):

        print()

        print(
            "Deal:",
            position
        )

        print(
            "Selling price:",
            deal.price
        )

        print(
            "Description:",
            deal.product_description
        )

        print(
            "URL:",
            deal.url
        )


if __name__ == "__main__":
    main()