from openai import OpenAI
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
from dotenv import load_dotenv

load_dotenv()

endpoint = "PUT_YOUR_OPENAI_ENDPOINT"
deployment_name = "gpt-5.1"

LINK_INSTRUCTIONS = """
        You are selecting pages that would be useful for creating
        a company brochure.

        Select pages such as:
        - About
        - Company
        - Products
        - Services
        - Careers

        Ignore pages such as:
        - Privacy
        - Terms
        - Cookies
        - Login

        Return only the selected URLs.
        Return one URL per line.
        """

BROCHURE_INSTRUCTIONS = """
        Create a concise company brochure using only the
        information provided.

        Include:
        - Company overview
        - Products or services
        - Company culture
        - Careers

        Do not invent information that is not present
        in the supplied website content.

        Return the brochure in Markdown.
        """

def main():
    client = OpenAI(
        base_url=endpoint
    )

    website_url = "https://target.com"

    # -------------------------------------------------
    # Step 1: Read the homepage
    # -------------------------------------------------

    response = requests.get(website_url, verify=False)
    soup = BeautifulSoup(response.text, "html.parser")
    homepage_text = soup.get_text(" ", strip=True)

    links = []

    for link in soup.find_all("a"):
        href = link.get("href")

        if href:
            links.append(urljoin(website_url, href))

    print("\nLinks found:")
    for link in links:
        print(link)


    # -------------------------------------------------
    # Step 2: Ask LLM which links are useful
    # -------------------------------------------------

    link_response = client.responses.create(
                    model = deployment_name,
                    instructions = LINK_INSTRUCTIONS,
                    input="\n".join(links)
                )

    selected_links = link_response.output_text.splitlines()

    print("\nSelected links:")
    for link in selected_links:
        print(link)

    # -------------------------------------------------
    # Step 3: Read selected pages
    # -------------------------------------------------

    company_content = f"""
    Homepage:

    {homepage_text}
    """

    for link in selected_links:

        response = requests.get(link, verify=False)

        soup = BeautifulSoup(response.text, "html.parser")

        page_text = soup.get_text(" ", strip=True)

        company_content += f"""

        Page: {link}

        {page_text}
        """

    print(company_content)


    # -------------------------------------------------
    # Step 4: Generate the brochure
    # -------------------------------------------------

    brochure_response = client.responses.create(
                        model = deployment_name,
                        instructions = BROCHURE_INSTRUCTIONS,
                        input=company_content
                    )

    print("\n--- COMPANY BROCHURE ---\n")
    print(brochure_response.output_text)


if __name__ == "__main__":
    main()