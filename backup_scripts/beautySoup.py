import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs


def fetch_page_content(url, max_chars=5000):
    """Fetch and extract text content from a webpage."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unnecessary tags
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        text = " ".join(
            p.get_text(strip=True)
            for p in paragraphs
            if p.get_text(strip=True)
        )

        return text[:max_chars] if text else "No content found."

    except Exception as e:
        return f"Could not fetch content: {e}"


def clean_google_url(url):
    """Extract real URL from Google redirect links."""
    parsed = urlparse(url)

    if parsed.path == "/url":
        query_params = parse_qs(parsed.query)
        return query_params.get("q", [url])[0]

    return url


def fetch_raw_results(query, fetch_full_content=False, num_results=5):
    """Search Google by scraping results page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    search_url = "https://www.google.com/search"
    params = {
        "q": query,
        "num": num_results,
        "hl": "en"
    }

    response = requests.get(
        search_url,
        headers=headers,
        params=params,
        timeout=8
    )

    print(f"Status code: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    context = []

    results = soup.select("a[href]")

    filtered_results = []

    for result in results:
        href = result.get("href")

        if href and href.startswith("/url?q="):
            filtered_results.append(result)

    print(f"--- Search Results for: {query} ---")
    print(f"--- Total results found: {len(filtered_results)} ---\n")

    for i, result in enumerate(filtered_results[:num_results], 1):
        try:
            raw_link = result.get("href")
            link = clean_google_url(raw_link)

            title = result.get_text(strip=True)

            if not title:
                title = "No title"

            print(f"[{i}] Title: {title}")
            print(f"    Link: {link}")

            if fetch_full_content:
                print("    Fetching content...", end=" ")
                content = fetch_page_content(link)
                print("Done!")

                print(f"Content: {content[:500]}\n")

                context.append({
                    "title": title,
                    "link": link,
                    "content": content
                })

            else:
                context.append({
                    "title": title,
                    "link": link,
                    "content": ""
                })

        except Exception as e:
            print(f"Error processing result: {e}")

    print("=== Finished processing all results ===")
    print(response.url)

    return context


if __name__ == "__main__":
    fetch_raw_results(
        "Celebration of Festivals in Spanish culture",
        fetch_full_content=True
    )