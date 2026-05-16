import requests
from bs4 import BeautifulSoup
from ddgs import DDGS



def fetch_page_content(url, max_chars=50000):
    """Fetch and extract text 
    content from a URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'} #dictionary with user-agent to mimic a browser
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose() #cleaning by tag

        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())

        return text[:max_chars] if len(text) > max_chars else text #Anwers format
    except Exception as e:
        return f"Could not fetch content: {e}"


def fetch_raw_results(query, num_results=10, fetch_full_content=False):
    """
    Search with DuckDuckGo and optionally 
    fetch full content of results. 
    Format: [{{"title": "...", "link": "...", "content": "..."}}, ...]
    """

    context = []

    print(f"--- Search Results for: {query} ---")

    with DDGS() as ddgs:    #Searcher context manager, using query and #Results
        results = list(ddgs.text(query, max_results=num_results)) 

    print(f"--- Total results returned: {len(results)} ---\n")

    for i, item in enumerate(results, 1):
        try:
            title = item.get('title', 'No title found')
            link = item.get('href', '') #get the URL of result  
            snippet = item.get('body', '')

            print(f"[{i}] Title: {title}")
            print(f"    Link:  {link}", flush=True)

            if fetch_full_content:
                print(f"    Fetching content...", end=" ", flush=True)
                content = fetch_page_content(link)
                print("Done!")
                print(f"    Content: {content[:500]}...\n")
                context.append({
                    "position:":i,
                    "title": title,
                    "link": link,
                    "content": content
                })
            else:
                print(f"    Snippet: {snippet}\n")
                context.append({
                    "title": title,
                    "link": link,
                    "content": snippet
                })

        except Exception as e:
            print(f"    Error processing result: {e}\n")

    print("=== Finished processing all results ===")
    return context

