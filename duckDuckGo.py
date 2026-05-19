import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
import chardet
from charset_normalizer import from_bytes



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


def fetch_raw_results(query, num_results=10, fetch_full_content=False, print_on=False):
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

            print(f"[{i}] Title: {title}") if print_on else None
            print(f"    Link:  {link}", flush=True) if print_on else None

            if fetch_full_content:
                print(f"    Fetching content...", end=" ", flush=True) if print_on else None
                # content = fetch_page_content(link)
                # content_cleaned = content.replace("\n", " ").strip()
                # print("Done!") if print_on else None
                # print(f"    Content: {content[:500]}...\n") if print_on else None

                raw_bytes = fetch_page_content(link)  # debe devolver bytes
                if isinstance(raw_bytes, str):  # si fetch_page_content devuelve str, convertir a bytes
                    raw_bytes = raw_bytes.encode('utf-8', errors='replace')
                
                detected = chardet.detect(raw_bytes)
                # detected = from_bytes(raw_bytes)[0].encoding
                encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
                content = raw_bytes.decode(encoding, errors="replace")  # reemplaza caracteres inválidos

                content_cleaned = content.replace("\n", " ").strip()
                
                print("Done!") if print_on else None
                print(f"    Content: {content_cleaned[:500]}...\n") if print_on else None

                context.append({
                    "position:":i,
                    "title": title,
                    "link": link,
                    "content": content_cleaned
                })
            else:
                print(f"    Snippet: {snippet}\n") if print_on else None
                context.append({
                    "title": title,
                    "link": link,
                    "content": snippet
                })

        except Exception as e:
            print(f"    Error processing result: {e}\n")

    print("=== Finished processing all results ===")
    return context

