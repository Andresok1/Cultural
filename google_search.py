from googleapiclient.discovery import build
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

### environment variables from .env file
API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

### extract clean text from webpage
def fetch_page_content(url, max_chars=5000):
    """Fetch and extract text content from a URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        
        # Get text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
        
        return text[:max_chars] + "..." if len(text) > max_chars else text
    except Exception as e:
        return f"Could not fetch content: {e}"

### search Google and optionally fetch full content of the results
def fetch_raw_results(query, fetch_full_content=False):
    # Initialize the search service
    service = build("customsearch", "v1", developerKey=API_KEY)
    
    # Execute the search (query)
    res = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=5).execute()

    
    # Debug: check how many results returned
    context = []
    items = res.get('items', [])
    print(f"--- Search Results for: {query} ---")
    print(f"--- Total results returned: {len(items)} ---\n")
    
    for i, item in enumerate(items, 1):
        try:
            print(f"[{i}] Title: {item['title']}")
            print(f"    Link:  {item['link']}", flush=True)
            
            if fetch_full_content:
                print(f"    Fetching content...", end=" ", flush=True)
                content = fetch_page_content(item['link'])
                print("Done!")
                # Truncate content for display
                print(f"Content: {content[:500]}...\n")
                context.append({
                    "title": item['title'],
                    "link": item['link'],
                    "content": content
                })
            else:
                print(f"Snippet: {item['snippet']}\n")
                context.append({
                    "title": item['title'],
                    "link": item['link'],
                    "content": item['snippet']
                })
        except Exception as e:
            print(f"    Error processing result: {e}\n")
    
    print("=== Finished processing all results ===")

    return context

if __name__ == "__main__":
    fetch_raw_results("Celebration of Festivals in Spanish culture", fetch_full_content=True)
