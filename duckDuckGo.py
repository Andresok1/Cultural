import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException
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


def fetch_raw_results(query,key, num_results=10, print_on=False):
    """
    Search with DuckDuckGo (posiblemente no sea DuckDuckGo sino uno general, usa DDGS) and optionally 
    fetch full content of results. 
    Format: [{{"title": "...", "link": "...", "content": "..."}}, ...]
    """

    context = []

    print(f"--- Search Results for: {query} ---")

    backends = ["google", "bing", "duckduckgo", "brave", "yahoo"] 

    results_original = []

    print(f"Results using:")

    for backend in backends:
        try:
            with DDGS() as ddgs:    #Searcher context manager, using query and #Results
                raw_results  = list(ddgs.text(
                    query, 
                    backend= backend,
                    max_results = num_results
                )) 
            if raw_results is not None:
                print(f"    {backend} backend:{len(raw_results)}")

            results_original.extend(raw_results)

            for r in raw_results:
                r["backend"] = backend

        except DDGSException as e:
            print(f"    {backend} backend: 0")

    unique_results = {item['href']: item for item in results_original}.values()

    eliminated = len(results_original) - len(unique_results)
    print(f"    Repeated results eliminated: {eliminated}")
    results = list(unique_results)
    
    
    print(f"--- Total results retrieved: {len(results)} ---")


    too_short  = 0

    for i, item in enumerate(results, 1):   #Extract info from url, clean info and ealuate for "too short"
        try:
            title = item.get('title', 'Error: No title found')
            link = item.get('href', 'Error: No link found') #get the URL of result  
            backend = item.get('backend', 'Error: No backend found')

            print(f"[{i}] Title: {title}") if print_on else None
            print(f"    Link:  {link}", flush=True) if print_on else None

            print(f"    Fetching content...", end=" ", flush=True) if print_on else None
 

            raw_bytes = fetch_page_content(link)  # It reads url and returns content in bytes.
            if isinstance(raw_bytes, str):  #If fetch_page_content returns a str, it converts in bytes.
                raw_bytes = raw_bytes.encode('utf-8', errors='replace')
            
            detected = chardet.detect(raw_bytes)
            encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
            content = raw_bytes.decode(encoding, errors="replace")  # replace invalid characters with a placeholder

            content_cleaned = content.replace("\n", " ").strip()
            
            print("Done!") if print_on else None
            print(f"    Content: {content_cleaned[:500]}...\n") if print_on else None

            if len(content_cleaned.split()) > 300:

                context.append(content_cleaned)

            else: 
                too_short += 1

        except Exception as e:
            print(f"    Error processing result: {e}\n")

    print("Erased documents with few content:", too_short)
    print(f"Final ranking size for '{query}', which also belongs to '{key}':", len(context))
    print("=== Finished processing all results ===")
    print("\n")


    return context

