from hashlib import sha256
import requests
import json
import unicodedata
from result_paths import KNOWLEDGE_DIR
from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException
import chardet
from charset_normalizer import from_bytes



UNREADABLE_THRESHOLD = 0.01


def unreadable_text(text):
    if not text:
        return False
    problematic = sum(
        char == "\ufffd" or (unicodedata.category(char) == "Cc" and char not in "\n\r\t")
        for char in text
    )
    return problematic / len(text) > UNREADABLE_THRESHOLD




def fetch_page_content(url, max_chars=50000):
    """Fetch and extract text 
    content from a URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'} #dictionary with user-agent to mimic a browser
        response = requests.get(url, headers=headers, timeout=5)
        if not 200 <= response.status_code < 300:
            return None
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type not in ("text/html", "application/xhtml+xml"):
            return None
        soup = BeautifulSoup(response.text, 'html.parser')

        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose() #cleaning by tag

        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())

        if not text.strip():
            return None
        if unreadable_text(text):
            return None

        return text[:max_chars] if len(text) > max_chars else text #Anwers format
    except Exception as e:
        pass
        return None


def fetch_raw_results(query,key, num_results=10, min_length=500, print_on=False):
    """
    Search with DuckDuckGo (posiblemente no sea DuckDuckGo sino uno general, usa DDGS) and optionally 
    fetch full content of results. 
    Returns valid documents as dictionaries containing document_id and text.
    """

    context = []
    sources = []

    backends = ["google", "bing", "duckduckgo", "brave", "yahoo"] 

    results_original = []

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

    results = list(unique_results)


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
            if raw_bytes is None:
                continue
            if isinstance(raw_bytes, str):  #If fetch_page_content returns a str, it converts in bytes.
                raw_bytes = raw_bytes.encode('utf-8', errors='replace')
            
            detected = chardet.detect(raw_bytes)
            encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
            content = raw_bytes.decode(encoding, errors="replace")  # replace invalid characters with a placeholder

            if unreadable_text(content):
                continue
            content_cleaned = " ".join(content.split())
            
            print("Done!") if print_on else None
            print(f"    Content: {content_cleaned[:500]}...\n") if print_on else None

            if len(content_cleaned.split()) > min_length:

                document_id = "doc_" + sha256(link.encode("utf-8")).hexdigest()
                context.append({"document_id": document_id, "text": content_cleaned})
                sources.append({"document_id": document_id, "title": title, "url": link})

            else: 
                too_short += 1

        except Exception as e:
            pass

    print(f"Retrieved: {len(results_original)} | Duplicates: {eliminated} | Invalid: {len(results) - len(context)} | Valid: {len(context)}")


    sources_path = KNOWLEDGE_DIR / "document_sources.json"
    source_report = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else {}
    source_report = {identifier: source for identifier, source in source_report.items()
                     if identifier.startswith("doc_") and isinstance(source, dict) and "url" in source}
    for source in sources:
        source_report[source["document_id"]] = source
    sources_path.write_text(json.dumps(source_report, ensure_ascii=False, indent=2), encoding="utf-8")

    return context

