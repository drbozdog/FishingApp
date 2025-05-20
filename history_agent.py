import os, requests, json
from typing import Dict, Any, Tuple
from colorama import init, Fore, Style
from datetime import datetime
from agent import OpenAIManager

# Initialize colorama
init(autoreset=True)

# ---------- Wikipedia tools ----------
API_BASE = "https://{lang}.wikipedia.org/w/api.php"

def wikipedia_search(
    query: str,
    limit: int = 10,
    language: str = "en",
) -> Dict[str, Any]:
    """
    Search Wikipedia and return the top `limit` hits.

    Args:
        query:     Free-text search string.
        limit:     Max results (1-50). Default 10.
        language:  Wiki language code, default "en".

    Returns:
        {
          "results": [
             {"title": "World War II", "pageid": 12345, "snippet": "..."},
             ...
          ]
        }
    """
    if not (1 <= limit <= 50):
        raise ValueError("limit must be between 1 and 50")

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }
    resp = requests.get(API_BASE.format(lang=language), params=params, timeout=10)
    resp.raise_for_status()
    rows = resp.json()["query"]["search"]

    return {"results": [
        {"title": r["title"], "pageid": r["pageid"], "snippet": r["snippet"]}
        for r in rows
    ]}

def wikipedia_explore(
    title: str = None,
    pageid: int = None,
    section: int = None,
    language: str = "en",
) -> Dict[str, Any]:
    """
    Fetch full article text, a single section, or just the TOC.

    Args:
        title:    Page title (preferred) – set *either* title or pageid.
        pageid:   Numeric page ID if you stored it from search.
        section:  Section index (int). 0 or None → intro & infobox only.
                 -1 → return list of sections (table of contents).
        language: Wiki language code, default "en".

    Returns:
        {
          "title": "World War II",
          "pageid": 12345,
          "content": "World War II was a global conflict...",  # absent when section=-1
          "sections": [
              {"index": 1, "line": "Background"},
              {"index": 2, "line": "Course of the war"},
              ...
          ]                                         # only present when section=-1
        }
    """
    import html2text
        
    if not title and pageid is None:
        raise ValueError("Must provide title or pageid")
    if title and pageid:
        raise ValueError("Provide only one of title or pageid")

    params = {
        "action": "parse",
        "format": "json",
        "prop": "text" if section != -1 else "sections",
        "redirects": True,
    }
    if title:
        params["page"] = title
    else:
        params["pageid"] = pageid
    if section not in (None, 0, -1):
        params["section"] = section
    elif section in (0, None):
        params["prop"] = "text|sections"

    resp = requests.get(API_BASE.format(lang=language), params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()["parse"]

    result = {
        "title": data["title"],
        "pageid": data["pageid"],
    }
    if section == -1:
        result["sections"] = [
            {"index": int(s["index"]), "line": s["line"]} for s in data["sections"]
        ]
    else:
        # Convert HTML to markdown
        h2t = html2text.HTML2Text()
        h2t.ignore_links = False
        h2t.ignore_images = False
        h2t.ignore_tables = False
        h2t.body_width = 0  # No line wrapping
        
        # MediaWiki wraps HTML; convert to markdown
        html_content = data["text"]["*"]
        markdown_content = h2t.handle(html_content)
        
        result["content"] = markdown_content
    return result

def answer_history_question(question: str, verbose: bool = False) -> Tuple[str, list]:
    """
    Process a historical question using the Wikipedia tools.

    Args:
        question: The historical question to answer
        verbose: If True, prints detailed interaction logs and token usage

    Returns:
        A tuple containing:
        - final_output (str): The textual answer from the agent
        - raw_responses (list): List of raw responses for token tracking
    """
    # Initialize the OpenAI manager
    manager = OpenAIManager()
    
    # Define the system message for the history expert
    system_msg = (
        "You are a knowledgeable history expert that can answer questions about historical events, "
        "figures, and periods. Use Wikipedia to research and provide accurate, well-sourced answers. "
        "When answering questions:\n"
        "1. First search for relevant Wikipedia articles\n"
        "2. Explore the most relevant articles to gather detailed information\n"
        "3. Synthesize the information into a clear, comprehensive answer\n"
        "4. Always cite your sources by mentioning the Wikipedia articles you used\n"
        "5. If there are conflicting accounts or interpretations, mention them\n"
        "6. Focus on accuracy and factual information rather than speculation"
    )
    
    # Define the tool functions
    tool_funcs = {
        "wikipedia_search": wikipedia_search,
        "wikipedia_explore": wikipedia_explore
    }
    
    # Run the conversation
    return manager.run_conversation(question, system_msg, tool_funcs)

def main():
    """Example usage of the history agent."""
    example_question = "What were the main causes of World War II?"
    
    print(f"Running example query: {example_question}")
    final_output, raw_responses = answer_history_question(
        example_question, 
        verbose=True
    )
    
    print("\n" + "="*50)
    print("Example Main Execution Complete")
    print("Final output:")
    print(final_output)
    print("="*50 + "\n")

if __name__ == "__main__":
    print("To test the history agent, uncomment the line below:")
    print("main()") 