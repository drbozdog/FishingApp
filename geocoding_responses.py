"""
River-segment assistant — synchronous version using AzureOpenAI
Requires: openai-python ≥ 1.30 (Responses API), colorama, geopy, tenacity, python-dotenv
"""

from __future__ import annotations
import os, json, requests, re
from typing import Any, Dict, List
from geopy.geocoders import Nominatim
from tenacity import retry, wait_fixed, stop_after_attempt
from agent import OpenAIManager
from openai import AzureOpenAI
from colorama import Fore

# ───────────────────  1.  TOOLS  ───────────────────
def place_coordinates(name: str, context: str | None = None) -> Dict[str, Any]:
    """Get GPS coordinates for a place using either OpenStreetMap or Wikipedia."""
    query = f"{name}, {context}" if context else name
    nominatim = Nominatim(user_agent="place-coord-tool/0.1")

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(5))
    def _lookup(q): return nominatim.geocode(q, exactly_one=True, timeout=10)

    if (loc := _lookup(query)):
        return {"lat": loc.latitude, "lon": loc.longitude, "source": "nominatim-osm"}

    resp = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{name}")
    if resp.ok and "coordinates" in resp.json():
        c = resp.json()["coordinates"]
        return {"lat": c["lat"], "lon": c["lon"], "source": "wikipedia"}

    raise ValueError(f"Could not geocode: {query}")

API_BASE = "https://{lang}.wikipedia.org/w/api.php"

def wikipedia_search(query: str, limit: int, language: str) -> Dict[str, Any]:
    """Search Wikipedia articles in a specific language."""
    # Ensure limit is an integer
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer")

    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")

    params = {"action":"query","list":"search","srsearch":query,"srlimit":limit,"format":"json"}
    rows = requests.get(API_BASE.format(lang=language),
                        params=params, timeout=10).json()["query"]["search"]
    return {"results":[{"title":r["title"],"pageid":r["pageid"],"snippet":r["snippet"]} for r in rows]}

def wikipedia_explore(title: str|None, pageid: int|None,
                      section: int, language: str) -> Dict[str, Any]:
    """
    Get content from a Wikipedia page by title or ID.
    
    Args:
        title (str|None): The title of the Wikipedia page. Must be None if pageid is provided.
            Example: "Mount Everest"
        pageid (int|None): The Wikipedia page ID. Must be None if title is provided.
            Example: 123456
        section (int): Section number to retrieve:
            -1: Return list of all sections
            0: Return full article content and section list
            >0: Return specific section content
        language (str): Wikipedia language code (e.g., "en", "ro", "fr", "de")
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - For section=-1: {"title": str, "pageid": int, "sections": List[Dict]}
            - For section>=0: {"title": str, "pageid": int, "content": str}
            
    Examples:
        # Get full article content and sections
        wikipedia_explore(title="Mount Everest", pageid=None, section=0, language="en")
        
        # Get section list
        wikipedia_explore(title="Mount Everest", pageid=None, section=-1, language="en")
        
        # Get specific section by ID
        wikipedia_explore(title="Mount Everest", pageid=None, section=1, language="en")
        
        # Using page ID instead of title
        wikipedia_explore(title=None, pageid=123456, section=0, language="en")
    
    Raises:
        ValueError: If both title and pageid are provided or if neither is provided
    """
    import html2text
    # Priority for pageid if both are provided
    if pageid is not None:
        use_pageid = True
    elif title:
        use_pageid = False
    else:
        raise ValueError("must provide either title or pageid")

    params = {"action":"parse","format":"json","redirects":True,
              "prop":"text" if section != -1 else "sections"}
    params["pageid" if use_pageid else "page"] = pageid if use_pageid else title
    if section not in (None,0,-1):
        params["section"] = section
    elif section in (0,None):
        params["prop"] = "text|sections"

    data = requests.get(API_BASE.format(lang=language),
                        params=params, timeout=10).json()["parse"]
    out = {"title": data["title"], "pageid": data["pageid"]}
    if section == -1:
        out["sections"] = [{"index":int(s["index"]), "line":s["line"]} for s in data["sections"]]
    else:
        h2t = html2text.HTML2Text(); h2t.body_width = 0
        out["content"] = h2t.handle(data["text"]["*"])
    return out

def extract_coordinates(response_text: str) -> Dict[str, float]:
    """
    Extract coordinates from the response text.
    
    Args:
        response_text (str): The response text containing coordinates
        
    Returns:
        Dict[str, float]: Dictionary containing start and end point coordinates
        
    Example response format:
    1. Limita superioară – confluenţa Pârâul Săgagea cu Râul Poșaga
       Latitudine: 46.45000 N  
       Longitudine: 23.33889 E
       
    2. Limita inferioară – confluenţa Râul Poșaga cu Râul Arieșul Mare
       Latitudine: 46.43083 N  
       Longitudine: 23.44028 E
    """
    # Initialize coordinates dictionary
    coordinates = {
        "start_point_latitude": None,
        "start_point_longitude": None,
        "end_point_latitude": None,
        "end_point_longitude": None
    }
    
    # Regular expressions to match coordinates
    lat_pattern = r"Latitudine:\s*(\d+\.\d+)\s*N"
    lon_pattern = r"Longitudine:\s*(\d+\.\d+)\s*E"
    
    # Split text into upper and lower limit sections
    sections = response_text.split("2. Limita")
    
    if len(sections) >= 1:
        # Extract coordinates for start point (upper limit)
        if lat_match := re.search(lat_pattern, sections[0]):
            coordinates["start_point_latitude"] = float(lat_match.group(1))
        if lon_match := re.search(lon_pattern, sections[0]):
            coordinates["start_point_longitude"] = float(lon_match.group(1))
            
    if len(sections) >= 2:
        # Extract coordinates for end point (lower limit)
        if lat_match := re.search(lat_pattern, sections[1]):
            coordinates["end_point_latitude"] = float(lat_match.group(1))
        if lon_match := re.search(lon_pattern, sections[1]):
            coordinates["end_point_longitude"] = float(lon_match.group(1))
    
    return coordinates

# requirements.txt
# duckduckgo-search>=8.0.0, requests>=2.32

from typing import List, Dict
import os, requests, itertools
from duckduckgo_search import DDGS


def search_web(query: str) -> List[Dict[str, str]]:
    """
    Search the web and return a list of results with title, link, and snippet.
    The function automatically chooses an engine and, if that engine is
    unavailable (missing API key, quota exhausted, network error, etc.),
    retries with the next one in the priority list until it succeeds.

    Parameters
    ----------
    query : str
        Search terms (e.g. "Turing Award 2024").

    Returns
    -------
    list[dict]
        Each dict has: title, link, snippet.

    Raises
    ------
    RuntimeError
        If every engine fails.
    """

    # ---------- helpers for each engine ---------- #
    def _duck(query, n):
        print(f"{Fore.BLUE}Trying DuckDuckGo search engine...{Fore.RESET}")
        with DDGS() as ddgs:
            results = [
                {"title": r["title"], "link": r["href"], "snippet": r["body"]}
                for r in ddgs.text(query, max_results=n)
            ]
            print(f"{Fore.GREEN}✓ DuckDuckGo search successful{Fore.RESET}")
            return results

    def _brave(query, n):
        print(f"{Fore.BLUE}Trying Brave search engine...{Fore.RESET}")
        key = os.getenv("BRAVE_API_KEY")
        if not key:
            print(f"{Fore.RED}✗ Brave search failed: API key not set{Fore.RESET}")
            raise RuntimeError("BRAVE_API_KEY not set")
        url = "https://api.search.brave.com/res/v1/web/search"
        hdr = {"Accept": "application/json", "X-Subscription-Token": key}
        res = requests.get(url, params={"q": query, "source": "web", "count": n}, headers=hdr, timeout=6)
        res.raise_for_status()
        results = [
            {"title": h["title"], "link": h["url"], "snippet": h["snippet"]}
            for h in res.json()["web"]["results"]
        ]
        print(f"{Fore.GREEN}✓ Brave search successful{Fore.RESET}")
        return results

    def _google(query, n):
        print(f"{Fore.BLUE}Trying Google Custom Search engine...{Fore.RESET}")
        key, cx = os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_CSE_ID")
        if not (key and cx):
            print(f"{Fore.RED}✗ Google search failed: API keys not set{Fore.RESET}")
            raise RuntimeError("Google keys not set")
        res = requests.get(
            "https://customsearch.googleapis.com/customsearch/v1",
            params={"q": query, "key": key, "cx": cx, "num": n},
            timeout=6,
        )
        res.raise_for_status()
        results = [
            {"title": i["title"], "link": i["link"], "snippet": i["snippet"]}
            for i in res.json().get("items", [])
        ]
        print(f"{Fore.GREEN}✓ Google search successful{Fore.RESET}")
        return results

    def _serper(query, n):
        print(f"{Fore.BLUE}Trying Serper search engine...{Fore.RESET}")
        key = os.getenv("SERPER_API_KEY")
        if not key:
            print(f"{Fore.RED}✗ Serper search failed: API key not set{Fore.RESET}")
            raise RuntimeError("SERPER_API_KEY not set")
        res = requests.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": n},
            headers={"X-API-KEY": key},
            timeout=6,
        )
        res.raise_for_status()
        results = [
            {"title": h["title"], "link": h["link"], "snippet": h["snippet"]}
            for h in res.json()["organic"]
        ]
        print(f"{Fore.GREEN}✓ Serper search successful{Fore.RESET}")
        return results

    def _tavily(query, n):
        print(f"{Fore.BLUE}Trying Tavily search engine...{Fore.RESET}")
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            print(f"{Fore.RED}✗ Tavily search failed: API key not set{Fore.RESET}")
            raise RuntimeError("TAVILY_API_KEY not set")
        url = f"https://api.tavily.com/search?q={query}&n={n}"
        res = requests.get(url, headers={"Authorization": key}, timeout=6)
        res.raise_for_status()
        results = res.json()["results"]
        print(f"{Fore.GREEN}✓ Tavily search successful{Fore.RESET}")
        return results

    # ---------- iterate until one engine works ---------- #
    engines = (_duck, _serper, _tavily)
    errors = []
    num_results = 12

    print(f"\n{Fore.CYAN}Searching for: {query}{Fore.RESET}")
    for engine_fn in engines:
        try:
            results = engine_fn(query, num_results)
            if results:                       # success!
                return results[:num_results]
        except Exception as exc:
            errors.append(f"{engine_fn.__name__}: {str(exc)}")
            print(f"{Fore.RED}✗ {engine_fn.__name__} failed: {str(exc)}{Fore.RESET}")
            continue

    print(f"{Fore.RED}✗ All search engines failed{Fore.RESET}")
    raise RuntimeError(
        f"All search engines failed for query '{query}'. "
        f"Errors encountered: {errors}"
    )

def search_google_maps_text(query: str) -> dict:
    """
    Google Places Text Search (POST https://places.googleapis.com/v1/places:searchText)
    – returns up to 20 results.
    """
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": os.environ["GOOGLE_MAPS_API_KEY"],
        "X-Goog-FieldMask": "*"        # all fields; trim in production if desired
    }
    body = {
        "textQuery": query,
        "maxResultCount": 20           # hard-coded as requested
    }
    resp = requests.post(url, json=body, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()

TOOL_FUNCS = {
    "place_coordinates": place_coordinates,
    "wikipedia_search":  wikipedia_search,
    "wikipedia_explore": wikipedia_explore,
    "search_web": search_web,
    "search_google_maps_text": search_google_maps_text,
}

# ───────────────────  2.  SYSTEM MESSAGE  ───────────────────
SYSTEM_MSG = ("You are a helpful Romanian geography assistant. "
              "Identify places, use the tools for coordinates, "
              "use the place_coordinates first and then the other tools"
              "and provide precise GPS locations (river limits ±5 km).")

# ───────────────────  3.  MAIN EXECUTION  ───────────────────
def run(question: str):
    manager = OpenAIManager()
    answer, logs = manager.run_conversation(question, SYSTEM_MSG, TOOL_FUNCS)
    return answer, logs

def extract_coordinates_structured(
    prior_messages: List[Dict[str, str]],
) -> Dict[str, float]:
    """
    Extract coordinates from the response text using OpenAI structured output.
    
    Args:
        prior_messages: The exact message list from the conversation, in order.
        
    Returns:
        Dict containing start and end point coordinates
        
    Example response format:
    {
        "start_point_latitude": 46.45000,
        "start_point_longitude": 23.33889,
        "end_point_latitude": 46.43083,
        "end_point_longitude": 23.44028
    }
    """
    # Append the new "please convert to JSON" request
    messages = prior_messages + [
        {
            "role": "user",
            "content": (
                "Convert the coordinates from the answer you just gave into valid JSON with these exact keys: "
                "start_point_latitude (float), start_point_longitude (float), "
                "end_point_latitude (float), end_point_longitude (float). "
                "Extract only the numerical values, removing the N and E indicators."
            ),
        }
    ]

    # Use the OpenAIManager to get structured output
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2025-04-01-preview",
    )
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )

    # Parse the assistant's reply
    content = completion.choices[0].message.content
    if isinstance(content, dict):
        return content
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        print("Structured output unavailable or malformed")
        raise ValueError("Assistant did not return valid JSON") from exc

def get_river_segment_coordinates(
    county: str,
    river_name: str,
    segment: str,
    length: str,
    country: str
) -> Dict[str, float]:
    """
    Get coordinates for a river segment using just the essential parameters.
    
    Args:
        county: The county where the river is located
        river_name: The name of the river
        segment: Description of the segment (e.g. "start point - end point")
        length: Length of the segment (e.g. "10 km")
        country: Country where the river is located
        
    Returns:
        Dict containing start and end point coordinates:
        {
            "start_point_latitude": float,
            "start_point_longitude": float,
            "end_point_latitude": float,
            "end_point_longitude": float
        }
    """
    # Format the query
    query = (f"Give me the GPS coordinates for the limits for the following river segment:\n"
            f"county: {county} | river: {river_name} | segment: {segment} | "
            f"length: {length} | country: {country}")
    
    # Get the response using the existing run function
    answer, _ = run(query)
    
    # Create conversation for structured output
    conversation = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    
    # Extract coordinates
    return extract_coordinates_structured(conversation)

if __name__ == "__main__":
    # Example usage of the new function
    coordinates = get_river_segment_coordinates(
        county="ALBA",
        river_name="Râul Poșaga",
        segment="pârâul Săgagea – confluence with Râul Arieșul Mare",
        length="10 km",
        country="Romania"
    )
    print("\nExtracted Coordinates:")
    print(json.dumps(coordinates, indent=2))
