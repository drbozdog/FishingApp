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

def get_rivers(
    south_lat: float,
    west_lon: float,
    north_lat: float,
    east_lon: float,
) -> str:
    """
    Retrieve all OpenStreetMap rivers inside a bounding box.

    Parameters
    ----------
    south_lat : float
        minimum latitude  (south edge)
    west_lon  : float
        minimum longitude (west edge)
    north_lat : float
        maximum latitude  (north edge)
    east_lon  : float
        maximum longitude (east edge)

    Returns
    -------
    str
        Raw JSON string returned by the Overpass API.

    Notes
    -----
    * The bounding box must be **south,west,north,east** (Overpass requirement).
    * The query fetches *ways* **and** *relations* tagged ``waterway=river``.
    * Request geometries so we have the actual river linestrings.
    """

    query = f"""
    [out:json][timeout:25];
    (
      way["waterway"="river"]({south_lat},{west_lon},{north_lat},{east_lon});
      relation["waterway"="river"]({south_lat},{west_lon},{north_lat},{east_lon});
    );
    (._;>;);
    out body;
    """

    resp = requests.post(
        "https://overpass-api.de/api/interpreter",
        data=query.encode("utf-8"),
        headers={"Accept-Charset": "utf-8"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.text

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
    """Get content from a Wikipedia page by title or ID."""
    import html2text
    if (title and pageid) or (not title and pageid is None):
        raise ValueError("pass exactly one of title or pageid")

    params = {"action":"parse","format":"json","redirects":True,
              "prop":"text" if section != -1 else "sections"}
    params["page" if title else "pageid"] = title or pageid
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

TOOL_FUNCS = {
    "place_coordinates": place_coordinates,
    "wikipedia_search":  wikipedia_search,
    "wikipedia_explore": wikipedia_explore,
    "get_rivers": get_rivers,
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

def identify_river_by_bbox(
    south_lat: float,
    west_lon: float,
    north_lat: float,
    east_lon: float,
    river_hint: str,
    padding_km: float = 50.0,
) -> tuple[str, list]:
    """Identify the river inside a bounding box.

    ``river_hint`` is the expected river name. The bounding box is expanded by
    ``padding_km`` in all directions before querying. The assistant uses the
    ``get_rivers`` tool to list rivers in the area and returns the matching
    river name.
    """

    pad_deg = padding_km / 111.0
    south = south_lat - pad_deg
    west = west_lon - pad_deg
    north = north_lat + pad_deg
    east = east_lon + pad_deg

    question = (
        f"Find the river named '{river_hint}' inside the bounding box with "
        f"coordinates ({south}, {west}, {north}, {east}) given as south,west," 
        "north,east. Use the get_rivers tool to fetch the list of rivers and "
        "return the matching name exactly as found in that list."
    )

    return run(question)

def identify_river_by_points(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    river_hint: str,
    padding_km: float = 50.0,
) -> tuple[str, list]:
    """Identify the river using the bounding box defined by two points."""

    south = min(start_lat, end_lat)
    north = max(start_lat, end_lat)
    west = min(start_lon, end_lon)
    east = max(start_lon, end_lon)

    return identify_river_by_bbox(south, west, north, east, river_hint, padding_km)

if __name__ == "__main__":
    # Hard-coded example for testing river identification
    coordinates = {
        "start_point_latitude": 46.66222,
        "start_point_longitude": 22.51917,
        "end_point_latitude": 46.53576,
        "end_point_longitude": 22.45575,
    }
    river_hint = "Râul Crișul Băița"

    river_name, _ = identify_river_by_points(
        coordinates["start_point_latitude"],
        coordinates["start_point_longitude"],
        coordinates["end_point_latitude"],
        coordinates["end_point_longitude"],
        river_hint,
    )
    print("\nIdentified river:")
    print(river_name)
