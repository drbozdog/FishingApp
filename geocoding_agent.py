import os, requests, json, pprint
from typing import Dict, Any
from geopy.geocoders import Nominatim
from tenacity import retry, wait_fixed, stop_after_attempt
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from agents import (
    Agent, Runner, function_tool, ItemHelpers,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)
# Import colorama for colored console output
from colorama import init, Fore, Style
# Import the specific item types for type checking
from agents.items import ToolCallItem, ToolCallOutputItem, MessageOutputItem, ReasoningItem

# Initialize colorama
init(autoreset=True)

# Load environment variables from .env file
load_dotenv()

# 1️⃣  wire up Azure OpenAI
aoai = AsyncAzureOpenAI(
    api_key        = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    # azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT"),    # deployment name
)

set_default_openai_client(aoai, use_for_tracing=False)
set_tracing_disabled(True)                    # avoids 401s against api.openai.com

# ---------- Place coordinates tool ----------
@function_tool
def place_coordinates(
    name: str,
    context: str | None = None
) -> Dict[str, Any]:
    """
    Return WGS-84 coordinates for a single place name.

    Args:
        name:     The locality / dam / confluence label (e.g. "Stâna de Mureș")
        context:  Optional extra text that helps disambiguate
                  (e.g. "Mureș river, Alba county, Romania")
    """
    query = f"{name}, {context}" if context else name

    nominatim = Nominatim(user_agent="place-coord-tool/0.1")

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(5))
    def _lookup(q): return nominatim.geocode(q, exactly_one=True, timeout=10)

    loc = _lookup(query)
    if not loc:                              # fallback to Wikipedia
        resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{name}"
        )
        if resp.ok and "coordinates" in resp.json():
            c = resp.json()["coordinates"]
            return {"lat": c["lat"], "lon": c["lon"], "source": "wikipedia"}
        raise ValueError(f"Could not geocode: {query}")

    return {
        "lat":    loc.latitude,
        "lon":    loc.longitude,
        "source": "nominatim-osm"
    }

# ---------- Wikipedia tools ----------
API_BASE = "https://{lang}.wikipedia.org/w/api.php"

@function_tool
def wikipedia_search(
    query: str,
    limit: int  ,
    language: str,
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
             {"title": "Mureș (river)", "pageid": 202435, "snippet": "…"},
             …
          ]
        }
    """
    if not (1 <= limit <= 50):
        raise ValueError("limit must be between 1 and 50")

    params = {
        "action": "query",
        "list":   "search",
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

@function_tool
def wikipedia_explore(
    title: str,
    pageid: int,
    section: int,
    language: str,
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
          "title": "Mureș (river)",
          "pageid": 202435,
          "content": "The Mureș is a river in…",   # absent when section=-1
          "sections": [
              {"index": 1, "line": "Course"},
              {"index": 2, "line": "Tributaries"},
              …
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
        "title":  data["title"],
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

# Function to safely get a value from a raw_item, which could be a dict or an object
def safe_get(item, key):
    if isinstance(item, dict):
        return item.get(key)
    else:
        return getattr(item, key, None)

# Function to display agent interaction items with color
def display_item(item):
    if isinstance(item, ToolCallItem):
        # Safely access the name and arguments from raw_item (which could be a dict or object)
        raw_item = item.raw_item
        tool_name = safe_get(raw_item, "name")
        
        # Handle different argument formats
        if isinstance(raw_item, dict) and "arguments" in raw_item:
            tool_args = raw_item["arguments"]
        elif hasattr(raw_item, "arguments"):
            tool_args = raw_item.arguments
        else:
            # If arguments aren't available in the expected format, try to parse from a string
            # This handles cases where arguments might be a JSON string
            args_str = str(raw_item)
            try:
                tool_args = json.loads(args_str)
            except:
                tool_args = {"raw_arguments": args_str}
        
        print(f"{Fore.YELLOW}TOOL CALL ➜{Style.RESET_ALL} {tool_name}")
        # Pretty print the arguments
        if isinstance(tool_args, str):
            try:
                # Try to parse string as JSON for better display
                parsed_args = json.loads(tool_args)
                print(json.dumps(parsed_args, indent=2))
            except:
                print(tool_args)
        else:
            print(json.dumps(tool_args, indent=2))
        
    elif isinstance(item, ToolCallOutputItem):
        # Access the name from raw_item and the output directly
        raw_item = item.raw_item
        tool_name = safe_get(raw_item, "name")
        
        print(f"{Fore.CYAN}TOOL RESULT ➜{Style.RESET_ALL} {tool_name}")
        formatted_json = json.dumps(item.output, indent=2)
        # Colorize certain parts of the JSON output
        formatted_json = (formatted_json
            .replace('"lat":', f'{Fore.GREEN}"lat":{Style.RESET_ALL}')
            .replace('"lon":', f'{Fore.GREEN}"lon":{Style.RESET_ALL}')
            .replace('"title":', f'{Fore.YELLOW}"title":{Style.RESET_ALL}')
            .replace('"pageid":', f'{Fore.YELLOW}"pageid":{Style.RESET_ALL}')
            .replace('"source":', f'{Fore.BLUE}"source":{Style.RESET_ALL}')
        )
        print(formatted_json)
        
    elif isinstance(item, MessageOutputItem):
        print(f"{Fore.MAGENTA}ASSISTANT SAID ➜{Style.RESET_ALL}")
        message = ItemHelpers.text_message_output(item)
        print(message)
        
    elif isinstance(item, ReasoningItem):
        print(f"{Fore.GREEN}REASONING ➜{Style.RESET_ALL}")
        content = safe_get(item.raw_item, "content")
        print(content)
        
    else:
        print(f"{Fore.WHITE}OTHER ITEM ➜{Style.RESET_ALL}", item.type)
        # Print the raw item for debugging
        try:
            print(f"Raw item: {item.raw_item}")
        except:
            print("Could not access raw_item")

# Function to calculate and display token usage from raw_responses
def display_token_usage(raw_responses):
    total_prompt_tokens = 0
    total_completion_tokens = 0
    valid_responses = 0
    incomplete_responses = 0
    
    print(f"\n{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}TOKEN USAGE SUMMARY{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*80}{Style.RESET_ALL}\n")
    
    for i, response in enumerate(raw_responses):
        # Check if response has usage data
        has_usage = hasattr(response, 'usage') and response.usage is not None
        
        if not has_usage:
            incomplete_responses += 1
            print(f"Call {i+1}: {Fore.RED}No usage data available{Style.RESET_ALL}")
            print()
            continue
        
        usage_str = str(response.usage)
        print(f"Call {i+1}:")
        
        # Try to extract tokens from attributes if available
        if hasattr(response.usage, 'input_tokens') and hasattr(response.usage, 'output_tokens'):
            # Azure OpenAI format
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            print(f"  {Fore.CYAN}Input tokens:{Style.RESET_ALL} {input_tokens}")
            print(f"  {Fore.GREEN}Output tokens:{Style.RESET_ALL} {output_tokens}")
            total_prompt_tokens += input_tokens
            total_completion_tokens += output_tokens
            valid_responses += 1
            
        elif hasattr(response.usage, 'prompt_tokens') and hasattr(response.usage, 'completion_tokens'):
            # Standard OpenAI format
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            print(f"  {Fore.CYAN}Prompt tokens:{Style.RESET_ALL} {prompt_tokens}")
            print(f"  {Fore.GREEN}Completion tokens:{Style.RESET_ALL} {completion_tokens}")
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            valid_responses += 1
            
        elif 'input_tokens=' in usage_str or 'prompt_tokens=' in usage_str:
            # Parse string representation
            import re
            input_match = re.search(r'input_tokens=(\d+)', usage_str)
            if not input_match:
                input_match = re.search(r'prompt_tokens=(\d+)', usage_str)
            
            output_match = re.search(r'output_tokens=(\d+)', usage_str)
            if not output_match:
                output_match = re.search(r'completion_tokens=(\d+)', usage_str)
            
            input_tokens = int(input_match.group(1)) if input_match else 0
            output_tokens = int(output_match.group(1)) if output_match else 0
            
            if input_tokens > 0 or output_tokens > 0:
                print(f"  {Fore.CYAN}Input/Prompt tokens:{Style.RESET_ALL} {input_tokens}")
                print(f"  {Fore.GREEN}Output/Completion tokens:{Style.RESET_ALL} {output_tokens}")
                total_prompt_tokens += input_tokens
                total_completion_tokens += output_tokens
                valid_responses += 1
            else:
                incomplete_responses += 1
                print(f"  {Fore.RED}Could not extract token counts{Style.RESET_ALL}")
        else:
            incomplete_responses += 1
            print(f"  {Fore.RED}Unrecognized usage format{Style.RESET_ALL}")
        
        # Always print the raw usage string for reference
        print(f"  {Fore.YELLOW}Usage:{Style.RESET_ALL} {usage_str}")
        
        # If model information is available, show that too
        if hasattr(response, 'model'):
            print(f"  {Fore.MAGENTA}Model:{Style.RESET_ALL} {response.model}")
        
        print()
    
    # Calculate correct total
    total_tokens = total_prompt_tokens + total_completion_tokens
    
    # Display grand totals
    print(f"{Fore.YELLOW}{'='*40}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}GRAND TOTALS:{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}Total Input/Prompt Tokens:{Style.RESET_ALL} {total_prompt_tokens}")
    print(f"  {Fore.GREEN}Total Output/Completion Tokens:{Style.RESET_ALL} {total_completion_tokens}")
    print(f"  {Fore.BLUE}Total Tokens Used:{Style.RESET_ALL} {total_tokens}")
    print(f"  Responses with valid usage: {valid_responses}")
    print(f"  Responses with incomplete/missing usage: {incomplete_responses}")
    
    # Estimate cost (very rough approximation)
    # These rates are ballpark figures and will vary by model and region
    # GPT-4 rates as of May 2024
    PRICE_PROMPT = 0.01 / 1000  # $0.01 per 1000 tokens for input
    PRICE_COMP = 0.03 / 1000    # $0.03 per 1000 tokens for output
    
    # For GPT-4-mini/nano, adjust rates
    if any(hasattr(r, 'model') and (getattr(r, 'model', '').endswith('-mini') or getattr(r, 'model', '').endswith('-nano')) for r in raw_responses):
        PRICE_PROMPT = 0.003 / 1000  # $0.003 per 1000 tokens for input
        PRICE_COMP = 0.015 / 1000    # $0.015 per 1000 tokens for output
    
    est_cost = (total_prompt_tokens * PRICE_PROMPT) + (total_completion_tokens * PRICE_COMP)
    
    print(f"  {Fore.RED}Estimated Cost (USD):{Style.RESET_ALL} ${est_cost:.6f}")

# 2️⃣  build an agent
planner = Agent(
    name="Geo-planner",
    instructions=(
        "You are a helpful assistant that can answer questions about the geography of Romania. "
        "Find and extract the river limits, first try to identify the key places mentioned, "
        "then get their coordinates, and finally provide a clear answer with the location data."
        "If the place_coordinates doesn't find the location, try different queries, "
        "Wikipedia has a lot of information about Romania and its rivers, "
        "Approximation of the river limits is allowed, as long as the coordinates are within 5km to the actual limits."
    ),
    tools=[place_coordinates, wikipedia_search, wikipedia_explore],
    model='o4-mini'
)

async def geocode_row_data(row_data: Dict[str, str], agent: Agent, verbose: bool = False) -> tuple[str, list, list]:
    """
    Processes a single row of data to get GPS coordinates for a river segment.

    Args:
        row_data: A dictionary representing a row from the CSV file.
                  Expected keys: 'JUDET', 'DENUMIRE_RAU', 'SECTOR_LIMIT', 'LUNGIME_KM'.
                                 'Country' is assumed to be 'Romania'.
        agent: The Agent instance to use for processing.
        verbose: If True, prints detailed interaction logs and token usage.

    Returns:
        A tuple containing:
        - final_output (str): The textual result from the agent.
        - new_items (list): List of interaction items for detailed logging.
        - raw_responses (list): List of raw responses for token tracking.
    """
    # Construct the query from row_data
    # TODO: Confirm these column names with the user if different
    county = row_data.get('JUDET', 'N/A')
    river = row_data.get('DENUMIRE_RAU', 'N/A')
    segment = row_data.get('SECTOR_LIMIT', 'N/A')
    length = row_data.get('LUNGIME_KM', 'N/A')
    country = row_data.get('COUNTRY', 'Romania') # Assuming 'Romania' if not specified

    q = f"""Give me the GPS coordinates for the limits for the following river segment:
    county:{county} river:{river} segment:{segment} length:{length} Km country:{country}"""

    if verbose:
        print(f"Processing query: {q}")

    result = await Runner.run(agent, q, max_turns=20)

    if verbose:
        # Print agent interaction with color
        print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}AGENT INTERACTION LOG (for query: {q}){Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}\n")

        for item in result.new_items:
            try:
                display_item(item)
                print()
            except Exception as e:
                print(f"{Fore.RED}Error displaying item: {str(e)}{Style.RESET_ALL}")
                print(f"Item type: {type(item)}")
                print(f"Item dict: {item.__dict__}")
                print()

        # Print final result for this specific call
        print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}FINAL ANSWER (for query: {q}){Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}\n")
        print(Fore.CYAN + result.final_output + Style.RESET_ALL)

        # Display token usage for this specific call
        display_token_usage(result.raw_responses)

    return result.final_output, result.new_items, result.raw_responses

# 3️⃣  run it (Example usage, will be replaced by ETL script)
async def main():
    # This main function is now an example of how to use geocode_row_data
    # The actual ETL process will be in a separate script.
    example_row = {
        'JUDET': 'ALBA',
        'DENUMIRE_RAU': 'Râul Poșaga',
        'SECTOR_LIMIT': 'Conf. pârâul Săgagea – conf. Râul Arieșul Mare',
        'LUNGIME_KM': '10',
        'COUNTRY': 'Romania'
    }
    
    print(f"Running example geocoding for row: {example_row}")
    final_output, _, raw_responses = await geocode_row_data(example_row, planner, verbose=True)
    
    print("\n" + "="*50)
    print("Example Main Execution Complete.")
    print("Final output received by main:")
    print(final_output)
    print("="*50 + "\n")
    
    # Display cumulative token usage if you were to run multiple calls here
    # For a single call, display_token_usage inside geocode_row_data (if verbose) handles it.
    # If you were to loop and collect raw_responses:
    # all_raw_responses = []
    # all_raw_responses.extend(raw_responses) # from the call above
    # display_token_usage(all_raw_responses) # to show totals

if __name__ == "__main__":
    import asyncio
    # To test the refactored geocode_row_data function:
    # asyncio.run(main())
    # The ETL script will call geocode_row_data directly.
    print("geocoding_agent.py is intended to be used as a module now.")
    print("To test geocode_row_data function, uncomment asyncio.run(main()) in __main__ block.") 