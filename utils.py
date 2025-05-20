import re
import logging
import pandas as pd
import time
import os
import json
import sys
import ssl
import requests
import random
from threading import Lock
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if running in a terminal 
IS_TERMINAL = sys.stdout.isatty()

# Create a cache for geocoding results
class GeocodingCache:
    """Cache for geocoding results to avoid redundant API calls"""
    def __init__(self, cache_file='geocoding_cache.json'):
        self.cache_file = cache_file
        self.cache = {}
        self.lock = Lock()  # Thread-safe operations
        self.load_cache()
        
    def load_cache(self):
        """Load cache from file if it exists"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logging.info(f"Loaded {len(self.cache)} geocoding entries from cache")
            except Exception as e:
                logging.error(f"Error loading geocoding cache: {str(e)}")
                self.cache = {}
    
    def save_cache(self, force=False):
        """
        Save cache to file
        
        Args:
            force (bool): If True, will save regardless of any conditions
        """
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved {len(self.cache)} entries to geocoding cache")
        except Exception as e:
            logging.error(f"Error saving geocoding cache: {str(e)}")
    
    def get(self, city, county=None):
        """Get value from cache using city and county as key"""
        key = self._make_key(city, county)
        return self.cache.get(key)
    
    def set(self, city, county, value):
        """Set value in cache using city and county as key"""
        key = self._make_key(city, county)
        self.cache[key] = value
        # Save periodically to avoid too much I/O
        if len(self.cache) % 5 == 0:  # Reduced to every 5 entries for more frequent saves
            self.save_cache()
    
    def _make_key(self, city, county):
        """Generate a consistent cache key from city and county"""
        if county:
            return f"{city.strip()}/{county.strip()}/Romania"
        return f"{city.strip()}/Romania"
        
    def clear(self):
        """Clear the cache both in memory and on disk"""
        self.cache = {}
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
                logging.info(f"Deleted cache file: {self.cache_file}")
            except Exception as e:
                logging.error(f"Error deleting cache file {self.cache_file}: {str(e)}")


# Initialize the cache
geocoding_cache = GeocodingCache()


def force_save_cache():
    """Force saving the geocoding cache to disk immediately"""
    logging.info("Forcing geocoding cache save...")
    geocoding_cache.save_cache(force=True)
    return True


# Create a rate limiter to enforce Nominatim's usage policy (1 request per second max)
class RateLimiter:
    """Simple rate limiter to ensure we respect Nominatim's usage policy"""
    def __init__(self, min_interval=1.0):
        self.min_interval = min_interval
        self.last_request_time = 0
        
    def wait(self):
        """Wait until enough time has passed since the last request"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logging.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

# Initialize the rate limiter
rate_limiter = RateLimiter(min_interval=1.1)  # Slightly more than 1s to be safe

# Constants for Nominatim API
NOMINATIM_API_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0"

# Create a session for requests
nominatim_session = requests.Session()
nominatim_session.headers.update({"User-Agent": USER_AGENT})

# Test the Nominatim connection
try:
    rate_limiter.wait()
    response = nominatim_session.get(
        NOMINATIM_API_URL,
        params={
            "q": "Bucharest, Romania",
            "format": "json",
            "limit": 1
        }
    )
    response.raise_for_status()
    test_result = response.json()
    if test_result:
        logging.info("Nominatim API connection test successful")
    else:
        logging.warning("Nominatim API connection test returned empty result")
except Exception as e:
    logging.error(f"Nominatim API connection test failed: {str(e)}")


def extract_clean_city_name(location_str):
    """
    Extract a clean city name from a location string for geocoding.
    Removes any county or additional information.
    
    Args:
        location_str (str): Location string potentially containing city and other information
        
    Returns:
        str: Clean city name for geocoding
    """
    if not location_str or pd.isna(location_str):
        return None
        
    location_str = str(location_str).strip()
    
    # Remove "Localitatea" prefix
    location_str = re.sub(r'^Localitatea\s+', '', location_str, flags=re.IGNORECASE)
    
    # Remove river prefixes
    location_str = re.sub(r'^(râul|raul|r\.|pârâul|paraul|p\.)\s+', '', location_str, flags=re.IGNORECASE)
    
    # Remove anything after comma or opening parenthesis
    location_str = re.sub(r'[,\(].*$', '', location_str).strip()
    
    return location_str

def get_lat_lon(city: str, county: str = None, max_retries=3, initial_delay=2):
    """
    Get the latitude and longitude of a city using direct requests to OSM Nominatim API.
    Uses local caching to avoid redundant API calls.
    
    This function respects Nominatim Usage Policy:
    - Maximum of 1 request per second
    - Proper caching to minimize redundant queries
    - Meaningful user-agent
    
    See: https://operations.osmfoundation.org/policies/nominatim/
    
    Args:
        city (str): Name of the city
        county (str, optional): Name of the county
        max_retries (int): Maximum number of retries for API calls
        initial_delay (int): Initial delay between retries in seconds
        
    Returns:
        tuple: (latitude, longitude) or (None, None) if not found
    """
    if not city:
        return None, None
    
    # Clean the city name
    clean_city = extract_clean_city_name(city)
    if not clean_city:
        return None, None
    
    # Check cache first
    cached_result = geocoding_cache.get(clean_city, county)
    if cached_result is not None:
        return cached_result
    
    # Format the query for the API
    if county:
        query = f"{clean_city}, {county}, Romania"
    else:
        query = f"{clean_city}, Romania"
    
    # Setup request parameters
    params = {
        "q": query,
        "format": "json",
        "limit": 1
    }
    
    # Implement retry with exponential backoff
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            logging.info(f"Geocoding attempt {attempt+1}/{max_retries} for '{query}'")
            
            # Respect Nominatim's rate limit
            rate_limiter.wait()
            
            # Make the API request
            response = nominatim_session.get(NOMINATIM_API_URL, params=params)
            response.raise_for_status()
            
            # Parse the response
            results = response.json()
            
            if results and len(results) > 0:
                # Extract coordinates
                latitude = float(results[0]["lat"])
                longitude = float(results[0]["lon"])
                result = (latitude, longitude)
                
                # Save to cache without forcing an immediate save
                geocoding_cache.set(clean_city, county, result)
                
                return result
            else:
                logging.warning(f"No coordinates found for: '{query}'")
                # Cache the failed result too
                geocoding_cache.set(clean_city, county, (None, None))
                return None, None
                
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors
            status_code = e.response.status_code if e.response else 0
            
            if status_code == 429:  # Too Many Requests
                logging.warning(f"Rate limit exceeded. Waiting {delay * 2}s before retry...")
                time.sleep(delay * 2)
                delay *= 2
            else:
                logging.warning(f"HTTP error for '{query}': {str(e)}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 1.5
                
        except requests.exceptions.Timeout:
            # Handle timeouts
            logging.warning(f"Timeout for '{query}'. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 1.5
            
        except requests.exceptions.RequestException as e:
            # Handle connection errors, SSL errors, etc.
            if 'SSL' in str(e) or 'certificate' in str(e):
                logging.warning(f"SSL error for '{query}': {str(e)}. Retrying in {delay}s...")
            else:
                logging.warning(f"Request error for '{query}': {str(e)}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 1.5
            
        except Exception as e:
            # Handle other errors
            logging.error(f"Error getting coordinates for '{query}': {str(e)}")
            break
    
    # If we get here, all retries failed
    logging.error(f"Failed to geocode '{query}' after {max_retries} attempts")
    geocoding_cache.set(clean_city, county, (None, None))
    return None, None


def batch_geocode(locations):
    """
    Process locations sequentially with progress display and robust error handling.
    Uses direct requests to the Nominatim API, strictly respects usage policy of maximum 1 request per second.
    Randomizes locations before processing to avoid hitting the same geographic regions in sequence.
    
    Args:
        locations: List of (city, county) tuples
        
    Returns:
        List of (latitude, longitude) tuples in the same order as the input locations
    """
    # Create a mapping to preserve original order
    original_indices = {loc: idx for idx, loc in enumerate(locations)}
    results = [None] * len(locations)
    
    # Skip processing if no locations to geocode
    if not locations:
        return []
    
    # Debug logging for geocoding requests
    logging.info(f"Starting batch geocoding for {len(locations)} locations")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_locations = []
    for loc in locations:
        if loc not in seen and loc[0]:  # Ensure city is not None or empty
            seen.add(loc)
            unique_locations.append(loc)
    
    logging.info(f"After deduplication: {len(unique_locations)} unique locations to process")
    
    # Check which locations are already in the cache
    uncached_locations = []
    location_to_coords = {}
    
    for city, county in unique_locations:
        cached_result = geocoding_cache.get(city, county)
        if cached_result is not None:
            location_to_coords[(city, county)] = cached_result
            logging.info(f"Cache hit for '{city}', '{county}': {cached_result}")
        else:
            uncached_locations.append((city, county))
            logging.info(f"Cache miss for '{city}', '{county}', will fetch from API")
    
    logging.info(f"Need to geocode {len(uncached_locations)} locations (not in cache)")
    
    # Randomize the uncached locations to avoid hitting same geographic areas in sequence
    if uncached_locations:
        random.shuffle(uncached_locations)
        logging.info("Randomized the order of locations to geocode")
    
    # Process uncached locations sequentially (single worker)
    if uncached_locations:
        total = len(uncached_locations)
        
        for i, (city, county) in enumerate(uncached_locations):
            try:
                # Get coordinates with direct API request
                coords = get_lat_lon(city, county)
                location_to_coords[(city, county)] = coords
                logging.info(f"Geocoded '{city}', '{county}' to {coords}")
            except Exception as e:
                logging.error(f"Error geocoding '{city}', '{county}': {str(e)}")
                location_to_coords[(city, county)] = (None, None)
            
            # Update progress in terminal
            if IS_TERMINAL:
                progress = (i + 1) / total * 100
                sys.stdout.write(f"\rGeocoding: {progress:.1f}% ({i+1}/{total})")
                sys.stdout.flush()
            
            # Save cache every 10 locations
            if (i + 1) % 10 == 0 or (i + 1) == total:
                try:
                    geocoding_cache.save_cache()
                except Exception as e:
                    logging.error(f"Error saving cache: {str(e)}")
        
        # Add a newline after progress display
        if IS_TERMINAL:
            sys.stdout.write("\n")
            sys.stdout.flush()
    
    # Map results back to original order
    for i, (city, county) in enumerate(locations):
        results[i] = location_to_coords.get((city, county), (None, None))
    
    # Final save cache to disk
    try:
        geocoding_cache.save_cache()
        logging.info("Geocoding cache saved successfully")
    except Exception as e:
        logging.error(f"Final cache save failed: {str(e)}")
    
    logging.info(f"Batch geocoding completed for {len(locations)} locations")
    
    return results

def categorize_habitat(habitat_name):
    """
    Categorize a habitat based on its name.
    
    Args:
        habitat_name (str): Name of the habitat
        
    Returns:
        str: Category of the habitat (River, Lake/Reservoir, Canal, Stream, Other)
    """
    habitat = str(habitat_name).lower()
    if any(word in habitat for word in ['râul', 'râu']):
        return 'River'
    elif any(word in habitat for word in ['lac', 'balta', 'acumulare']):
        return 'Lake/Reservoir'
    elif any(word in habitat for word in ['canal']):
        return 'Canal'
    elif any(word in habitat for word in ['pârâul', 'pârâu', 'valea']):
        return 'Stream'
    else:
        return 'Other'

def extract_locations_and_types(limit_str):
    """
    Extract start and end locations, types, and rivers from a limit string.
    
    Args:
        limit_str (str): String describing the limits of a waterway
        
    Returns:
        tuple: (start_location, end_location, start_type, end_type, start_from_spring, end_river)
    """
    if not limit_str or pd.isna(limit_str) or limit_str == '':
        return None, None, None, None, False, None
    
    limit_str = str(limit_str).strip()
    
    # Initialize values
    start_location = None
    end_location = None
    start_type = None
    end_type = None
    start_from_spring = False
    end_river = None
    
    # Special location type keywords
    spring_terms = ['izvor', 'izvoare', 'izvorului', 'izvoarelor']
    bridge_terms = ['pod', 'podul', 'podului']
    dam_terms = ['baraj', 'barajul', 'barajului', 'acumulare']
    confluence_terms = ['conf.', 'confluență', 'confluenta', 'vărsare', 'varsare']
    boundary_terms = ['limita', 'limitei', 'limitelor', 'hotar', 'hotarul']
    entrance_terms = ['intrare', 'intrarea', 'intrării']
    gorge_terms = ['cheile', 'chei', 'cheiul', 'cheiului']
    border_terms = ['frontieră', 'frontiera', 'graniță', 'granita', 'granița', 'granitei', 'graniței']
    
    # City indicators
    city_indicators = ['oraș', 'orasul', 'orașul', 'municipiul', 'comuna', 'sat', 'satul', 'localitatea', 'city', 'town', 'village']
    
    # Non-city indicators
    non_city_terms = spring_terms + bridge_terms + dam_terms + confluence_terms + boundary_terms + entrance_terms + gorge_terms + border_terms
    non_city_terms += ['km', 'amonte', 'aval', 'confluenţa', 'râului', 'raului', 'pârâului', 'paraului']
    non_city_terms += ['județ', 'judetul', 'județean', 'județeană', 'judet', 'judetean', 'judeteana']
    
    # Check for phrases like "from city X to city Y"
    from_to_pattern = r'(?i)(?:from|de la)\s+(?:the\s+)?(?:city\s+of\s+|orașul\s+|localitatea\s+)?([A-Za-zĂăÂâÎîȘșȚț\s-]+?)(?:\s+to|\s+până la|\s+pana la|\s+la)\s+(?:the\s+)?(?:city\s+of\s+|orașul\s+|localitatea\s+)?([A-Za-zĂăÂâÎîȘșȚț\s-]+)'
    from_to_match = re.search(from_to_pattern, limit_str)
    
    if from_to_match:
        start_location = from_to_match.group(1).strip()
        end_location = from_to_match.group(2).strip()
        start_type = 'City'
        end_type = 'City'
        return start_location, end_location, start_type, end_type, start_from_spring, end_river
    
    # Check for various delimiters
    delimiters = ['-', '–', 'până la', 'pana la', 'to the', 'to']
    
    # Helper function to check if text contains city indicators
    def contains_city_indicator(text):
        lower_text = text.lower()
        for indicator in city_indicators:
            if indicator in lower_text:
                return True
        return False
    
    # First check for simple delimiters
    for delimiter in delimiters:
        if delimiter in limit_str:
            parts = limit_str.split(delimiter, 1)
            
            # Process start location
            start_part = parts[0].strip()
            start_type_found = False
            
            # Check for special types in start location
            for term in spring_terms:
                if term in start_part.lower():
                    start_type = 'Spring'
                    start_from_spring = True
                    start_type_found = True
                    break
                    
            for term in bridge_terms:
                if term in start_part.lower():
                    start_type = 'Bridge'
                    start_type_found = True
                    break
                    
            for term in dam_terms:
                if term in start_part.lower():
                    start_type = 'Dam/Reservoir'
                    start_type_found = True
                    break
            
            for term in boundary_terms:
                if term in start_part.lower():
                    start_type = 'Boundary'
                    start_type_found = True
                    break
            
            for term in entrance_terms:
                if term in start_part.lower():
                    start_type = 'Entrance'
                    start_type_found = True
                    break
                    
            for term in gorge_terms:
                if term in start_part.lower():
                    start_type = 'Gorge'
                    start_type_found = True
                    break
                    
            for term in border_terms:
                if term in start_part.lower():
                    start_type = 'Border'
                    start_type_found = True
                    break
            
            # If no special type identified, check if it might be a city
            if not start_type_found:
                # If it has city indicators or doesn't contain non-city terms, it's likely a city
                if contains_city_indicator(start_part) or not any(term in start_part.lower() for term in non_city_terms):
                    # Further clean the location: remove river prefixes, etc.
                    cleaned_start = re.sub(r'^(râul|raul|r\.|pârâul|paraul|p\.)\s+', '', start_part, flags=re.IGNORECASE).strip()
                    # If it looks like a proper location name (first letter capitalized, not just numbers)
                    if cleaned_start and not cleaned_start[0].isdigit() and any(c.isalpha() for c in cleaned_start):
                        start_type = 'City'
            
            # Set start location
            start_location = start_part
            
            # Process end location if it exists
            if len(parts) > 1:
                end_part = parts[1].strip()
                end_type_found = False
                
                # Check for confluence in end part
                for term in confluence_terms:
                    if term in end_part.lower():
                        end_type = 'Confluence'
                        end_type_found = True
                        
                        # Try to extract the river name
                        river_pattern = r'(conf\.|confluență|confluenta|vărsare|varsare)(\s+cu|\s+în|\s+in|\s+)?\s+(râul|raul|r\.|pârâul|paraul|p\.)?\s*([A-Za-zĂăÂâÎîȘșȚț\s-]+)'
                        river_match = re.search(river_pattern, end_part, re.IGNORECASE)
                        
                        if river_match:
                            end_river = river_match.group(4).strip()
                        break
                        
                # Check for other special types in end location
                if not end_type_found:
                    for term in bridge_terms:
                        if term in end_part.lower():
                            end_type = 'Bridge'
                            end_type_found = True
                            break
                            
                    for term in dam_terms:
                        if term in end_part.lower():
                            end_type = 'Dam/Reservoir'
                            end_type_found = True
                            break
                    
                    for term in boundary_terms:
                        if term in end_part.lower():
                            end_type = 'Boundary'
                            end_type_found = True
                            break
                    
                    for term in entrance_terms:
                        if term in end_part.lower():
                            end_type = 'Entrance'
                            end_type_found = True
                            break
                            
                    for term in gorge_terms:
                        if term in end_part.lower():
                            end_type = 'Gorge'
                            end_type_found = True
                            break
                            
                    for term in border_terms:
                        if term in end_part.lower():
                            end_type = 'Border'
                            end_type_found = True
                            break
                            
                    # If no special type identified, check if it might be a city
                    if not end_type_found:
                        # If it has city indicators or doesn't contain non-city terms, it's likely a city
                        if contains_city_indicator(end_part) or not any(term in end_part.lower() for term in non_city_terms):
                            # Further clean the location: remove river prefixes, etc.
                            cleaned_end = re.sub(r'^(râul|raul|r\.|pârâul|paraul|p\.)\s+', '', end_part, flags=re.IGNORECASE).strip()
                            # If it looks like a proper location name (first letter capitalized, not just numbers)
                            if cleaned_end and not cleaned_end[0].isdigit() and any(c.isalpha() for c in cleaned_end):
                                end_type = 'City'
                
                # Set end location
                end_location = end_part
            
            return start_location, end_location, start_type, end_type, start_from_spring, end_river
    
    # Check for single location that might be a confluence
    for term in confluence_terms:
        if term in limit_str.lower():
            end_type = 'Confluence'
            
            # Try to extract the river name
            river_pattern = r'(conf\.|confluență|confluenta|vărsare|varsare)(\s+cu|\s+în|\s+in|\s+)?\s+(râul|raul|r\.|pârâul|paraul|p\.)?\s*([A-Za-zĂăÂâÎîȘșȚț\s-]+)'
            river_match = re.search(river_pattern, limit_str, re.IGNORECASE)
            
            if river_match:
                end_river = river_match.group(4).strip()
                end_location = limit_str
            return None, end_location, None, end_type, False, end_river
    
    # Check for specific location types if no delimiter was found
    start_type_found = False
    
    for term in spring_terms:
        if term in limit_str.lower():
            start_type = 'Spring'
            start_from_spring = True
            start_type_found = True
            break
            
    for term in bridge_terms:
        if term in limit_str.lower():
            start_type = 'Bridge'
            start_type_found = True
            break
            
    for term in dam_terms:
        if term in limit_str.lower():
            start_type = 'Dam/Reservoir'
            start_type_found = True
            break
    
    for term in boundary_terms:
        if term in limit_str.lower():
            start_type = 'Boundary'
            start_type_found = True
            break
    
    for term in entrance_terms:
        if term in limit_str.lower():
            start_type = 'Entrance'
            start_type_found = True
            break
            
    for term in gorge_terms:
        if term in limit_str.lower():
            start_type = 'Gorge'
            start_type_found = True
            break
            
    for term in border_terms:
        if term in limit_str.lower():
            start_type = 'Border'
            start_type_found = True
            break
            
    # If no special type identified, check if it might be a city
    if not start_type_found:
        # If it has city indicators or doesn't contain non-city terms, it's likely a city
        if contains_city_indicator(limit_str) or not any(term in limit_str.lower() for term in non_city_terms):
            # Further clean the location: remove river prefixes, etc.
            cleaned_str = re.sub(r'^(râul|raul|r\.|pârâul|paraul|p\.)\s+', '', limit_str, flags=re.IGNORECASE).strip()
            # If it looks like a proper location name (first letter capitalized, not just numbers)
            if cleaned_str and not cleaned_str[0].isdigit() and any(c.isalpha() for c in cleaned_str):
                start_type = 'City'
    
    # If no delimiter found, just return the whole string as start location
    return limit_str, None, start_type, None, start_from_spring, None

def geocode_locations(df):
    """
    Add geocoding information to dataframe with location data.
    Uses the County column from the dataframe to improve geocoding.
    
    Args:
        df (pandas.DataFrame): DataFrame with columns 'start_location', 'end_location', 
                              'start_type', 'end_type', and 'County'
        
    Returns:
        pandas.DataFrame: DataFrame with added columns for coordinates
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Add coordinate columns if they don't exist
    for col in ['start_lat', 'start_lon', 'end_lat', 'end_lon']:
        if col not in result_df.columns:
            result_df[col] = None
    
    # Get all city locations that need geocoding
    geocoding_requests = []
    for idx, row in result_df.iterrows():
        county = row.get('County', None)
        
        # Only geocode City type locations
        if row.get('start_type') == 'City' and pd.notna(row.get('start_location')):
            clean_city = extract_clean_city_name(row['start_location'])
            if clean_city:
                geocoding_requests.append((idx, 'start', clean_city, county))
        
        if row.get('end_type') == 'City' and pd.notna(row.get('end_location')):
            clean_city = extract_clean_city_name(row['end_location'])
            if clean_city:
                geocoding_requests.append((idx, 'end', clean_city, county))
    
    # Skip if no geocoding needed
    if not geocoding_requests:
        return result_df
    
    # Log how many locations need geocoding
    logging.info(f"Geocoding {len(geocoding_requests)} locations")
    
    # Prepare batch geocoding requests
    batch_requests = []
    for _, _, city, county in geocoding_requests:
        batch_requests.append((city, county))
    
    # Get coordinates in batch
    coords_results = batch_geocode(batch_requests)
    
    # Update DataFrame with results
    for i, (idx, loc_type, _, _) in enumerate(geocoding_requests):
        lat, lon = coords_results[i]
        
        if loc_type == 'start':
            result_df.at[idx, 'start_lat'] = lat
            result_df.at[idx, 'start_lon'] = lon
        else:  # end
            result_df.at[idx, 'end_lat'] = lat
            result_df.at[idx, 'end_lon'] = lon
    
    # Ensure the cache is saved after all operations
    try:
        geocoding_cache.save_cache()
        logging.info("Final geocoding cache saved successfully")
    except Exception as e:
        logging.error(f"Error saving final geocoding cache: {str(e)}")
    
    return result_df

def is_stream_ending_in_river(waterway_name, limit_info):
    """
    Determine if a waterway is a stream that ends in another river.
    
    Args:
        waterway_name (str): Name of the waterway
        limit_info (str): String describing the limits of the waterway
        
    Returns:
        tuple: (is_stream_to_river, river_name, starts_at_spring) where:
               - is_stream_to_river is a boolean
               - river_name is the name of the river (if applicable)
               - starts_at_spring is a boolean indicating if the stream starts at a spring
    """
    # Check if it's a stream based on its name
    habitat_type = categorize_habitat(waterway_name)
    
    if habitat_type != 'Stream':
        return False, None, False
    
    # Check if it ends in a river and/or starts at a spring
    result = extract_locations_and_types(limit_info)
    
    # Unpack the values from the result
    _, _, start_type, end_type, starts_at_spring, end_river = result
    
    # Check if it starts at a spring
    is_starting_at_spring = starts_at_spring or start_type == 'Spring'
    
    # Check if it ends in a river
    if end_type == 'Confluence' and end_river:
        return True, end_river, is_starting_at_spring
    
    return False, None, is_starting_at_spring 

def find_closest_point_on_river(city_lat, city_lon, river_coordinates):
    """
    Find the point on a river that is closest to a given city's coordinates.
    
    Args:
        city_lat (float): Latitude of the city
        city_lon (float): Longitude of the city
        river_coordinates (list): List of (lon, lat) tuples representing the river's coordinates
        
    Returns:
        tuple: (index, distance, coordinates)
               - index: Index of the closest point in the river_coordinates list
               - distance: Distance in kilometers to the closest point
               - coordinates: (lon, lat) of the closest point
    """
    if not city_lat or not city_lon or not river_coordinates:
        return None, None, None
    
    # Convert city coordinates to radians
    city_lat_rad = math.radians(float(city_lat))
    city_lon_rad = math.radians(float(city_lon))
    
    closest_idx = None
    min_distance = float('inf')
    closest_point = None
    
    # Go through all points in the river
    for i, (point_lon, point_lat) in enumerate(river_coordinates):
        # Convert point coordinates to radians
        point_lat_rad = math.radians(float(point_lat))
        point_lon_rad = math.radians(float(point_lon))
        
        # Calculate Haversine distance
        dlon = point_lon_rad - city_lon_rad
        dlat = point_lat_rad - city_lat_rad
        a = math.sin(dlat/2)**2 + math.cos(city_lat_rad) * math.cos(point_lat_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = 6371 * c  # Earth radius in km
        
        # Update minimum if found a closer point
        if distance < min_distance:
            min_distance = distance
            closest_idx = i
            closest_point = (point_lon, point_lat)
    
    return closest_idx, min_distance, closest_point


def extract_river_segment(river_coordinates, start_idx, end_idx):
    """
    Extract a segment of a river between two points.
    
    Args:
        river_coordinates (list): List of (lon, lat) tuples representing the river's coordinates
        start_idx (int): Index of the start point
        end_idx (int): Index of the end point
        
    Returns:
        list: List of (lon, lat) tuples representing the segment
    """
    # Ensure start_idx <= end_idx
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    
    # Extract the segment
    return river_coordinates[start_idx:end_idx+1] 