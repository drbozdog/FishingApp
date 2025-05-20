import os
import re
import xml.etree.ElementTree as ET
from pykml import parser as kml_parser
import pandas as pd
import logging
import utils
import hashlib
import colorsys
from tqdm import tqdm
import csv
from collections import Counter
import argparse
import unicodedata
import requests
import json
import random
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a matching log file to track record matching details
matching_log_path = 'matching_log.csv'
matching_log_file = None
matching_log_writer = None

# Statistics counters
match_stats = {
    'total': 0,
    'spring_to_river': 0,
    'city_to_city': 0,
    'spring_to_city': 0,  # New segment type
    'city_to_river': 0,   # New segment type
    'other': 0,
    'matched': 0,
    'match_types': Counter(),
    'unmatched': 0
}

# Debug filter settings - specify waterway and county to filter
DEBUG_FILTER = {
    'enabled': False,
    'waterway': None,
    'county': None,
    'include_all_matching': False  # Flag to include all matching waterways
}

def initialize_matching_log():
    """Initialize the matching log file with headers"""
    global matching_log_file, matching_log_writer
    
    matching_log_file = open(matching_log_path, 'w', newline='', encoding='utf-8')
    matching_log_writer = csv.writer(matching_log_file)
    
    # Write headers
    matching_log_writer.writerow([
        'Habitat Name', 'River Name', 'Start Location', 'End Location', 
        'Start Type', 'End Type', 'County', 'Matched', 'Match Type', 'Notes'
    ])
    logging.info(f"Initialized matching log file: {matching_log_path}")

def log_matching_result(habitat_name, river_name, start_location, end_location, 
                       start_type, end_type, county, matched, match_type="", notes=""):
    """
    Log a matching result to the CSV file and console
    
    Args:
        habitat_name (str): Name of the habitat/waterway
        river_name (str): Name of the river (if applicable)
        start_location (str): Start location
        end_location (str): End location
        start_type (str): Type of start location (City, Spring, etc.)
        end_type (str): Type of end location (City, River, etc.)
        county (str): County name
        matched (bool): Whether the record was matched to a waterway in KML
        match_type (str): Type of match (exact, normalized, fuzzy)
        notes (str): Additional notes
    """
    global matching_log_writer, match_stats
    
    # Ensure all values are strings
    habitat_name = str(habitat_name) if habitat_name is not None else ""
    river_name = str(river_name) if river_name is not None else ""
    start_location = str(start_location) if start_location is not None else ""
    end_location = str(end_location) if end_location is not None else ""
    start_type = str(start_type) if start_type is not None else ""
    end_type = str(end_type) if end_type is not None else ""
    county = str(county) if county is not None else ""
    matched_str = "Yes" if matched else "No"
    
    # Write to the log file
    matching_log_writer.writerow([
        habitat_name, river_name, start_location, end_location,
        start_type, end_type, county, matched_str, match_type, notes
    ])
    
    # Update statistics
    match_stats['total'] += 1
    
    # Classify record type
    is_spring_to_river = False
    is_city_to_city = False
    is_spring_to_city = False  # New segment type
    is_city_to_river = False   # New segment type
    
    # Check for Spring to River segment (include Confluence as a valid end type)
    if "Spring to river stream" in notes or start_type == 'Spring' and (end_type == 'River' or end_type == 'Confluence'):
        match_stats['spring_to_river'] += 1
        is_spring_to_river = True
    # Check for City to City segment
    elif "City to city segment" in notes or (start_type == 'City' and end_type == 'City'):
        match_stats['city_to_city'] += 1
        is_city_to_city = True
    # Check for Spring to City segment
    elif start_type == 'Spring' and end_type == 'City':
        match_stats['spring_to_city'] += 1
        is_spring_to_city = True
    # Check for City to River segment
    elif start_type == 'City' and (end_type == 'River' or end_type == 'Confluence'):
        match_stats['city_to_river'] += 1
        is_city_to_river = True
    else:
        match_stats['other'] += 1
    
    # Track match status
    if matched:
        match_stats['matched'] += 1
        if match_type:
            match_stats['match_types'][match_type] += 1
    else:
        match_stats['unmatched'] += 1
    
    # Print to console - make format compact but informative
    if is_spring_to_river:
        segment_type = "SPRING→RIVER"
    elif is_city_to_city:
        segment_type = "CITY→CITY"
    elif is_spring_to_city:
        segment_type = "SPRING→CITY"
    elif is_city_to_river:
        segment_type = "CITY→RIVER"
    else:
        segment_type = "OTHER"
        
    match_status = f"✓{match_type}" if matched else "✗"
    
    # Format the log line
    log_line = f"{segment_type} | {match_status} | {habitat_name[:30]}"
    if river_name:
        log_line += f" → {river_name[:20]}"
    if start_location and end_location:
        log_line += f" | {start_location[:15]} → {end_location[:15]}"
    
    print(log_line)

def finalize_matching_log():
    """Close the matching log file and print summary statistics"""
    global matching_log_file, match_stats
    if matching_log_file:
        matching_log_file.close()
        logging.info(f"Finalized matching log file: {matching_log_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("MATCHING STATISTICS SUMMARY")
    print("="*80)
    
    total = match_stats['total']
    if total > 0:
        print(f"Total records processed: {total}")
        print(f"  Spring to river segments: {match_stats['spring_to_river']} ({match_stats['spring_to_river']/total*100:.1f}%)")
        print(f"  City to city segments: {match_stats['city_to_city']} ({match_stats['city_to_city']/total*100:.1f}%)")
        print(f"  Spring to city segments: {match_stats['spring_to_city']} ({match_stats['spring_to_city']/total*100:.1f}%)")
        print(f"  City to river segments: {match_stats['city_to_river']} ({match_stats['city_to_river']/total*100:.1f}%)")
        print(f"  Other segment types: {match_stats['other']} ({match_stats['other']/total*100:.1f}%)")
        print(f"\nMatched to KML waterways: {match_stats['matched']} ({match_stats['matched']/total*100:.1f}%)")
        print(f"Unmatched: {match_stats['unmatched']} ({match_stats['unmatched']/total*100:.1f}%)")
        
        if match_stats['matched'] > 0:
            print("\nMatch types breakdown:")
            for match_type, count in match_stats['match_types'].most_common():
                print(f"  {match_type}: {count} ({count/match_stats['matched']*100:.1f}%)")
    else:
        print("No records were processed.")
    
    print("="*80)

def normalize_waterway_name(name):
    """
    Normalize waterway names by removing prefixes like "Râul", "Pârâul", etc.
    
    Args:
        name (str): Waterway name
        
    Returns:
        str: Normalized name
    """
    if not name:
        return ""
    
    name = str(name).strip()
    
    # Remove prefixes - expanded with more variations
    prefixes = [
        'râul', 'râu', 'raul', 'rau', 'r.',
        'pârâul', 'pârâu', 'paraul', 'parau', 'p.', 
        'valea', 'vale', 'v.'
    ]
    name_lower = name.lower()
    
    # Keep trying prefixes until no more can be removed (handles double prefixes)
    prefix_removed = True
    while prefix_removed:
        prefix_removed = False
        for prefix in prefixes:
            if name_lower.startswith(prefix):
                # Remove the prefix and trim
                name = name[len(prefix):].strip()
                # Update lowercase version
                name_lower = name.lower()
                prefix_removed = True
                break
    
    # Capitalize the first letter if needed
    if name and name[0].islower():
        name = name[0].upper() + name[1:]
    
    # Remove suffixes like "superior", "inferior", "mare", "mic", etc.
    suffixes = [' superior', ' inferior', ' mare', ' mic', ' de sus', ' de jos']
    for suffix in suffixes:
        if name_lower.endswith(suffix):
            name = name[:-len(suffix)].strip()
            # Update lowercase version
            name_lower = name.lower()
    
    # Remove special characters and extra spaces
    name = re.sub(r'[^\w\sȘșȚțĂăÎîÂâ]', ' ', name)  # Keep Romanian characters
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def generate_color_for_association(association):
    """
    Generate a consistent color for an association based on its name or ID.
    Returns the color in AABBGGRR format for KML.
    
    Args:
        association (str): Association name or ID
        
    Returns:
        str: KML color in AABBGGRR format
    """
    # Convert association to string if it's not
    assoc_str = str(association)
    
    # Generate a hash value from the association name/ID
    hash_obj = hashlib.md5(assoc_str.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    
    # Use the hash to create a hue value (0-1)
    hue = (hash_int % 1000) / 1000.0
    
    # Convert HSV to RGB (using full saturation and value)
    rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    
    # Convert to 8-bit RGB values
    r = int(rgb[0] * 255)
    g = int(rgb[1] * 255)
    b = int(rgb[2] * 255)
    
    # Return KML color (AABBGGRR format)
    return f"ff{b:02x}{g:02x}{r:02x}"

def process_waterway_between_cities(start_city, end_city, start_county, end_county, waterway_data, association, limit_info, output_kml, streams_to_rivers, associations, association_names=None):
    """
    Process a waterway segment between two cities, finding the closest points on the waterway to the cities.
    
    Args:
        start_city (str): Name of the start city
        end_city (str): Name of the end city
        start_county (str): County of the start city (optional)
        end_county (str): County of the end city (optional)
        waterway_data (dict): Data for the waterway from KML
        association (str): Name of the association managing this segment
        limit_info (str): Limit information string
        output_kml: KML document to add the segment to
        streams_to_rivers: List to add the segment information to
        associations: Dictionary of associations and their colors
        association_names: Dictionary of association name mappings
    """
    # Get coordinates for start and end cities
    start_lat, start_lon = utils.get_lat_lon(start_city, start_county)
    end_lat, end_lon = utils.get_lat_lon(end_city, end_county)
    
    if not start_lat or not start_lon or not end_lat or not end_lon:
        logging.warning(f"Could not geocode one or both cities: {start_city}, {end_city}")
        return None
    
    # Find closest points on the waterway to the cities
    waterway_coords = waterway_data.get('coordinates', [])
    if not waterway_coords:
        logging.warning(f"No coordinates found for waterway")
        return None
    
    # Find closest point to start city
    start_idx, start_dist, start_point = utils.find_closest_point_on_river(start_lat, start_lon, waterway_coords)
    
    # Find closest point to end city
    end_idx, end_dist, end_point = utils.find_closest_point_on_river(end_lat, end_lon, waterway_coords)
    
    if start_idx is None or end_idx is None:
        logging.warning(f"Could not find closest points on waterway for {start_city} and {end_city}")
        return None
    
    # Extract the segment of the waterway between these points
    segment_coords = utils.extract_river_segment(waterway_coords, start_idx, end_idx)
    
    if len(segment_coords) < 2:
        logging.warning(f"Segment between {start_city} and {end_city} has too few points ({len(segment_coords)})")
        return None
    
    logging.info(f"Found waterway segment between {start_city} and {end_city} ({len(segment_coords)} points)")
    
    # Process association name
    if association_names and (isinstance(association, (int, float)) or (isinstance(association, str) and association.isdigit())):
        if str(association) in association_names:
            association = association_names[str(association)]
    
    display_association = association
    
    # Create a safe style ID from the association name
    association_id = re.sub(r'[^a-zA-Z0-9]', '', str(association))
    if not association_id:
        association_id = "unknown"
    
    # Ensure we have a color for this association
    if association not in associations:
        associations[association] = generate_color_for_association(association)
    
    # Create a description for the segment
    description = f"Waterway segment from {start_city} to {end_city}.<br/><b>Association:</b> {display_association}<br/><b>Limit info:</b> {limit_info}"
    
    # Create a name for the segment
    waterway_name = waterway_data.get('name', 'Unnamed Waterway')
    segment_name = f"{waterway_name}: {start_city} to {end_city}"
    
    # Add this segment to the KML
    placemark_xml = create_placemark_xml(
        name=segment_name,
        description=description,
        coordinates=segment_coords,
        association=display_association,
        association_id=association_id,
        style_color=associations[association]
    )
    
    # Find the folder to add the placemark to
    city_folder = output_kml.find('./Document/Folder[@id="city_to_city_folder"]')
    if city_folder is None:
        logging.warning("Could not find City to City folder - using first folder")
        city_folder = output_kml.find('./Document/Folder')
    
    if city_folder is not None:
        city_folder.append(placemark_xml)
    else:
        logging.error("Could not find any folder to add city segment placemark")
    
    # Add this segment to our output list
    segment_info = {
        'waterway_name': waterway_name,
        'start_city': start_city,
        'end_city': end_city,
        'start_coordinates': segment_coords[0],
        'end_coordinates': segment_coords[-1],
        'association': display_association,
        'limit_info': limit_info,
        'segment_length': len(segment_coords),
        'segment_type': 'city_to_city'
    }
    
    streams_to_rivers.append(segment_info)
    return segment_info

def process_kml(input_kml_path, output_kml_path):
    """
    Process the KML file to extract:
    1. Streams that start at springs and end at confluences with other rivers
    2. River segments between specified cities
    
    Args:
        input_kml_path (str): Path to the input KML file
        output_kml_path (str): Path to the output KML file
    """
    try:
        logging.info(f"Processing KML file: {input_kml_path}")
        
        # Initialize the matching log
        initialize_matching_log()
        
        # Print header for console logs
        print("\n" + "="*80)
        print("WATERWAY MATCHING LOGS")
        print("FORMAT: [TYPE] | [MATCH_STATUS] | [HABITAT_NAME] → [RIVER_NAME] | [START] → [END]")
        print("="*80)
        
        # Print debug filter info if enabled
        if DEBUG_FILTER['enabled']:
            print(f"\nDEBUG FILTER ACTIVE:")
            print(f"  Waterway: {DEBUG_FILTER['waterway']}")
            print(f"  County: {DEBUG_FILTER['county']}")
            print(f"  Include all matching: {DEBUG_FILTER['include_all_matching']}")
            print("="*80)
            
        # Parse the KML file
        logging.info("Parsing KML file...")
        with open(input_kml_path, 'rb') as f:
            root = kml_parser.parse(f).getroot()
        
        # Create the output KML structure
        output_kml = create_output_kml_structure()
        
        # Extract namespace from the root element
        ns = extract_namespace(root)
        
        # Find all placemarks
        placemarks = find_all_placemarks(root, ns)
        logging.info(f"Found {len(placemarks)} placemarks in KML file")
        
        # Dictionary to store all waterways by name for quick lookup
        waterways_by_name = {}
        waterway_limits = {}
        waterways_by_normalized_name = {}
        
        # For debugging - store ALL waterway names
        all_waterway_names = []
        
        # Dictionary to store association name mappings
        association_names = {}
        
        # Keep track of all waterways that match our debug filter
        debug_waterways = []  # Generic collection for waterways matching debug filter
        
        # Print direct debug info about the KML search
        debug_term = DEBUG_FILTER['waterway'].lower() if DEBUG_FILTER['enabled'] and DEBUG_FILTER['waterway'] else None
        # Also create normalized version of debug term for special char matching
        normalized_debug_term = normalize_special_chars(debug_term).lower() if debug_term else None
        
        if debug_term:
            print(f"\nSearching for waterways with '{debug_term}' as a complete word in their name or attributes")
            print(f"Also using normalized version without special characters: '{normalized_debug_term}'")
            print(f"Will reject partial matches like 'Mureșel' when searching for 'Mureș'")
            print("="*80)

        # First pass: collect all waterways and their data
        logging.info("Processing placemarks and extracting data...")
        for placemark in tqdm(placemarks, desc="Extracting placemark data"):
            placemark_data = extract_placemark_data(placemark, ns)
            if placemark_data and 'name' in placemark_data:
                name = placemark_data['name']
                normalized_name = normalize_waterway_name(name)
                
                # Also normalize special characters for matching
                ascii_name = normalize_special_chars(name).lower()
                
                # Store for full name listing
                all_waterway_names.append(name)
                
                # Print direct debug info for waterways with our search term as an exact word match
                if debug_term and (is_exact_word_match(debug_term, name.lower()) or 
                                  is_exact_word_match(normalized_debug_term, ascii_name)):
                    print(f"FOUND EXACT WORD MATCH: '{name}' contains word '{debug_term}' (normalized as '{ascii_name}')")
                
                # Check if this waterway matches our debug filter (if enabled)
                if DEBUG_FILTER['enabled'] and DEBUG_FILTER['waterway']:
                    is_match = False
                    match_reason = ""
                    
                    # Check name match - both with special chars and normalized - must be exact word match
                    if is_exact_word_match(debug_term, name.lower()):
                        is_match = True
                        match_reason = f"Name contains word '{debug_term}'"
                    elif is_exact_word_match(normalized_debug_term, ascii_name):
                        is_match = True
                        match_reason = f"Normalized name '{ascii_name}' contains word '{normalized_debug_term}'"
                    
                    # Also check attributes for matches - must be exact word match
                    if not is_match:
                        for key, value in placemark_data.items():
                            if not isinstance(value, str):
                                continue
                                
                            # Try matching both the original value and normalized value
                            value_lower = value.lower()
                            value_normalized = normalize_special_chars(value).lower()
                            
                            if is_exact_word_match(debug_term, value_lower):
                                is_match = True
                                match_reason = f"Attribute '{key}' value '{value}' contains word '{debug_term}'"
                                break
                            elif is_exact_word_match(normalized_debug_term, value_normalized):
                                is_match = True
                                match_reason = f"Normalized attribute '{key}' value '{value_normalized}' contains word '{normalized_debug_term}'"
                                break
                            
                    # Special case for exact waterway name matches - always include these
                    if name.lower() == debug_term or normalize_special_chars(name).lower() == normalized_debug_term:
                        is_match = True
                        match_reason = f"Exact waterway name match for '{debug_term}'"
                        
                    if is_match:
                        debug_waterways.append({
                            'name': name,
                            'data': placemark_data,
                            'source_id': placemark_data.get('source', 'Unknown'),
                            'match_reason': match_reason
                        })
                        print(f"FOUND MATCH: {name} - {match_reason}")
                
                waterways_by_name[name] = placemark_data
                waterways_by_normalized_name[normalized_name] = placemark_data
                
                # For debugging
                if placemark_data.get('waterway_type', '').lower() == 'stream':
                    logging.debug(f"Found stream: {name} (normalized: {normalized_name})")
                
                # Extract limit information if available
                description = placemark_data.get('description', '')
                if description:
                    waterway_limits[name] = description
                    waterway_limits[normalized_name] = description  # Store for normalized name too
        
        # If debugging waterways and none found, show all waterway names
        if debug_term and not debug_waterways:
            print(f"\nNO MATCHES FOUND FOR '{debug_term}'. Here are all waterway names in the KML:")
            # Sort and display names for easier reading
            all_waterway_names.sort()
            for i, name in enumerate(all_waterway_names):
                if i < 200:  # Show more names to help find the right one
                    print(f"  {i+1}. {name}")
                else:
                    print(f"  ... and {len(all_waterway_names) - 200} more")
                    break
            print("="*80)
        
        # Log all waterways matching debug term found in the KML, regardless of any filters
        if DEBUG_FILTER['enabled'] and DEBUG_FILTER['waterway'] and debug_waterways:
            print(f"\nFOUND {len(debug_waterways)} waterways matching '{debug_term}' in the KML file:")
            
            # Group waterways by name for better organization
            waterway_groups = {}
            for waterway in debug_waterways:
                name = waterway['name']
                if name not in waterway_groups:
                    waterway_groups[name] = []
                waterway_groups[name].append(waterway)
            
            # Print info about each waterway
            for idx, waterway in enumerate(debug_waterways):
                source_id = waterway['source_id']
                data = waterway['data']
                name = waterway['name']
                match_reason = waterway.get('match_reason', 'Unknown reason')
                
                # Print a compact summary to console (not just log)
                print(f"  {idx+1}. {name} - ID: {source_id}")
                print(f"     Match reason: {match_reason}")
                print(f"     water: {data.get('water', 'Unknown')}")
                print(f"     name:ro: {data.get('name:ro', 'Unknown')}")
                print(f"     Points: {len(data['coordinates'])}")
            
            print("="*80)
            
            # Add all matching waterways to the KML
            print(f"\nAdding {len(debug_waterways)} matching waterways to KML file...")
            
            # Create a special folder for debug waterways
            debug_folder = output_kml.find('./Document/Folder[@id="debug_folder"]')
            if debug_folder is None:
                document = output_kml.find('./Document')
                debug_folder = ET.SubElement(document, 'Folder', id="debug_folder")
                ET.SubElement(debug_folder, 'name').text = "Debug - Matching Waterways"
                ET.SubElement(debug_folder, 'description').text = f"Waterways matching '{DEBUG_FILTER['waterway']}'"
            
            # Add waterways to KML, grouped by name
            flat_index = 0
            for name, segments in waterway_groups.items():
                # Sort segments by length (largest first) for better visualization
                segments.sort(key=lambda x: calculate_segment_length(x['data']['coordinates']), reverse=True)
                
                # Create a subfolder for this waterway if it has multiple segments
                if len(segments) > 1:
                    waterway_folder = ET.SubElement(debug_folder, 'Folder', id=f"waterway_{flat_index}")
                    ET.SubElement(waterway_folder, 'name').text = f"{name} ({len(segments)} segments)"
                    ET.SubElement(waterway_folder, 'description').text = f"All segments of {name}"
                    target_folder = waterway_folder
                else:
                    target_folder = debug_folder
                
                # Add each segment
                for i, segment in enumerate(segments):
                    add_waterway_to_debug_folder(
                        name, 
                        segment['data'],
                        output_kml,
                        f"Segment {i+1} of {len(segments)}",
                        flat_index,
                        target_folder
                    )
                    flat_index += 1
            
            print(f"Added {len(debug_waterways)} segments from {len(waterway_groups)} waterways to KML")
        elif DEBUG_FILTER['enabled'] and DEBUG_FILTER['waterway']:
            print(f"\nWARNING: No waterways found matching '{debug_term}'!")
            print("="*80)
        
        # Track associations for color-coding
        associations = {}
        
        # Load ANPA data to get stream and river relationships and association names
        anpa_data = None
        anpa_habitats_path = 'data/ANPA_habitats_contractate_2025_full.csv'
        if os.path.exists(anpa_habitats_path):
            logging.info(f"Loading ANPA data from {anpa_habitats_path}")
            anpa_data = pd.read_csv(anpa_habitats_path)
            
            # If debug filter is enabled, filter the ANPA data
            if DEBUG_FILTER['enabled']:
                original_count = len(anpa_data)
                filters = []
                
                if DEBUG_FILTER['waterway']:
                    # Create a filter for waterway (case insensitive partial match)
                    waterway_filter = anpa_data['Habitat'].str.contains(
                        DEBUG_FILTER['waterway'], case=False, na=False
                    )
                    filters.append(waterway_filter)
                
                if DEBUG_FILTER['county']:
                    # Create a filter for county (case insensitive match)
                    county_filter = anpa_data['County'].str.contains(
                        DEBUG_FILTER['county'], case=False, na=False
                    )
                    filters.append(county_filter)
                
                # Apply the filters if any were created
                if filters:
                    combined_filter = filters[0]
                    for f in filters[1:]:
                        combined_filter = combined_filter & f
                    
                    anpa_data = anpa_data[combined_filter]
                    
                logging.info(f"DEBUG: Filtered ANPA data from {original_count} to {len(anpa_data)} records")
                
                # Display the filtered records
                if not anpa_data.empty:
                    print("\nDEBUG: Filtered ANPA records:")
                    for _, row in anpa_data.iterrows():
                        habitat = row['Habitat'] if 'Habitat' in row else 'Unknown'
                        county = row['County'] if 'County' in row else 'Unknown'
                        limits = row['Limits'] if 'Limits' in row else ''
                        print(f"  {habitat} ({county}): {limits[:100]}...")
                    print("="*80)
            
            # Create a mapping of habitat names to association names
            logging.info("Creating association name mappings...")
            for _, row in tqdm(anpa_data.iterrows(), desc="Processing ANPA data", total=len(anpa_data)):
                if 'Habitat' in row.index and 'Association_Name' in row.index:
                    habitat_name = str(row['Habitat']).strip() if not pd.isna(row['Habitat']) else ""
                    association_name = row['Association_Name']
                    
                    # If Association_Name is not available, try to get it from Association
                    if pd.isna(association_name) and 'Association' in row.index:
                        association_raw = row['Association']
                        if not pd.isna(association_raw):
                            # Keep the full association string including address
                            association_name = association_raw
                    
                    if not pd.isna(association_name):
                        normalized_habitat = normalize_waterway_name(habitat_name)
                        association_names[normalized_habitat] = association_name
                        
                        # For numeric IDs in the KML file, also store a mapping
                        if isinstance(association_name, str) and association_name.isdigit():
                            association_id = association_name
                            # Look for a better name in the Association column
                            if 'Association' in row.index and not pd.isna(row['Association']):
                                full_association = row['Association']
                                if isinstance(full_association, str):
                                    # Use the full association information
                                    association_names[association_id] = full_association
                        
                        # Also store a mapping using normalized_habitat + "_id"
                        if isinstance(association_name, (int, float)) or (isinstance(association_name, str) and association_name.isdigit()):
                            id_key = f"{normalized_habitat}_id"
                            association_names[id_key] = association_name
        
        # Initialize our output list for streams ending in rivers
        streams_to_rivers = []
        
        # IMPORTANT CHANGE: Start with ANPA data first (left join from ANPA to KML)
        if anpa_data is not None:
            logging.info("Starting with ANPA data to ensure all records have associations...")
            anpa_stream_count = 0  # Counter for streams ending in rivers in ANPA dataset
            city_to_city_count = 0  # Counter for city-to-city segments
            
            for _, row in tqdm(anpa_data.iterrows(), desc="Processing ANPA habitats", total=len(anpa_data)):
                habitat_name = row['Habitat'] if 'Habitat' in row.index else ''
                limit_info = row['Limits'] if 'Limits' in row.index else ''
                county = row['County'] if 'County' in row.index else None
                
                # Extract locations and types from limit info
                start_location, end_location, start_type, end_type, starts_at_spring, _ = utils.extract_locations_and_types(limit_info)
                
                # Get the association name
                association = None
                if 'Association_Name' in row.index and not pd.isna(row['Association_Name']):
                    association = row['Association_Name']
                elif 'Association' in row.index and not pd.isna(row['Association']):
                    association_raw = row['Association']
                    if isinstance(association_raw, str):
                        # Keep the full association info
                        association = association_raw
                
                # If association is a numeric ID, try to resolve it
                if association is not None and (isinstance(association, (int, float)) or (isinstance(association, str) and association.isdigit())):
                    association_id = str(association)
                    if association_id in association_names:
                        association = association_names[association_id]
                
                # Skip records without association
                if association is None or pd.isna(association):
                    log_matching_result(habitat_name, "", start_location, end_location, 
                                      start_type, end_type, county, False, 
                                      notes="No association found")
                    continue
                
                # Track this association for coloring
                if association not in associations:
                    associations[association] = generate_color_for_association(association)
                
                # METHOD 1: Check if this habitat is a stream that starts at spring and ends in a river
                is_stream_to_river, river_name, starts_at_spring = utils.is_stream_ending_in_river(habitat_name, limit_info)
                
                # If we didn't find a river from limit info, try to infer from the name
                if not is_stream_to_river or not river_name:
                    is_stream_to_river_from_name, river_from_name = is_stream_ending_in_river_from_name(habitat_name)
                    if is_stream_to_river_from_name and river_from_name:
                        is_stream_to_river = True
                        river_name = river_from_name
                        # Note: starts_at_spring remains unchanged since we can't determine it from the name
                
                # We want streams that both start at a spring AND end in a river
                if is_stream_to_river and starts_at_spring:
                    anpa_stream_count += 1  # Increment counter
                    
                    # Normalize both names for better matching
                    normalized_stream = normalize_waterway_name(habitat_name)
                    normalized_river = normalize_waterway_name(river_name) if river_name else "Unknown"
                    
                    # Clean up river names with "si afluenții"
                    if normalized_river.lower().endswith("si afluenții") or normalized_river.lower().endswith("și afluenții"):
                        clean_river = re.sub(r'(?i)\s+(?:și|si)\s+afluent(?:ii|ții)$', '', normalized_river).strip()
                        if clean_river:
                            normalized_river = clean_river
                    
                    # Now find this stream in the KML data
                    stream_found = False
                    stream_data = None
                    stream_name = None
                    match_type = ""
                    
                    # First try direct match by name
                    if habitat_name in waterways_by_name:
                        stream_found = True
                        stream_data = waterways_by_name[habitat_name]
                        stream_name = habitat_name
                        match_type = "exact"
                    # Then try normalized name match
                    elif normalized_stream in waterways_by_normalized_name:
                        stream_found = True
                        stream_data = waterways_by_normalized_name[normalized_stream]
                        stream_name = stream_data['name']
                        match_type = "normalized"
                    else:
                        # Try fuzzy matching for stream name
                        for potential_stream, potential_data in waterways_by_name.items():
                            if names_similar(potential_stream, habitat_name, threshold=0.55):
                                stream_found = True
                                stream_data = potential_data
                                stream_name = potential_stream
                                match_type = "fuzzy"
                                break
                    
                    # Log the matching result
                    log_matching_result(habitat_name, river_name, start_location, end_location, 
                                      start_type, end_type, county, stream_found, match_type,
                                      notes=f"Spring to river stream. KML match: {stream_name if stream_found else 'None'}")
                    
                    # If we found a matching stream in KML, add it to our outputs
                    if stream_found:
                        add_stream_to_output(
                            stream_name, stream_data, river_name or "Unknown River",
                            limit_info, output_kml, streams_to_rivers,
                            associations, association, association_names
                        )
                
                # METHOD 2: Check if this is a river segment between two cities
                # If both start and end are cities, process as a city-to-city segment
                if start_type == 'City' and end_type == 'City' and start_location and end_location:
                    # Find the waterway in KML
                    waterway_found = False
                    waterway_data = None
                    match_type = ""
                    waterway_name = ""
                    
                    # Try direct match by name
                    if habitat_name in waterways_by_name:
                        waterway_found = True
                        waterway_data = waterways_by_name[habitat_name]
                        waterway_name = habitat_name
                        match_type = "exact"
                    # Then try normalized name match
                    else:
                        normalized_habitat = normalize_waterway_name(habitat_name)
                        if normalized_habitat in waterways_by_normalized_name:
                            waterway_found = True
                            waterway_data = waterways_by_normalized_name[normalized_habitat]
                            waterway_name = waterway_data['name']
                            match_type = "normalized"
                    
                    # Log the matching result
                    log_matching_result(habitat_name, "", start_location, end_location, 
                                      start_type, end_type, county, waterway_found, match_type,
                                      notes=f"City to city segment. KML match: {waterway_name if waterway_found else 'None'}")
                    
                    # If we found the waterway, process the segment
                    if waterway_found:
                        segment_info = process_waterway_between_cities(
                            start_location, end_location, county, county,
                            waterway_data, association, limit_info,
                            output_kml, streams_to_rivers, associations, association_names
                        )
                        
                        if segment_info:
                            city_to_city_count += 1
                # Check if this is a spring to city segment
                elif start_type == 'Spring' and end_type == 'City' and start_location and end_location:
                    # Find the waterway in KML
                    waterway_found = False
                    waterway_data = None
                    match_type = ""
                    waterway_name = ""
                    
                    # Try direct match by name
                    if habitat_name in waterways_by_name:
                        waterway_found = True
                        waterway_data = waterways_by_name[habitat_name]
                        waterway_name = habitat_name
                        match_type = "exact"
                    # Then try normalized name match
                    else:
                        normalized_habitat = normalize_waterway_name(habitat_name)
                        if normalized_habitat in waterways_by_normalized_name:
                            waterway_found = True
                            waterway_data = waterways_by_normalized_name[normalized_habitat]
                            waterway_name = waterway_data['name']
                            match_type = "normalized"
                    
                    # Log the matching result
                    log_matching_result(habitat_name, "", start_location, end_location, 
                                      start_type, end_type, county, waterway_found, match_type,
                                      notes=f"Spring to city segment. KML match: {waterway_name if waterway_found else 'None'}")
                    
                    # If we found the waterway, process the segment (using the same processing function as city-to-city)
                    if waterway_found:
                        segment_info = process_waterway_between_cities(
                            start_location, end_location, county, county,
                            waterway_data, association, limit_info,
                            output_kml, streams_to_rivers, associations, association_names
                        )
                # Check if this is a city to river segment
                elif start_type == 'City' and (end_type == 'River' or end_type == 'Confluence') and start_location and end_location:
                    # Find the waterway in KML
                    waterway_found = False
                    waterway_data = None
                    match_type = ""
                    waterway_name = ""
                    
                    # Extract river name from confluence/river if available
                    river_name = ""
                    if end_type == 'Confluence':
                        river_name = extract_river_from_confluence(end_location)
                    elif end_type == 'River':
                        river_name = end_location
                    
                    # Try direct match by name
                    if habitat_name in waterways_by_name:
                        waterway_found = True
                        waterway_data = waterways_by_name[habitat_name]
                        waterway_name = habitat_name
                        match_type = "exact"
                    # Then try normalized name match
                    else:
                        normalized_habitat = normalize_waterway_name(habitat_name)
                        if normalized_habitat in waterways_by_normalized_name:
                            waterway_found = True
                            waterway_data = waterways_by_normalized_name[normalized_habitat]
                            waterway_name = waterway_data['name']
                            match_type = "normalized"
                    
                    # Log the matching result
                    log_matching_result(habitat_name, river_name, start_location, end_location, 
                                      start_type, end_type, county, waterway_found, match_type,
                                      notes=f"City to river segment. KML match: {waterway_name if waterway_found else 'None'}")
                    
                    # If we found the waterway, process the segment
                    if waterway_found:
                        segment_info = process_waterway_between_cities(
                            start_location, end_location, county, county,
                            waterway_data, association, limit_info,
                            output_kml, streams_to_rivers, associations, association_names
                        )
                else:
                    # Log non-city-to-city, non-spring-to-river segments
                    if not (is_stream_to_river and starts_at_spring):
                        # Check if this is a Spring-to-Confluence segment which should be classified as spring-to-river
                        if start_type == 'Spring' and end_type == 'Confluence':
                            # This is actually a spring-to-river segment but wasn't detected earlier
                            # Extract potential river name from end location (confluence typically mentions the river)
                            confluence_river = extract_river_from_confluence(end_location)
                            log_matching_result(habitat_name, confluence_river, start_location, end_location, 
                                               start_type, end_type, county, False, 
                                               notes="Spring to river stream (confluence). Not matched in KML.")
                        else:
                            log_matching_result(habitat_name, river_name or "", start_location, end_location, 
                                               start_type, end_type, county, False, 
                                               notes="Not a spring-to-river or city-to-city segment")
            
            logging.info(f"Processed {city_to_city_count} city-to-city waterway segments from ANPA data")
        
        # Now process any KML streams that match ANPA's patterns but weren't already processed
        logging.info("Looking for additional streams in KML data that match ANPA patterns...")
        for name, data in tqdm(waterways_by_name.items(), desc="Processing KML waterways"):
            # Skip if already added
            if any(s.get('stream_name') == name or s.get('waterway_name') == name for s in streams_to_rivers):
                continue
            
            # Only look at streams
            if data.get('waterway_type', '').lower() != 'stream':
                continue
                
            limit_info = waterway_limits.get(name, '')
            
            # Extract locations and types
            start_location, end_location, start_type, end_type, starts_at_spring, _ = utils.extract_locations_and_types(limit_info)
            
            # Check if this is a stream ending in a river
            is_stream_to_river, river_name, starts_at_spring = utils.is_stream_ending_in_river(name, limit_info)
            
            # Try to infer from name if not found in limits
            if not is_stream_to_river or not river_name:
                is_stream_to_river_from_name, river_from_name = is_stream_ending_in_river_from_name(name)
                if is_stream_to_river_from_name and river_from_name:
                    is_stream_to_river = True
                    river_name = river_from_name
                    # Note: starts_at_spring remains unchanged
            
            # Only include streams that both start at a spring AND end in a river
            if is_stream_to_river and starts_at_spring and river_name:
                normalized_stream = normalize_waterway_name(name)
                
                # Try to find a matching habitat in ANPA data for association
                association = None
                
                # Try to get the association name from our mapping
                if normalized_stream in association_names:
                    association = association_names[normalized_stream]
                # Check direct data association
                elif data.get('association') and str(data.get('association')) in association_names:
                    association = association_names[str(data.get('association'))]
                
                # Skip if we can't find an association
                if not association:
                    log_matching_result(name, river_name, start_location, end_location, 
                                      start_type, end_type, None, False, 
                                      notes="KML stream - No association found")
                    continue
                
                # Ensure we have a color for this association
                if association not in associations:
                    associations[association] = generate_color_for_association(association)
                
                log_matching_result(name, river_name, start_location, end_location, 
                                  start_type, end_type, None, True, "kml_only",
                                  notes="KML stream - with association")
                
                add_stream_to_output(
                    name, data, river_name, limit_info,
                    output_kml, streams_to_rivers, associations,
                    association, association_names
                )
        
        # Checking how many streams have been found
        logging.info(f"Found {len(streams_to_rivers)} streams starting at springs and ending in rivers with associations")
        logging.info(f"Used {len(associations)} different colors for associations")
        
        # Print additional statistics about the data
        logging.info(f"ANPA data contains {anpa_stream_count} streams starting at springs and ending in rivers")
        logging.info(f"Total water records in KML data: {len(waterways_by_name)}")
        logging.info(f"Total water records in ANPA data: {len(anpa_data) if anpa_data is not None else 0}")
        
        # Create style map section with colors for each association
        logging.info("Creating style maps for KML...")
        add_style_maps_to_kml(output_kml, associations)
        
        # Filter out streams with unknown associations if specified
        include_only_known = True  # Set to True to include only streams with known associations
        if include_only_known:
            filtered_streams = [s for s in streams_to_rivers if 'Unknown Association' not in s['association']]
            logging.info(f"Filtered to {len(filtered_streams)} streams with known associations for output")
            streams_to_rivers = filtered_streams
        
        # Write the output KML file
        logging.info(f"Writing output KML to {output_kml_path}...")
        tree = ET.ElementTree(output_kml)
        tree.write(output_kml_path, encoding='utf-8', xml_declaration=True)
        
        logging.info(f"Output KML saved to {output_kml_path}")
        
        # Also save the data as CSV for easier analysis
        csv_path = output_kml_path.replace('.kml', '.csv')
        pd.DataFrame(streams_to_rivers).to_csv(csv_path, index=False)
        logging.info(f"Output CSV saved to {csv_path}")
        
        # Summarize the processing results including matching statistics
        print("\n" + "="*80)
        print(f"KML PROCESSING SUMMARY")
        print("="*80)
        print(f"Found {len(streams_to_rivers)} streams/segments with valid associations")
        print(f"Used {len(associations)} different colors for fishing associations")
        print(f"Total water records in KML data: {len(waterways_by_name)}")
        print(f"Total water records in ANPA data: {len(anpa_data) if anpa_data is not None else 0}")
        print("="*80)

        # Close the matching log file and print statistics
        finalize_matching_log()
        
        return streams_to_rivers
        
    except Exception as e:
        logging.error(f"Error processing KML: {str(e)}")
        # Make sure to close the matching log file even if there's an error
        finalize_matching_log()
        raise

def add_style_maps_to_kml(kml_root, associations):
    """
    Add style maps for each association to the KML document
    
    Args:
        kml_root: KML root element
        associations: Dictionary mapping association names to colors
    """
    document = kml_root.find('./Document')
    
    # Add a style for each association
    for association, color in tqdm(associations.items(), desc="Creating KML styles"):
        # Create a safe ID from the association name
        assoc_id = re.sub(r'[^a-zA-Z0-9]', '', str(association))
        if not assoc_id:
            assoc_id = "unknown"
        
        # Normal style
        style = ET.SubElement(document, 'Style', id=f"style_{assoc_id}")
        line_style = ET.SubElement(style, 'LineStyle')
        ET.SubElement(line_style, 'color').text = color
        ET.SubElement(line_style, 'width').text = '4'
        
        # Highlighted style (slightly wider)
        highlight_style = ET.SubElement(document, 'Style', id=f"style_{assoc_id}_highlight")
        line_style = ET.SubElement(highlight_style, 'LineStyle')
        ET.SubElement(line_style, 'color').text = color
        ET.SubElement(line_style, 'width').text = '6'
        
        # Label style (make association name visible)
        ET.SubElement(style, 'BalloonStyle')
        label_style = ET.SubElement(style, 'LabelStyle')
        ET.SubElement(label_style, 'scale').text = '0.8'
        
        # Style map to connect normal and highlighted styles
        style_map = ET.SubElement(document, 'StyleMap', id=f"stylemap_{assoc_id}")
        normal_pair = ET.SubElement(style_map, 'Pair')
        ET.SubElement(normal_pair, 'key').text = 'normal'
        ET.SubElement(normal_pair, 'styleUrl').text = f"#style_{assoc_id}"
        highlight_pair = ET.SubElement(style_map, 'Pair')
        ET.SubElement(highlight_pair, 'key').text = 'highlight'
        ET.SubElement(highlight_pair, 'styleUrl').text = f"#style_{assoc_id}_highlight"

def add_stream_to_output(stream_name, stream_data, river_name, limit_info, output_kml, streams_to_rivers, associations, association=None, association_names=None):
    """Helper function to add a stream to the output KML and streams list"""
    logging.info(f"Found stream {stream_name} ending in river {river_name}")
    
    # Extract spring information
    _, _, start_type, _, starts_at_spring, _ = utils.extract_locations_and_types(limit_info)
    is_starting_at_spring = starts_at_spring or start_type == 'Spring'
    
    # If no association provided, use the one from stream_data or default
    if association is None:
        association = stream_data.get('association', 'Unknown Association')
    
    # Try to resolve numeric IDs to proper association names
    normalized_stream = normalize_waterway_name(stream_name)
    
    # Make sure association is not a numeric ID or at least has a proper label
    if association_names and (isinstance(association, (int, float)) or (isinstance(association, str) and association.isdigit())):
        # First check if we have a direct mapping for this ID
        if str(association) in association_names:
            association = association_names[str(association)]
        # Then check if we have a mapping for this stream's ID
        elif f"{normalized_stream}_id" in association_names:
            id_value = association_names[f"{normalized_stream}_id"]
            if str(id_value) in association_names:
                association = association_names[str(id_value)]
        # If still numeric, just label it as an association
        if isinstance(association, (int, float)) or (isinstance(association, str) and association.isdigit()):
            association = f"Association {association}"
    
    # Use the raw association name as is - no standardization
    display_association = association
    
    # Create a safe style ID from the association name
    association_id = re.sub(r'[^a-zA-Z0-9]', '', str(association))
    if not association_id:
        association_id = "unknown"
    
    # Ensure we have a color for this association
    if association not in associations:
        associations[association] = generate_color_for_association(association)
    
    # Create description with spring info
    spring_text = "Starts at a spring and " if is_starting_at_spring else ""
    description = f"{spring_text}Stream that ends at river {river_name}.<br/><b>Association:</b> {display_association}<br/><b>Limit info:</b> {limit_info}"
    
    # Add this stream to the KML
    placemark_xml = create_placemark_xml(
        name=stream_name,
        description=description,
        coordinates=stream_data['coordinates'],
        association=display_association,
        association_id=association_id,
        style_color=associations[association]
    )
    
    # Find the folder to add the placemark to
    springs_folder = output_kml.find('./Document/Folder[@id="springs_to_rivers_folder"]')
    if springs_folder is None:
        logging.warning("Could not find Springs to Rivers folder - using first folder")
        springs_folder = output_kml.find('./Document/Folder')
    
    if springs_folder is not None:
        springs_folder.append(placemark_xml)
    else:
        logging.error("Could not find any folder to add stream placemark")
    
    streams_to_rivers.append({
        'stream_name': stream_name,
        'river_name': river_name,
        'start_coordinates': stream_data['coordinates'][0],
        'end_coordinates': stream_data['coordinates'][-1],
        'association': display_association,
        'limit_info': limit_info,
        'starts_at_spring': is_starting_at_spring,
        'segment_type': 'spring_to_river'
    })

def extract_namespace(root):
    """Extract the namespace from the root element."""
    match = re.match(r'\{(.*?)\}', root.tag)
    return match.group(1) if match else ''

def find_all_placemarks(root, ns):
    """Find all Placemark elements in the KML."""
    ns_prefix = '{' + ns + '}' if ns else ''
    return root.findall(f'.//{ns_prefix}Placemark')

def extract_placemark_data(placemark, ns):
    """
    Extract data from a Placemark element.
    
    Returns:
        dict: A dictionary containing the placemark data, or None if it's not a waterway
    """
    ns_prefix = '{' + ns + '}' if ns else ''
    
    data = {}
    
    # Extract name (if available)
    name_elements = placemark.findall(f'./{ns_prefix}name')
    if name_elements:
        # Use the second name element if it's "river", "stream", etc.
        # Otherwise use the first name element
        if len(name_elements) > 1 and name_elements[1].text in ['river', 'stream', 'canal', 'drain']:
            data['name'] = name_elements[0].text if name_elements[0].text else "Unnamed"
            data['waterway_type'] = name_elements[1].text
        else:
            data['name'] = name_elements[0].text if name_elements[0].text else "Unnamed"
            data['waterway_type'] = 'unknown'
    
    # Extract description
    description = placemark.find(f'./{ns_prefix}description')
    if description is not None and description.text:
        data['description'] = description.text
    
    # Extract all attributes from SchemaData
    schema_data = placemark.find(f'.//{ns_prefix}SchemaData')
    if schema_data is not None:
        for simple_data in schema_data.findall(f'.//{ns_prefix}SimpleData'):
            if 'name' in simple_data.attrib and simple_data.text:
                attr_name = simple_data.attrib['name']
                data[attr_name] = simple_data.text

                # Also store the attribute with its full prefix for exact matching
                if ':' in attr_name:
                    data['full_prefix_' + attr_name.replace(':', '_')] = simple_data.text
    
    # Extract coordinates
    line_string = placemark.find(f'.//{ns_prefix}LineString')
    if line_string is not None:
        coords_element = line_string.find(f'.//{ns_prefix}coordinates')
        if coords_element is not None and coords_element.text:
            # Parse coordinates
            coords_text = coords_element.text.strip()
            coords_list = []
            
            for coord in coords_text.split():
                parts = coord.split(',')
                if len(parts) >= 2:
                    try:
                        lon, lat = float(parts[0]), float(parts[1])
                        coords_list.append((lon, lat))
                    except ValueError:
                        continue
            
            if coords_list:
                data['coordinates'] = coords_list
    
    # Only return data if it has coordinates
    return data if 'coordinates' in data else None

def create_output_kml_structure():
    """Create the basic structure for the output KML file."""
    kml = ET.Element('kml', xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, 'Document', id="root_doc")
    ET.SubElement(document, 'name').text = "Romanian Waterways - Fishing Segments"
    
    # Add a description
    ET.SubElement(document, 'description').text = """
    This KML file contains two types of fishing segments in Romania:
    1. Streams that start at springs and end in rivers
    2. Waterway segments between cities
    
    The waterways are color-coded by the fishing associations that manage them.
    Click on a waterway to see more information.
    """
    
    # Create folder for debug data
    folder_debug = ET.SubElement(document, 'Folder', id="debug_folder")
    ET.SubElement(folder_debug, 'name').text = "Debug - Full Waterways"
    ET.SubElement(folder_debug, 'description').text = "Full waterways for debugging"
    
    # Create folder for streams
    folder_springs = ET.SubElement(document, 'Folder', id="springs_to_rivers_folder")
    ET.SubElement(folder_springs, 'name').text = "Springs to Rivers"
    ET.SubElement(folder_springs, 'description').text = "Streams that start at springs and end in rivers, color-coded by fishing association"
    
    # Create folder for city-to-city segments
    folder_cities = ET.SubElement(document, 'Folder', id="city_to_city_folder")
    ET.SubElement(folder_cities, 'name').text = "City to City Segments"
    ET.SubElement(folder_cities, 'description').text = "Waterway segments between cities, color-coded by fishing association"
    
    return kml

def create_placemark_xml(name, description, coordinates, association, association_id, style_color):
    """
    Create a Placemark XML element.
    
    Args:
        name (str): Name of the waterway
        description (str): Description of the waterway
        coordinates (list): List of coordinates [(lon, lat), ...]
        association (str): Association managing the waterway
        association_id (str): Safe ID for the association for styling
        style_color (str): KML color code for the association
        
    Returns:
        ET.Element: Placemark XML element
    """
    placemark = ET.Element('Placemark')
    
    ET.SubElement(placemark, 'name').text = name
    ET.SubElement(placemark, 'description').text = description
    
    # Add style reference
    ET.SubElement(placemark, 'styleUrl').text = f"#stylemap_{association_id}"
    
    # Add extended data
    extended_data = ET.SubElement(placemark, 'ExtendedData')
    data_assoc = ET.SubElement(extended_data, 'Data', name="Association")
    ET.SubElement(data_assoc, 'displayName').text = "Association"
    ET.SubElement(data_assoc, 'value').text = str(association)
    
    # Add coordinates
    line_string = ET.SubElement(placemark, 'LineString')
    coords_text = ' '.join([f"{lon},{lat},0" for lon, lat in coordinates])
    ET.SubElement(line_string, 'coordinates').text = coords_text
    
    return placemark

def names_similar(name1, name2, threshold=0.6):
    """
    Check if two names are similar using fuzzy matching
    
    Args:
        name1 (str): First name
        name2 (str): Second name
        threshold (float): Similarity threshold (0-1)
        
    Returns:
        bool: True if names are similar, False otherwise
    """
    if not name1 or not name2:
        return False
    
    # Normalize both names
    name1 = normalize_waterway_name(name1).lower()
    name2 = normalize_waterway_name(name2).lower()
    
    # Direct match after normalization
    if name1 == name2:
        return True
    
    # Remove common words like "mare", "mic", "de sus", "de jos" for comparison
    common_words = ['mare', 'mic', 'de sus', 'de jos', 'superior', 'inferior']
    for word in common_words:
        name1 = name1.replace(f" {word}", "")
        name2 = name2.replace(f" {word}", "")
    
    # After removing common words, check again for direct match
    if name1 == name2:
        return True
    
    # Check if one is a substring of the other (must be at least 3 chars to avoid false positives)
    if (len(name1) >= 3 and name1 in name2) or (len(name2) >= 3 and name2 in name1):
        return True
    
    # For multi-word names, compare the first parts
    # (e.g. "Mureș Barcău" and "Mureș" should match)
    name1_parts = name1.split()
    name2_parts = name2.split()
    
    if len(name1_parts) > 1 and len(name2_parts) > 0:
        if name1_parts[0] == name2_parts[0]:
            return True
    
    if len(name2_parts) > 1 and len(name1_parts) > 0:
        if name2_parts[0] == name1_parts[0]:
            return True
    
    # Compare similarity ratio using difflib
    import difflib
    similarity = difflib.SequenceMatcher(None, name1, name2).ratio()
    
    # Boost similarity for names that share at least one word
    common_words = set(name1_parts) & set(name2_parts)
    if common_words:
        # Boost proportional to the number of common words and their length
        boost_factor = sum(len(word) for word in common_words) / 10
        similarity_boost = min(0.3, 0.1 * len(common_words) + boost_factor)
        similarity += similarity_boost
    
    return similarity >= threshold

def is_stream_ending_in_river_from_name(stream_name):
    """
    Check if a stream name contains information about which river it ends in.
    For example: 'Pârâul Olt' might be a stream ending in river 'Olt'.
    
    Args:
        stream_name (str): Stream name
        
    Returns:
        tuple: (is_stream_to_river, river_name)
    """
    # First normalize the name
    normalized_name = normalize_waterway_name(stream_name)
    
    # Split the name into parts
    parts = normalized_name.split()
    
    # If the name has at least two parts, the last part might be the river name
    if len(parts) >= 2:
        potential_river = parts[-1]
        # If the potential river name is long enough, it might be a valid river name
        if len(potential_river) >= 3:
            # Return the potential river name
            return True, potential_river
    
    # Check for common patterns in Romanian stream names
    common_endings = ["ului", "ei", "urilor", "elor"]
    for ending in common_endings:
        if normalized_name.endswith(ending) and len(normalized_name) > len(ending) + 3:
            # Extract the root name, which might be derived from the river name
            root_name = normalized_name[:-len(ending)]
            return True, root_name
    
    return False, None

def extract_river_from_confluence(confluence_text):
    """
    Extract river name from confluence description
    Example: "conf. râul Târgului" -> "râul Târgului"
    
    Args:
        confluence_text (str): Confluence description text
        
    Returns:
        str: Extracted river name or empty string
    """
    if not confluence_text:
        return ""
    
    # Common patterns for confluence descriptions in Romanian
    patterns = [
        r'conf(?:\.|luență)\s+(?:cu\s+)?(?:(?:râul|r\.|raul|pârâul|paraul|p\.)\s+)?(.+?)(?:\s+la|\s+în|\s+la|\s+pe|\s*$)',
        r'(?:cu|în|la)\s+(?:(?:râul|r\.|raul|pârâul|paraul|p\.)\s+)?(.+?)(?:\s+la|\s+în|\s*$)',
        r'(?:râul|r\.|raul|pârâul|paraul|p\.)\s+(.+?)(?:\s+la|\s+în|\s*$)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, confluence_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no match with patterns, return the whole text as a fallback
    # Remove common prefixes
    cleaned = re.sub(r'^conf(?:\.|luență)\s+(?:cu\s+)?', '', confluence_text, flags=re.IGNORECASE)
    return cleaned.strip()

def add_waterway_to_debug_folder(name, data, output_kml, match_info="", index=0, target_folder=None):
    """
    Add a waterway to the debug folder in the KML
    
    Args:
        name (str): Name of the waterway
        data (dict): Waterway data
        output_kml: KML document to add the waterway to
        match_info (str): Information about how this waterway matched the filter
        index (int): Index for generating different colors
        target_folder: Specific folder to add the waterway to (or None for default)
    """
    if 'coordinates' not in data or not data['coordinates']:
        logging.warning(f"DEBUG: Cannot add waterway {name} - no coordinates")
        return
    
    # Create a folder for debugging if it doesn't exist and no target folder specified
    if target_folder is None:
        debug_folder = output_kml.find('./Document/Folder[@id="debug_folder"]')
        if debug_folder is None:
            document = output_kml.find('./Document')
            debug_folder = ET.SubElement(document, 'Folder', id="debug_folder")
            ET.SubElement(debug_folder, 'name').text = "Debug - Matching Waterways"
            ET.SubElement(debug_folder, 'description').text = "Waterways matching the debug filter"
        target_folder = debug_folder
    
    # Calculate segment length in kilometers
    segment_length_km = calculate_segment_length(data['coordinates'])
    
    # Get the first and last coordinates for the segment to show its bounds
    start_coords = data['coordinates'][0]
    end_coords = data['coordinates'][-1]
    
    # Generate a color based on the index - to distinguish multiple waterways
    # Cycle through some distinct colors: red, blue, green, yellow, purple, cyan
    colors = ["ff0000ff", "ffff0000", "ff00ff00", "ff00ffff", "ffff00ff", "ffffff00"]
    color = colors[index % len(colors)]
    
    # Extract additional attributes - handle prefixed attributes better
    source = "Unknown"
    name_ro = "Unknown"
    water = "Unknown"
    
    # Look for attributes with various patterns
    for key, value in data.items():
        if key == 'source' or key.endswith(':source') or key.endswith('_source'):
            source = value
        if key == 'name:ro' or key.endswith(':name:ro') or key.endswith('_name_ro'):
            name_ro = value
        if key == 'water' or key.endswith(':water') or key.endswith('_water'):
            water = value
    
    # Build a detailed description with all attributes
    description = (f"Waterway matching filter - {name}\n"
                  f"Match type: {match_info}\n"
                  f"Segment length: {segment_length_km:.2f} km\n"
                  f"Coordinates: {len(data['coordinates'])} points\n"
                  f"Start: {start_coords[1]:.4f}, {start_coords[0]:.4f}\n"
                  f"End: {end_coords[1]:.4f}, {end_coords[0]:.4f}\n\n"
                  f"KML Attributes:\n"
                  f"source: {source}\n"
                  f"name:ro: {name_ro}\n"
                  f"water: {water}\n")
    
    # Add any other attributes found in the data
    extra_attrs = []
    for key, value in data.items():
        if key not in ['name', 'coordinates', 'description', 'source', 'name:ro', 'water', 'waterway_type']:
            extra_attrs.append(f"{key}: {value}")
    
    if extra_attrs:
        description += "\nOther attributes:\n" + "\n".join(extra_attrs)
    
    # Add the waterway to the KML
    placemark_xml = create_placemark_xml(
        name=f"{name} - {match_info} [source:{source}]",
        description=description,
        coordinates=data['coordinates'],
        association="DEBUG",
        association_id=f"debug_{index}",
        style_color=color
    )
    
    target_folder.append(placemark_xml)
    logging.info(f"DEBUG: Added waterway {name} to KML output - "
                 f"Length: {segment_length_km:.2f} km, Points: {len(data['coordinates'])}, "
                 f"source: {source}, name:ro: {name_ro}, water: {water}")

def calculate_segment_length(coordinates):
    """
    Calculate the length of a segment in kilometers
    
    Args:
        coordinates (list): List of (lon, lat) coordinates
        
    Returns:
        float: Length in kilometers
    """
    import math
    
    def haversine(lon1, lat1, lon2, lat2):
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    total_length = 0
    for i in range(len(coordinates) - 1):
        lon1, lat1 = coordinates[i]
        lon2, lat2 = coordinates[i + 1]
        total_length += haversine(lon1, lat1, lon2, lat2)
    
    return total_length

def normalize_special_chars(text):
    """
    Normalize special characters to their ASCII equivalents
    For example, ș -> s, ă -> a, î -> i
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # Specific Romanian character mappings
    romanian_chars = {
        'ș': 's', 'ş': 's', 'Ș': 'S', 'Ş': 'S',
        'ț': 't', 'ţ': 't', 'Ț': 'T', 'Ţ': 'T',
        'ă': 'a', 'Ă': 'A',
        'â': 'a', 'Â': 'A',
        'î': 'i', 'Î': 'I'
    }
    
    # First apply our specific mapping for Romanian
    for char, replacement in romanian_chars.items():
        text = text.replace(char, replacement)
    
    # Then use unicodedata for any remaining characters
    # This will convert all accented characters to their closest ASCII equivalent
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    return text

def is_exact_word_match(search_term, text):
    """
    Check if search_term is an exact word within text (not just a substring of a larger word)
    For example, "mures" should match "râul mures" but not "muresel"
    
    Args:
        search_term (str): The term to search for
        text (str): The text to search in
        
    Returns:
        bool: True if search_term is an exact word match, False otherwise
    """
    # Add word boundaries to the search term
    # This will ensure we match 'mures' in 'raul mures' but not in 'muresel'
    if not search_term or not text:
        return False
        
    # Convert both to lowercase for case-insensitive matching
    search_term = search_term.lower()
    text = text.lower()
    
    # Add spaces around the text for boundary checking
    padded_text = f" {text} "
    
    # Check if the search term exists as a whole word
    return f" {search_term} " in padded_text or \
           padded_text.startswith(f"{search_term} ") or \
           padded_text.endswith(f" {search_term}") or \
           search_term == text

def fetch_waterway_from_overpass(bounds, waterway_name=None):
    """
    Fetch waterway data from Overpass API within specified bounds
    
    Args:
        bounds (tuple): (south, west, north, east) coordinates
        waterway_name (str, optional): Name of the waterway to filter by
        
    Returns:
        dict: JSON response from Overpass API
    """
    south, west, north, east = bounds
    
    if not waterway_name:
        logging.warning("No waterway name provided, skipping search")
        return {"elements": []}
    
    # First normalize the name for better matching
    normalized_name = normalize_waterway_name(waterway_name)
    
    # Create a list of name variations to try
    name_variations = [normalized_name]
    
    # Remove prefixes if they exist
    prefixes = ['paraul', 'paraului', 'raul', 'raului', 'valea', 'vaii', 
                'pârâul', 'pârâului', 'râul', 'râului', 'parau', 'rau', 'vale']
    
    # Extract base name without prefix
    base_name = normalized_name
    for prefix in prefixes:
        if normalized_name.lower().startswith(prefix + " "):
            base_name = normalized_name[len(prefix)+1:]
            name_variations.append(base_name)
            break
    
    # Add ASCII (without diacritics) versions
    ascii_variations = []
    for name in name_variations:
        ascii_name = normalize_special_chars(name)
        if ascii_name != name:
            ascii_variations.append(ascii_name)
    name_variations.extend(ascii_variations)
    
    # Add versions with common first words removed
    # For example "Pârâul Timișana" -> "Timișana"
    base_words = []
    for name in list(name_variations):
        parts = name.split()
        if len(parts) > 1:
            base_word = parts[-1]
            if len(base_word) >= 3:  # Only keep meaningful words
                base_words.append(base_word)
    name_variations.extend(base_words)
    
    # Remove duplicates and empty strings
    name_variations = list(set(var for var in name_variations if var))
    
    # Log all variations we're trying
    logging.info(f"Searching for waterway with these name variations: {name_variations}")
    
    # Create an Overpass query that tries all name variations
    query_parts = []
    for name_var in name_variations:
        # Escape special characters for regex
        escaped_name = re.escape(name_var)
        
        # Add queries for both exact and prefix-based matching
        query_parts.append(f'way["waterway"]["name"~"^{escaped_name}$",i]({south},{west},{north},{east});')
        query_parts.append(f'way["waterway"]["name:ro"~"^{escaped_name}$",i]({south},{west},{north},{east});')
        
        # Also try as part of a name (with word boundaries)
        # Use string concatenation to avoid backslash issues in f-strings
        query_parts.append('way["waterway"]["name"~"\\b' + escaped_name + '\\b",i](' + f'{south},{west},{north},{east});')
        query_parts.append('way["waterway"]["name:ro"~"\\b' + escaped_name + '\\b",i](' + f'{south},{west},{north},{east});')
        
        # For shorter names, we also try with common prefixes
        if len(name_var) >= 4:
            for prefix in ["raul", "paraul", "valea", "râul", "pârâul"]:
                prefixed_name = f"{prefix} {escaped_name}"
                query_parts.append(f'way["waterway"]["name"~"^{prefixed_name}$",i]({south},{west},{north},{east});')
                query_parts.append(f'way["waterway"]["name:ro"~"^{prefixed_name}$",i]({south},{west},{north},{east});')
    
    # Construct the full query
    query_parts_joined = "\n      ".join(query_parts)
    query = """
    [out:json][timeout:30];
    (
      """ + query_parts_joined + """
    );
    out body geom;
    """
    
    # Send request to Overpass API
    overpass_url = "https://overpass-api.de/api/interpreter"
    try:
        logging.info(f"Sending Overpass query for waterway: {waterway_name}")
        response = requests.post(overpass_url, data={"data": query})
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Log the size of the response for debugging
        json_response = response.json()
        element_count = len(json_response.get("elements", []))
        logging.info(f"Received {element_count} elements from Overpass API")
        
        return json_response
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching from Overpass API: {str(e)}")
        return {"elements": []}

def test_overpass_with_random_segments(input_csv, output_kml_path, num_samples=5):
    """
    Test Overpass API with random city-to-city segments from ANPA dataset
    
    Args:
        input_csv (str): Path to ANPA CSV file
        output_kml_path (str): Path to output KML file
        num_samples (int): Number of random samples to test
    """
    logging.info(f"Testing Overpass API with {num_samples} random city-to-city segments")
    
    # Read ANPA dataset
    try:
        data = pd.read_csv(input_csv)
    except Exception as e:
        logging.error(f"Error reading CSV file: {str(e)}")
        return
    
    # Create output KML structure
    output_kml = create_output_kml_structure()
    
    # Find habitat segments
    segments = []
    for _, row in data.iterrows():
        limit_info = row.get('Limits', '')
        if not isinstance(limit_info, str):
            continue
            
        # Extract locations and types
        start_location, end_location, start_type, end_type, starts_at_spring, _ = utils.extract_locations_and_types(limit_info)
        
        # Get basic segment info
        county = row.get('County', None)
        habitat_name = row.get('Habitat', 'Unknown')
        
        # Only include records with a valid habitat name
        if pd.isna(habitat_name) or habitat_name == 'Unknown':
            continue
        
        # Get coordinates for start and end locations if they're cities
        start_lat, start_lon = None, None
        end_lat, end_lon = None, None
        
        if start_type == 'City':
            start_lat, start_lon = utils.get_lat_lon(start_location, county)
        
        if end_type == 'City':
            end_lat, end_lon = utils.get_lat_lon(end_location, county)
        
        # Only include segments where we have coordinates for at least one endpoint
        if (start_lat and start_lon) or (end_lat and end_lon):
            # If we're missing one endpoint, use the other with a buffer
            if not start_lat or not start_lon:
                start_lat, start_lon = end_lat, end_lon
            if not end_lat or not end_lon:
                end_lat, end_lon = start_lat, start_lon
                
            segments.append({
                'habitat_name': habitat_name,
                'start_location': start_location,
                'end_location': end_location,
                'start_type': start_type,
                'end_type': end_type,
                'start_coords': (start_lat, start_lon),
                'end_coords': (end_lat, end_lon),
                'county': county,
                'limits': limit_info,
                'starts_at_spring': starts_at_spring
            })
    
    logging.info(f"Found {len(segments)} habitat segments with coordinates")
    
    # Select random samples
    if len(segments) > num_samples:
        samples = random.sample(segments, num_samples)
    else:
        samples = segments
    
    # Process each sample
    for i, segment in enumerate(samples):
        habitat_name = segment['habitat_name']
        start_location = segment['start_location']
        end_location = segment['end_location']
        start_lat, start_lon = segment['start_coords']
        end_lat, end_lon = segment['end_coords']
        county = segment['county']
        start_type = segment['start_type']
        end_type = segment['end_type']
        
        logging.info(f"\nSample {i+1}: {habitat_name}")
        logging.info(f"  From: {start_location} ({start_type}: {start_lat}, {start_lon})")
        logging.info(f"  To: {end_location} ({end_type}: {end_lat}, {end_lon})")
        logging.info(f"  County: {county}")
        
        # Create a bounding box around the start and end points (add a buffer)
        buffer = 0.1  # about 10km buffer
        south = min(start_lat, end_lat) - buffer
        north = max(start_lat, end_lat) + buffer
        west = min(start_lon, end_lon) - buffer
        east = max(start_lon, end_lon) + buffer
        
        # Fetch waterways within the bounds - search only for the specific river
        logging.info(f"  Fetching waterway: {habitat_name} within bounds: ({south}, {west}, {north}, {east})")
        
        # Create a specific folder in the KML for this sample
        sample_folder = ET.SubElement(output_kml.find('./Document'), 'Folder', id=f"sample_{i+1}")
        ET.SubElement(sample_folder, 'name').text = f"Sample {i+1}: {habitat_name}"
        ET.SubElement(sample_folder, 'description').text = f"Start: {start_location} ({start_type})\nEnd: {end_location} ({end_type})\nCounty: {county}"
        
        # Add markers for start and end points
        add_marker_to_kml(
            sample_folder, 
            f"Start: {start_location}", 
            f"Type: {start_type}", 
            start_lat, start_lon
        )
        
        add_marker_to_kml(
            sample_folder, 
            f"End: {end_location}", 
            f"Type: {end_type}", 
            end_lat, end_lon
        )
        
        # Try with the habitat name only - no fallback to all waterways
        response = fetch_waterway_from_overpass((south, west, north, east), habitat_name)
        
        # Process the response
        waterway_coords = process_overpass_response(
            response, 
            output_kml, 
            f"{habitat_name}: {start_location} to {end_location}",
            sample_folder  # Pass the sample folder to add waterways to it
        )
        
        if waterway_coords:
            logging.info(f"  Successfully added OSM waterway data to KML")
            
            # Add the bounding box to the KML for reference
            add_bounding_box_to_kml(
                sample_folder,
                south, west, north, east,
                f"Search Area for {habitat_name}"
            )
        else:
            logging.warning(f"  No OSM waterway data found for this segment with specific name search")
            logging.info(f"  Trying broader search approach with terms extracted from: '{habitat_name}'")
            
            # Create a subfolder for the search results
            search_folder = ET.SubElement(sample_folder, 'Folder', id=f"search_folder_{i+1}")
            ET.SubElement(search_folder, 'name').text = f"Search results for '{habitat_name}'"
            ET.SubElement(search_folder, 'description').text = f"Waterways matching search terms extracted from '{habitat_name}'"
            
            # Fetch waterways using search terms from habitat name
            search_response = fetch_waterways_by_search_term((south, west, north, east), habitat_name)
            
            # Process the search response
            search_coords = process_overpass_response(
                search_response,
                output_kml,
                f"Search results for '{habitat_name}'",
                search_folder
            )
            
            if search_coords:
                logging.info(f"  Successfully found {len(search_response.get('elements', []))} waterways matching search terms")
                
                # Add a note about the search results
                note_placemark = ET.SubElement(sample_folder, 'Placemark')
                ET.SubElement(note_placemark, 'name').text = f"Search results for '{habitat_name}'"
                ET.SubElement(note_placemark, 'description').text = f"Found {len(search_response.get('elements', []))} waterways matching search terms extracted from '{habitat_name}'"
            else:
                logging.warning(f"  No waterways found matching search terms")
                
                # Add a note to the KML
                note_placemark = ET.SubElement(sample_folder, 'Placemark')
                ET.SubElement(note_placemark, 'name').text = "No waterway found"
                ET.SubElement(note_placemark, 'description').text = f"No waterway data found for '{habitat_name}' using either exact or search term approaches"
            
            # Add the bounding box to the KML to show the search area
            add_bounding_box_to_kml(
                sample_folder,
                south, west, north, east,
                f"Search Area for '{habitat_name}'"
            )
        
        # Add a small delay to avoid overloading the API
        time.sleep(1)
    
    # Write the output KML file
    tree = ET.ElementTree(output_kml)
    tree.write(output_kml_path, encoding='utf-8', xml_declaration=True)
    logging.info(f"\nOutput KML saved to {output_kml_path}")

def process_overpass_response(response, output_kml, name="Unknown Waterway", target_folder=None):
    """
    Process Overpass API response and add waterways to KML
    
    Args:
        response (dict): JSON response from Overpass API
        output_kml: KML document to add waterways to
        name (str): Name to use for the waterway
        target_folder: Optional specific folder to add waterways to
        
    Returns:
        list: List of processed waterway coordinates
    """
    if not response or "elements" not in response:
        logging.warning(f"Invalid response from Overpass API")
        return []
    
    elements = response["elements"]
    if not elements:
        logging.warning(f"No waterways found in Overpass API response")
        return []
    
    logging.info(f"Found {len(elements)} waterway elements from Overpass API")
    
    # Use the target folder if provided, otherwise find or create the Overpass folder
    if target_folder is None:
        overpass_folder = output_kml.find('./Document/Folder[@id="overpass_folder"]')
        if overpass_folder is None:
            document = output_kml.find('./Document')
            overpass_folder = ET.SubElement(document, 'Folder', id="overpass_folder")
            ET.SubElement(overpass_folder, 'name').text = "Overpass API Waterways"
            ET.SubElement(overpass_folder, 'description').text = "Waterways fetched from OpenStreetMap via Overpass API"
        target_folder = overpass_folder
    
    all_waterway_coords = []
    
    # Process each waterway element
    for i, element in enumerate(elements):
        if element.get("type") != "way" or "geometry" not in element:
            continue
        
        # Extract waterway details
        waterway_id = element.get("id", "unknown")
        waterway_name = element.get("tags", {}).get("name", name)
        waterway_type = element.get("tags", {}).get("waterway", "unknown")
        
        # Extract coordinates
        geometry = element.get("geometry", [])
        coordinates = []
        for point in geometry:
            if "lat" in point and "lon" in point:
                coordinates.append((point["lon"], point["lat"]))
        
        if not coordinates:
            continue
            
        all_waterway_coords.extend(coordinates)
        
        # Add this waterway to the KML
        waterway_length = calculate_segment_length(coordinates)
        
        description = (
            f"Waterway from OpenStreetMap\n"
            f"ID: {waterway_id}\n"
            f"Name: {waterway_name}\n"
            f"Type: {waterway_type}\n"
            f"Length: {waterway_length:.2f} km\n"
            f"Points: {len(coordinates)}\n\n"
            f"Tags:\n"
        )
        
        # Add all tags to the description
        for key, value in element.get("tags", {}).items():
            description += f"{key}: {value}\n"
        
        placemark_xml = create_placemark_xml(
            name=f"OSM: {waterway_name} ({waterway_type})",
            description=description,
            coordinates=coordinates,
            association="OpenStreetMap",
            association_id=f"osm_{waterway_id}",
            style_color="ff00aaff"  # Yellow color for OSM data
        )
        
        target_folder.append(placemark_xml)
        logging.info(f"Added OSM waterway {waterway_name} (ID: {waterway_id}) to KML - {waterway_length:.2f} km, {len(coordinates)} points")
    
    return all_waterway_coords

def add_bounding_box_to_kml(folder, south, west, north, east, name="Bounding Box"):
    """Add a bounding box polygon to the KML folder"""
    placemark = ET.SubElement(folder, 'Placemark')
    ET.SubElement(placemark, 'name').text = name
    ET.SubElement(placemark, 'description').text = f"South: {south}, West: {west}, North: {north}, East: {east}"
    
    # Add style for the box
    ET.SubElement(placemark, 'styleUrl').text = "#box_style"
    
    # Add polygon geometry
    polygon = ET.SubElement(placemark, 'Polygon')
    outer_boundary = ET.SubElement(polygon, 'outerBoundaryIs')
    linear_ring = ET.SubElement(outer_boundary, 'LinearRing')
    
    # Create coordinates string (clockwise order)
    coords = f"{west},{south},0 {east},{south},0 {east},{north},0 {west},{north},0 {west},{south},0"
    ET.SubElement(linear_ring, 'coordinates').text = coords
    
    return placemark

def create_output_kml_structure():
    """Create the basic structure for the output KML file."""
    kml = ET.Element('kml', xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, 'Document', id="root_doc")
    ET.SubElement(document, 'name').text = "Romanian Waterways - OSM Data"
    
    # Add a description
    ET.SubElement(document, 'description').text = """
    Waterways fetched from OpenStreetMap via Overpass API
    """
    
    # Create folder for Overpass data
    folder_overpass = ET.SubElement(document, 'Folder', id="overpass_folder")
    ET.SubElement(folder_overpass, 'name').text = "Overpass API Waterways"
    ET.SubElement(folder_overpass, 'description').text = "Waterways fetched from OpenStreetMap via Overpass API"
    
    # Add basic styles for OSM data
    style = ET.SubElement(document, 'Style', id="osm_style")
    line_style = ET.SubElement(style, 'LineStyle')
    ET.SubElement(line_style, 'color').text = "ff00aaff"  # Yellow
    ET.SubElement(line_style, 'width').text = '4'
    
    # Add marker style for points
    marker_style = ET.SubElement(document, 'Style', id="marker_style")
    icon_style = ET.SubElement(marker_style, 'IconStyle')
    ET.SubElement(icon_style, 'color').text = "ff0000ff"  # Red
    ET.SubElement(icon_style, 'scale').text = '1.0'
    
    # Add bounding box style
    box_style = ET.SubElement(document, 'Style', id="box_style")
    line_style = ET.SubElement(box_style, 'LineStyle')
    ET.SubElement(line_style, 'color').text = "ff00ff00"  # Green
    ET.SubElement(line_style, 'width').text = '2'
    poly_style = ET.SubElement(box_style, 'PolyStyle')
    ET.SubElement(poly_style, 'color').text = "4000ff00"  # Transparent green
    ET.SubElement(poly_style, 'fill').text = '1'
    ET.SubElement(poly_style, 'outline').text = '1'
    
    return kml

def add_marker_to_kml(folder, name, description, lat, lon):
    """Add a marker placemark to the KML folder"""
    placemark = ET.SubElement(folder, 'Placemark')
    ET.SubElement(placemark, 'name').text = name
    ET.SubElement(placemark, 'description').text = description
    
    # Add style for the marker
    ET.SubElement(placemark, 'styleUrl').text = "#marker_style"
    
    # Add point geometry
    point = ET.SubElement(placemark, 'Point')
    ET.SubElement(point, 'coordinates').text = f"{lon},{lat},0"
    
    return placemark

def fetch_all_waterways_from_overpass(bounds, original_name=None):
    """
    Fetch all waterway data from Overpass API within specified bounds
    
    Args:
        bounds (tuple): (south, west, north, east) coordinates
        original_name (str, optional): Original waterway name to include in the query
        
    Returns:
        dict: JSON response from Overpass API
    """
    south, west, north, east = bounds
    
    # Create a query for all waterways in the bounding box
    query_parts = [f'way["waterway"]({south},{west},{north},{east});']
    
    # If original name is provided, also add specific queries for it
    if original_name:
        # Normalize and clean the name
        clean_name = original_name.lower()
        for prefix in ['râul', 'raul', 'pârâul', 'paraul', 'r.', 'p.']:
            if clean_name.startswith(prefix + ' '):
                clean_name = clean_name[len(prefix):].strip()
                break
                
        # Escape special characters for regex
        escaped_name = re.escape(clean_name)
        
        # Add a query for partial name matches (case insensitive)
        query_parts.append(f'way["waterway"]["name"~"{escaped_name}",i]({south},{west},{north},{east});')
        query_parts.append(f'way["waterway"]["name:ro"~"{escaped_name}",i]({south},{west},{north},{east});')
    
    # Construct the full query
    query = """
    [out:json][timeout:30];
    (
      """ + "\n      ".join(query_parts) + """
    );
    out body geom;
    """
    
    # Send request to Overpass API
    overpass_url = "https://overpass-api.de/api/interpreter"
    try:
        if original_name:
            logging.info(f"Sending Overpass query for ALL waterways in region with preference for '{original_name}': ({south},{west},{north},{east})")
        else:
            logging.info(f"Sending Overpass query for ALL waterways in region: ({south},{west},{north},{east})")
            
        response = requests.post(overpass_url, data={"data": query})
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Log the size of the response for debugging
        json_response = response.json()
        element_count = len(json_response.get("elements", []))
        logging.info(f"Received {element_count} waterway elements from Overpass API")
        
        return json_response
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching from Overpass API: {str(e)}")
        return {"elements": []}

def fetch_waterways_by_search_term(bounds, search_term):
    """
    Fetch waterway data from Overpass API using a broader search term approach
    
    Args:
        bounds (tuple): (south, west, north, east) coordinates
        search_term (str): The search term to use for filtering waterways
        
    Returns:
        dict: JSON response from Overpass API
    """
    south, west, north, east = bounds
    
    if not search_term:
        logging.warning("No search term provided, skipping search")
        return {"elements": []}
    
    # Normalize and clean the search term
    cleaned_term = search_term.lower()
    for prefix in ['râul', 'raul', 'pârâul', 'paraul', 'r.', 'p.']:
        if cleaned_term.startswith(prefix + ' '):
            cleaned_term = cleaned_term[len(prefix):].strip()
            break
            
    # Extract individual words that might be useful for searching
    search_words = []
    for word in cleaned_term.split():
        if len(word) >= 3:  # Only use words that are meaningful (at least 3 chars)
            search_words.append(word)
    
    # If no useful words found, use the original term
    if not search_words:
        search_words = [cleaned_term]
    
    # Log the search terms
    logging.info(f"Searching for waterways with these terms: {search_words}")
    
    # Create query parts - one query for each search word
    query_parts = []
    for word in search_words:
        # Escape special characters for regex
        escaped_word = re.escape(word)
        
        # Add query for this word - search in both name and name:ro tags
        query_parts.append(f'way["waterway"]["name"~"{escaped_word}",i]({south},{west},{north},{east});')
        query_parts.append(f'way["waterway"]["name:ro"~"{escaped_word}",i]({south},{west},{north},{east});')
    
    # Construct the full query
    query = """
    [out:json][timeout:30];
    (
      """ + "\n      ".join(query_parts) + """
    );
    out body geom;
    """
    
    # Send request to Overpass API
    overpass_url = "https://overpass-api.de/api/interpreter"
    try:
        logging.info(f"Sending Overpass query using search terms from: '{search_term}'")
        response = requests.post(overpass_url, data={"data": query})
        response.raise_for_status()
        
        # Log the size of the response
        json_response = response.json()
        element_count = len(json_response.get("elements", []))
        logging.info(f"Received {element_count} waterway elements from Overpass API")
        
        return json_response
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching from Overpass API: {str(e)}")
        return {"elements": []}

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process Romanian waterways for fishing segments')
    parser.add_argument('--input', default="data/hotosm_rou_waterways_lines_kml.kml", help='Input KML file')
    parser.add_argument('--output', default="data/romanian_fishing_segments.kml", help='Output KML file')
    parser.add_argument('--debug-waterway', help='Filter for a specific waterway (for debugging)')
    parser.add_argument('--debug-county', help='Filter for a specific county (for debugging)')
    parser.add_argument('--include-all-matching', action='store_true', 
                       help='Include all waterways that match the debug waterway name (partial matches)')
    parser.add_argument('--test-overpass', action='store_true',
                       help='Test Overpass API with random city-to-city segments')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of random samples to test with Overpass')
    
    args = parser.parse_args()
    
    # Hardcode the ANPA CSV file path
    anpa_csv_path = "data/ANPA_habitats_contractate_2025_full.csv"
    
    # Test Overpass API if requested
    if args.test_overpass:
        output_path = "data/osm_waterways.kml" if args.output == "data/romanian_fishing_segments.kml" else args.output
        test_overpass_with_random_segments(anpa_csv_path, output_path, args.num_samples)
        exit(0)
    
    # Set debug filter if requested
    if args.debug_waterway or args.debug_county:
        DEBUG_FILTER['enabled'] = True
        DEBUG_FILTER['waterway'] = args.debug_waterway
        DEBUG_FILTER['county'] = args.debug_county
        DEBUG_FILTER['include_all_matching'] = args.include_all_matching
        logging.info(f"Debug mode enabled - filtering for waterway: {args.debug_waterway}, county: {args.debug_county}")
        if args.include_all_matching:
            logging.info(f"Including all waterways that match '{args.debug_waterway}'")
    
    input_kml = args.input
    output_kml = args.output
    
    if not os.path.exists(input_kml):
        logging.error(f"Input KML file not found: {input_kml}")
    else:
        process_kml(input_kml, output_kml) 