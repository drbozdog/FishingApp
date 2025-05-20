import pandas as pd
import folium
from folium import plugins
import json

def is_valid_coords(coords):
    """Check if coordinates are valid numbers"""
    try:
        if not all(isinstance(coords[key], (int, float)) for key in [
            'start_point_latitude', 'start_point_longitude',
            'end_point_latitude', 'end_point_longitude'
        ]):
            return False
        return True
    except (KeyError, TypeError):
        return False

def create_interactive_map():
    # Read the CSV file
    df = pd.read_csv('data/processing_status.csv')
    print(f"Found {len(df)} rows in the CSV")
    
    # Create a base map
    m = folium.Map(
        location=[46.0667, 23.5833],
        zoom_start=9,
        tiles='OpenStreetMap'
    )
    
    # Process each waterway
    for idx, row in df.iterrows():
        try:
            if row['status'] == 'succeeded' and row['geocoded_output']:
                coords = json.loads(row['geocoded_output'])
                
                # Skip if coordinates are not valid
                if not is_valid_coords(coords):
                    print(f"Skipping row {idx}: Invalid coordinates for {row['Habitat']}")
                    continue
                
                # Create a feature group for this waterway
                fg = folium.FeatureGroup(name=row['Habitat'])
                
                # Create start and end markers using regular Markers (draggable)
                start_marker = folium.Marker(
                    location=[coords['start_point_latitude'], coords['start_point_longitude']],
                    popup=f"Start: {row['Habitat']} - {row['Limits']}",
                    icon=folium.Icon(color='green'),
                    draggable=True
                )
                
                end_marker = folium.Marker(
                    location=[coords['end_point_latitude'], coords['end_point_longitude']],
                    popup=f"End: {row['Habitat']} - {row['Limits']}",
                    icon=folium.Icon(color='red'),
                    draggable=True
                )
                
                # Add unique IDs to the markers
                start_marker._name = f"start_{idx}"
                end_marker._name = f"end_{idx}"
                
                # Draw a line between points
                line = folium.PolyLine(
                    locations=[
                        [coords['start_point_latitude'], coords['start_point_longitude']],
                        [coords['end_point_latitude'], coords['end_point_longitude']]
                    ],
                    weight=4,
                    color='blue',
                    opacity=0.8,
                    popup=f"{row['Habitat']}\nLength: {row['Length_surface']}"
                )
                line._name = f"line_{idx}"
                
                # Add elements to feature group
                start_marker.add_to(fg)
                end_marker.add_to(fg)
                line.add_to(fg)
                
                # Add the feature group to the map
                fg.add_to(m)
                
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            print(f"Problematic row data: {row['Habitat']}")

    # Add custom JavaScript for marker drag handling and controls
    custom_js = """
    <script>
    var markerPairs = {};
    var polylines = {};
    
    document.addEventListener('DOMContentLoaded', function() {
        // Create coordinates display div
        var coordsDiv = document.createElement('div');
        coordsDiv.style.position = 'fixed';
        coordsDiv.style.bottom = '10px';
        coordsDiv.style.left = '10px';
        coordsDiv.style.backgroundColor = 'white';
        coordsDiv.style.padding = '10px';
        coordsDiv.style.borderRadius = '5px';
        coordsDiv.style.zIndex = '1000';
        coordsDiv.id = 'coords-display';
        document.body.appendChild(coordsDiv);

        // Initialize marker and line tracking
        setTimeout(function() {
            map.eachLayer(function(layer) {
                if (layer instanceof L.FeatureGroup) {
                    var groupId = null;
                    var startMarker = null;
                    var endMarker = null;
                    var polyline = null;

                    layer.eachLayer(function(subLayer) {
                        if (subLayer instanceof L.Marker) {
                            if (subLayer.options.icon.options.color === 'green') {
                                startMarker = subLayer;
                            } else if (subLayer.options.icon.options.color === 'red') {
                                endMarker = subLayer;
                            }
                        } else if (subLayer instanceof L.Polyline) {
                            polyline = subLayer;
                        }
                    });

                    if (startMarker && endMarker && polyline) {
                        groupId = startMarker.options.name.split('_')[1];
                        markerPairs[groupId] = {start: startMarker, end: endMarker};
                        polylines[groupId] = polyline;

                        // Add drag events to both markers
                        [startMarker, endMarker].forEach(function(marker) {
                            marker.on('drag', function(e) {
                                var position = marker.getLatLng();
                                var isStart = marker.options.icon.options.color === 'green';
                                var line = polylines[groupId];
                                var pair = markerPairs[groupId];
                                
                                // Update the polyline
                                var newLatLngs = [
                                    pair.start.getLatLng(),
                                    pair.end.getLatLng()
                                ];
                                line.setLatLngs(newLatLngs);

                                // Update coordinates display
                                coordsDiv.innerHTML = `
                                    ${isStart ? 'Start' : 'End'} Marker<br>
                                    Lat: ${position.lat.toFixed(6)}<br>
                                    Lng: ${position.lng.toFixed(6)}
                                `;
                            });
                        });
                    }
                }
            });
        }, 1000);

        // Create and add the Show All button
        var controlDiv = document.querySelector('.leaflet-control-layers-overlays');
        var showAllDiv = document.createElement('div');
        showAllDiv.style.marginTop = '10px';
        showAllDiv.style.borderTop = '1px solid #ccc';
        showAllDiv.style.paddingTop = '10px';
        
        var showAllBtn = document.createElement('button');
        showAllBtn.innerHTML = 'Show All';
        showAllBtn.style.width = '100%';
        showAllBtn.style.padding = '5px';
        showAllBtn.style.backgroundColor = '#fff';
        showAllBtn.style.border = '1px solid #ccc';
        showAllBtn.style.borderRadius = '4px';
        showAllBtn.style.cursor = 'pointer';
        
        showAllBtn.onmouseover = function() {
            this.style.backgroundColor = '#f4f4f4';
        }
        showAllBtn.onmouseout = function() {
            this.style.backgroundColor = '#fff';
        }
        
        showAllDiv.appendChild(showAllBtn);
        controlDiv.appendChild(showAllDiv);
        
        // Get all checkboxes in the layer control
        var checkboxes = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
        
        function updateLayers(selectedCheckbox) {
            checkboxes.forEach(function(checkbox) {
                if (checkbox !== selectedCheckbox) {
                    checkbox.checked = false;
                    var event = new Event('change');
                    checkbox.dispatchEvent(event);
                }
            });
        }
        
        // Add click event listener to each checkbox
        checkboxes.forEach(function(checkbox) {
            checkbox.addEventListener('change', function(e) {
                if (e.target.checked) {
                    updateLayers(e.target);
                }
            });
        });
        
        // Add click event listener to Show All button
        showAllBtn.addEventListener('click', function() {
            checkboxes.forEach(function(checkbox) {
                checkbox.checked = true;
                var event = new Event('change');
                checkbox.dispatchEvent(event);
            });
            
            var bounds = [];
            map.eachLayer(function(layer) {
                if (layer.getBounds) {
                    bounds.push(layer.getBounds());
                }
            });
            if (bounds.length > 0) {
                map.fitBounds(bounds[0].extend(bounds[bounds.length - 1]));
            }
        });
        
        // Uncheck all checkboxes initially
        checkboxes.forEach(function(checkbox) {
            checkbox.checked = false;
            var event = new Event('change');
            checkbox.dispatchEvent(event);
        });
    });
    </script>
    """
    
    # Add the custom JavaScript to the map
    m.get_root().html.add_child(folium.Element(custom_js))
    
    # Add layer control to the map
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add fullscreen control
    plugins.Fullscreen().add_to(m)
    
    # Add mouse position
    plugins.MousePosition().add_to(m)
    
    # Save the map
    m.save('waterways_map.html')
    print("\nMap saved to waterways_map.html")

if __name__ == '__main__':
    create_interactive_map()