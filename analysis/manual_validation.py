import csv
import json

INPUT_CSV = 'data/processing_status.csv'
OUTPUT_HTML = 'manual_validation.html'

def load_records(csv_file):
    records = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') == 'succeeded' and row.get('geocoded_output'):
                try:
                    coords = json.loads(row['geocoded_output'])
                    rec = {
                        'row_identifier': row['row_identifier'],
                        'habitat': row['Habitat'],
                        'limits': row['Limits'],
                        'start_lat': float(coords['start_point_latitude']),
                        'start_lon': float(coords['start_point_longitude']),
                        'end_lat': float(coords['end_point_latitude']),
                        'end_lon': float(coords['end_point_longitude']),
                    }
                except Exception:
                    continue
                records.append(rec)
    return records


def generate_html(records):
    records_json = json.dumps(records)
    html_template = """
<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'/>
<title>Geocoding Validation</title>
<link rel='stylesheet' href='https://unpkg.com/leaflet/dist/leaflet.css'/>
<script src='https://unpkg.com/leaflet/dist/leaflet.js'></script>
<style>
  #map { height: 80vh; }
  #controls { margin-top: 10px; }
  button { margin-right: 5px; padding: 5px 10px; }
</style>
</head>
<body>
<h3 id='header'></h3>
<div id='info'></div>
<div id='map'></div>
<div id='controls'>
  <button onclick='saveRecord()'>Save &amp; Next</button>
  <button onclick='skipRecord()'>Skip</button>
</div>
<div id='download' style='display:none'>
  <p>All records reviewed.</p>
  <button onclick='downloadCSV()'>Download results</button>
</div>
<script>
const dataRecords = RECORDS_JSON_PLACEHOLDER;
let currentIndex = 0;
let validated = [];
let map, startMarker, endMarker, line;

function init() {
    map = L.map('map');
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19}).addTo(map);
    setupRecord(0);
}

function setupRecord(i) {
    const rec = dataRecords[i];
    document.getElementById('header').innerText = 'Record ' + (i + 1) + ' of ' + dataRecords.length;
    document.getElementById('info').innerHTML = '<b>' + rec.habitat + '</b><br>' + rec.limits;
    if (startMarker) map.removeLayer(startMarker);
    if (endMarker) map.removeLayer(endMarker);
    if (line) map.removeLayer(line);

    startMarker = L.marker([rec.start_lat, rec.start_lon], {draggable:true}).addTo(map);
    endMarker = L.marker([rec.end_lat, rec.end_lon], {draggable:true}).addTo(map);
    line = L.polyline([startMarker.getLatLng(), endMarker.getLatLng()], {color:'blue'}).addTo(map);
    map.fitBounds(line.getBounds(), {maxZoom: 15});
    startMarker.on('drag', updateLine);
    endMarker.on('drag', updateLine);
}

function updateLine() {
    line.setLatLngs([startMarker.getLatLng(), endMarker.getLatLng()]);
}

function saveRecord() {
    const rec = dataRecords[currentIndex];
    validated.push({
        row_identifier: rec.row_identifier,
        habitat: rec.habitat,
        limits: rec.limits,
        start_lat: startMarker.getLatLng().lat,
        start_lon: startMarker.getLatLng().lng,
        end_lat: endMarker.getLatLng().lat,
        end_lon: endMarker.getLatLng().lng
    });
    nextRecord();
}

function skipRecord() {
    nextRecord();
}

function nextRecord() {
    currentIndex++;
    if (currentIndex >= dataRecords.length) {
        document.getElementById('controls').style.display = 'none';
        document.getElementById('map').style.display = 'none';
        document.getElementById('download').style.display = 'block';
        document.getElementById('header').innerText = 'All records reviewed';
    } else {
        setupRecord(currentIndex);
    }
}

function downloadCSV() {
    let csv = 'row_identifier,habitat,limits,start_lat,start_lon,end_lat,end_lon\n';
    validated.forEach(r => {
        const row = [r.row_identifier, r.habitat, r.limits, r.start_lat, r.start_lon, r.end_lat, r.end_lon];
        csv += row.join(',') + '\n';
    });
    const blob = new Blob([csv], {type:'text/csv'});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'validated_geocoding.csv';
    link.click();
}
window.onload = init;
</script>
</body>
</html>
"""
    html = html_template.replace('RECORDS_JSON_PLACEHOLDER', records_json)
    return html


def main():
    records = load_records(INPUT_CSV)
    html = generate_html(records)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved validation map to {OUTPUT_HTML} with {len(records)} records")

if __name__ == '__main__':
    main()
