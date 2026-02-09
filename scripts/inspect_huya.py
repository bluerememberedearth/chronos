import subprocess
import json
import sys

def inspect_stream(url):
    print(f"Inspecting {url}...")
    # Get full JSON dump from yt-dlp
    cmd = ['yt-dlp', '-J', url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running yt-dlp:", result.stderr)
        return

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Failed to parse JSON")
        return

    print("Title:", data.get('title'))
    print("Uploader:", data.get('uploader'))
    
    formats = data.get('formats', [])
    print(f"Found {len(formats)} formats.")
    
    # Check for HTTP headers in formats
    for f in formats:
        if 'http_headers' in f:
            print(f"\nFormat {f.get('format_id')} headers:")
            print(json.dumps(f['http_headers'], indent=2))
            break # Just show one for now
            
    # Also check if there are specific 'url' fields that look like m3u8 or flv
    # and if they token parameters.
    
    # Check best format
    best = data.get('url')
    if best:
        print("\nBest URL:", best[:100], "...")


if __name__ == "__main__":
    inspect_stream("https://www.huya.com/lpl")
