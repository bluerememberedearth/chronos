import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VODFetcher")

# Example: T1 vs BLG Worlds 2024 Finals Game 5 (High probability of action)
DEFAULT_VOD_URL = "https://www.youtube.com/watch?v=SomeLoLGameID" 
# Better: Use a search or specific ID. Let's use a generic LoL Gameplay video to avoid broken links if possible, 
# or just let user provide it. For now, I'll use a placeholder variable.

def fetch_youtube_clip(url: str, output_path: str = "data/test_vod.mp4"):
    """
    Downloads the first 2 minutes of a YouTube video for testing.
    """
    if not os.path.exists("data"):
        os.makedirs("data")

    logger.info(f"Downloading clip from {url} to {output_path}...")
    
    # yt-dlp command to download best mp4 format, max 1 minute
    # -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' 
    cmd = [
        "yt-dlp",
        url,
        "-o", output_path,
        "--download-sections", "*00:05:00-00:06:00", # Download minute 5 to 6 (usually laning/early action)
        "--force-keyframes-at-cuts",
        "-f", "mp4"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("Download complete.")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        return None
    except FileNotFoundError:
        logger.error("yt-dlp not found. Please install it/run setup_env.sh")
        return None

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else input("Enter YouTube URL: ")
    fetch_youtube_clip(url)
