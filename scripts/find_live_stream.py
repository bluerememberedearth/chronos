import subprocess
import concurrent.futures

CHANNELS = [
    "https://www.twitch.tv/shroud",
    "https://www.twitch.tv/xqc",
    "https://www.twitch.tv/summit1g",
    "https://www.twitch.tv/lirik",
    "https://www.twitch.tv/pokimane",
    "https://www.twitch.tv/ninja",
    "https://www.twitch.tv/riotgames",
    "https://www.twitch.tv/lck",
    "https://www.twitch.tv/lcs",
    "https://www.twitch.tv/lec",
    "https://www.twitch.tv/otplol_",
    "https://www.twitch.tv/midbeast",
    "https://www.twitch.tv/doublelift",
    "https://www.twitch.tv/lol_tyler1",
    "https://www.twitch.tv/caedrel"
]

def check_channel(url):
    print(f"Checking {url}...")
    try:
        cmd = ["/Users/solana/chronos_v3/.venv/bin/python3", "-m", "streamlink", url, "best"]
        # Just check if it finds streams, don't play
        result = subprocess.run(cmd + ["--json"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and "error" not in result.stdout:
            print(f"LIVE: {url}")
            return url
    except Exception as e:
        pass
    return None

def find_live():
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(check_channel, url) for url in CHANNELS]
        for future in concurrent.futures.as_completed(futures):
            url = future.result()
            if url:
                print(f"Found live channel: {url}")
                return url
    print("No live channels found.")
    return None

if __name__ == "__main__":
    find_live()
