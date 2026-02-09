import streamlink
import logging
import cv2
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamlinkDebug")

def test_streamlink(url):
    session = streamlink.Streamlink()
    # Huya often needs user-agent
    session.set_option("http-headers", "User-Agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    
    logger.info(f"Resolving {url}...")
    try:
        streams = session.streams(url)
    except Exception as e:
        logger.error(f"Streamlink error: {e}")
        return

    if not streams:
        logger.info("No streams found.")
        return

    logger.info(f"Found qualities: {list(streams.keys())}")
    
    if 'best' in streams:
        stream = streams['best']
        stream_url = stream.url
        logger.info(f"Best stream URL: {stream_url}")
        
        # Test if CV2 can open it
        logger.info("Attempting to open with OpenCV...")
        cap = cv2.VideoCapture(stream_url)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                logger.info("Success! Frame read.")
            else:
                logger.error("Failed to read frame.")
            cap.release()
        else:
            logger.error("Failed to open with OpenCV.")
    else:
        logger.info("No 'best' stream found.")

if __name__ == "__main__":
    test_streamlink("https://www.huya.com/lpl")
