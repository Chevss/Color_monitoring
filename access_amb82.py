import cv2
import requests
import numpy as np
from urllib.parse import urlparse

def get_mjpeg_stream(url):
    """
    Access MJPEG stream from Ameba board and process frames
    """
    # Parse the URL to get just the scheme, netloc, and path
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    # If no path is provided, use the /stream endpoint
    if not parsed_url.path or parsed_url.path == "/":
        stream_url = f"{base_url}/stream"
    else:
        stream_url = url
    
    print(f"Connecting to stream at {stream_url}")
    
    # Open stream with requests
    r = requests.get(stream_url, stream=True, timeout=10)
    
    if r.status_code != 200:
        print(f"Failed to connect: {r.status_code}")
        return
    
    # Variables for parsing MJPEG stream
    bytes_data = bytes()
    boundary = None
    
    # Find the multipart boundary
    content_type = r.headers.get('content-type', '')
    if 'boundary=' in content_type:
        boundary = content_type.split('boundary=')[1].strip()
        print(f"Found boundary: {boundary}")
    else:
        print("Could not find boundary in content type header")
        boundary = "123456789000000000000987654321"  # Default in your Arduino code
    
    boundary_bytes = f"--{boundary}".encode()
    
    # Process the stream
    cv2.namedWindow("AMB28 Stream", cv2.WINDOW_NORMAL)
    
    try:
        for chunk in r.iter_content(chunk_size=1024):
            if not chunk:
                continue
                
            bytes_data += chunk
            
            # Look for the start of an image
            a = bytes_data.find(b'\xff\xd8')
            # Look for the end of an image
            b = bytes_data.find(b'\xff\xd9')
            
            if a != -1 and b != -1 and a < b:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                
                # Decode the JPEG image
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if img is not None and img.size > 0:
                    # Display the image
                    cv2.imshow("AMB28 Stream", img)
                    
                    # Process key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save a snapshot
                        cv2.imwrite("snapshot.jpg", img)
                        print("Snapshot saved as snapshot.jpg")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        cv2.destroyAllWindows()
        print("Stream closed")

if __name__ == "__main__":
    # Replace with your Ameba board's IP address
    camera_ip = "192.168.1.39"
    get_mjpeg_stream(f"http://{camera_ip}/stream")