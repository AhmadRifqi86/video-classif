import requests
import time

# Load video URLs from a text file
def load_video_urls(file_path="link_percobaan.txt"):
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines() if line.strip()]

# Base API URL
API_URL = "http://localhost:5000/get_labels?url="

def measure_latency():
    video_urls = load_video_urls()
    if not video_urls:
        print("No video URLs found in the file.")
        return

    latencies = []
    start_time = time.time()  # Start measuring time

    for url in video_urls:
        req_start = time.time()  # Start time for individual request
        response = requests.get(API_URL + url)
        req_end = time.time()  # End time for individual request
        latencies.append(req_end - req_start)

    end_time = time.time()  # End measuring time

    avg_latency = sum(latencies) / len(latencies)
    total_time = end_time - start_time

    print(f"Total videos processed: {len(video_urls)}")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average latency per video: {avg_latency:.2f} seconds")

if __name__ == "__main__":
    measure_latency()
