import requests
import time

# Load video URLs from a text file
def load_video_urls(file_path="link_percobaan.txt"):
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines() if line.strip()]

# Base API URL
API_URL = "http://localhost:5000/get_labels?url="

def classify_video(url):
    response = requests.get(API_URL + url)
    return response.json()  # Assuming the API returns JSON

def main():
    video_urls = load_video_urls()
    if not video_urls:
        print("No video URLs found in the file.")
        return

    start_time = time.time()  # Start measuring time
    
    results = []
    for url in video_urls:
        result = classify_video(url)
        results.append(result)
    
    end_time = time.time()  # End measuring time
    total_time = end_time - start_time
    throughput = len(video_urls) / total_time  # Videos processed per second
    
    print(f"Total videos processed: {len(video_urls)}")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} videos/second")

if __name__ == "__main__":
    main()