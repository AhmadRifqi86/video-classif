
# import sys
# import subprocess
# import snscrape.modules.twitter as sntwitter

# def download_twitter_video(url, category):
#     # Command yt-dlp dengan opsi --cookies-from-browser
#     output_path = f'/media/arifadh/CRUCIAL2T/socmed/{category}/%(title)s.%(ext)s'
#     command = [
#         'yt-dlp', 
#         '--cookies-from-browser', 'firefox',  # Replace 'firefox' with the browser you use
#         '-f', 'bestvideo+bestaudio/best',
#         '-o', output_path,
#         url
#     ]

#     # Running the command with subprocess
#     try:
#         subprocess.run(command, check=True)
#         print("Video downloaded successfully.")
#     except subprocess.CalledProcessError as e:
#         print("An error occurred:", e)


# def download_videos_from_username(username, category, n):
#     # Using snscrape to get tweets with media from a user account
#     tweet_count = 0
#     for tweet in sntwitter.TwitterUserScraper(username).get_items():
#         # Check if tweet has media (video)
#         if 'media' in tweet.entities and any(m.type == 'video' for m in tweet.media):
#             tweet_url = f"https://twitter.com/{username}/status/{tweet.id}"
#             download_twitter_video(tweet_url, category)
#             tweet_count += 1

#             if tweet_count >= n:
#                 break
#         else:
#             print(f"Skipping tweet {tweet.id}, no video found.")

#     print(f"Downloaded {tweet_count} videos from @{username}")

# def download_videos_from_hashtag(hashtag, category, n):
#     # Using snscrape to search tweets with media based on a hashtag
#     tweet_count = 0
#     for tweet in sntwitter.TwitterHashtagScraper(hashtag).get_items():
#         # Check if tweet has media (video)
#         if 'media' in tweet.entities and any(m.type == 'video' for m in tweet.media):
#             tweet_url = f"https://twitter.com/{tweet.user.username}/status/{tweet.id}"
#             download_twitter_video(tweet_url, category)
#             tweet_count += 1
            
#             # Stop after downloading N videos
#             if tweet_count >= n:
#                 break
#         else:
#             print(f"Skipping tweet {tweet.id}, no video found.")

#     print(f"Downloaded {tweet_count} videos from #{hashtag}")

# def download_videos_from_username_and_hashtag(username, hashtag, category, n):
#     # Using snscrape to search tweets with a specific hashtag from a specific user
#     tweet_count = 0
#     for tweet in sntwitter.TwitterSearchScraper(f'from:{username} #{hashtag}').get_items():
#         # Check if tweet has media (video)
#         if 'media' in tweet.entities and any(m.type == 'video' for m in tweet.media):
#             tweet_url = f"https://twitter.com/{username}/status/{tweet.id}"
#             download_twitter_video(tweet_url, category)
#             tweet_count += 1
            
#             if tweet_count >= n:
#                 break
#         else:
#             print(f"Skipping tweet {tweet.id}, no video found.")

#     print(f"Downloaded {tweet_count} videos from @{username} with #{hashtag}")


# # Check if the necessary arguments are provided
# if len(sys.argv) < 4:
#     print("Please provide the Twitter username (or None), hashtag (or None), category, and number of videos (N) as command line arguments.\nEg: python twitter_downloader.py <Username/None> <Hashtag/None> <Category> <N>")
# else:
#     username = sys.argv[1]
#     hashtag = sys.argv[2]
#     category = sys.argv[3]
    
#     try:
#         n = int(sys.argv[4])
#         if category and n > 0:
#             if username.lower() != "none" and hashtag.lower() != "none":
#                 # Case: Both username and hashtag are specified
#                 download_videos_from_username_and_hashtag(username, hashtag, category, n)
#             elif username.lower() != "none":
#                 # Case: Only username is specified
#                 download_videos_from_username(username, category, n)
#             elif hashtag.lower() != "none":
#                 # Case: Only hashtag is specified
#                 download_videos_from_hashtag(hashtag, category, n)
#             else:
#                 print("Both username and hashtag cannot be None.")
#         else:
#             print("Invalid category or number of videos (N) provided.")
#     except ValueError:
#         print("N must be an integer.")


from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import requests
import os

USERNAME = "bra33954v1l"  # TikTok username
SAVE_DIRECTORY = "/home/arifadh/Downloads/tiktok_videos"
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

def download_video(video_url, save_path):
    response = requests.get(video_url, stream=True)
    with open(save_path, "wb") as video_file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                video_file.write(chunk)

def scrape_videos(username):
    # Set up the Selenium WebDriver
    driver = webdriver.Chrome()  # Install ChromeDriver if not available
    driver.get(f"https://www.tiktok.com/@{username}")
    time.sleep(5)

    # Scroll the page to load videos
    for _ in range(5):  # Adjust the range for more scrolling
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    # Extract video URLs
    video_elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/video/')]")
    video_urls = [element.get_attribute("href") for element in video_elements]
    driver.quit()

    return video_urls

def main():
    video_urls = scrape_videos(USERNAME)
    for video_url in video_urls:
        print(f"Downloading: {video_url}")
        video_id = video_url.split("/")[-1]
        save_path = os.path.join(SAVE_DIRECTORY, f"{USERNAME}_{video_id}.mp4")
        download_video(video_url, save_path)

if __name__ == "__main__":
    main()