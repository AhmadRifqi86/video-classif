# import sys
# import os
# import re

# import requests
# import bs4

# from tqdm import tqdm
# from pathlib import Path


# def download_video(url, file_name) -> None:
#     """Download a video from a URL into a filename.

#     Args:
#         url (str): The video URL to download
#         file_name (str): The file name or path to save the video to.
#     """

#     response = requests.get(url, stream=True)
#     total_size = int(response.headers.get("content-length", 0))
#     block_size = 1024
#     progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

#     download_path = os.path.join(Path.home(), "Downloads", file_name)

#     with open(download_path, "wb") as file:
#         for data in response.iter_content(block_size):
#             progress_bar.update(len(data))
#             file.write(data)

#     progress_bar.close()
#     print("Video downloaded successfully!")


# def download_twitter_video(url):
#     """Extract the highest quality video url to download into a file

#     Args:
#         url (str): The twitter post URL to download from
#     """

#     api_url = f"https://twitsave.com/info?url={url}"

#     response = requests.get(api_url)
#     data = bs4.BeautifulSoup(response.text, "html.parser")
#     download_button = data.find_all("div", class_="origin-top-right")[0]
#     quality_buttons = download_button.find_all("a")
#     highest_quality_url = quality_buttons[0].get("href") # Highest quality video url
    
#     file_name = data.find_all("div", class_="leading-tight")[0].find_all("p", class_="m-2")[0].text # Video file name
#     file_name = re.sub(r"[^a-zA-Z0-9]+", ' ', file_name).strip() + ".mp4" # Remove special characters from file name
    
#     download_video(highest_quality_url, file_name)


# if len(sys.argv) < 2:
#     print("Please provide the Twitter video URL as a command line argument.\nEg: python twitter_downloader.py <URL>")
# else:
#     url = sys.argv[1]
#     if url:
#         download_twitter_video(url)
#     else:
#         print("Invalid Twitter video URL provided.")


import sys
import subprocess
import snscrape.modules.twitter as sntwitter

def download_twitter_video(url, category):
    # Command yt-dlp dengan opsi --cookies-from-browser
    output_path = f'/home/arifadh/Desktop/socmed_vid/{category}/%(title)s.%(ext)s'
    command = [
        'yt-dlp', 
        '--cookies-from-browser', 'firefox',  # Replace 'firefox' with the browser you use
        '-f', 'bestvideo+bestaudio/best',
        '-o', output_path,
        url
    ]

    # Running the command with subprocess
    try:
        subprocess.run(command, check=True)
        print("Video downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print("An error occurred:", e)


def download_videos_from_username(username, category, n):
    # Using snscrape to get tweets with media from a user account
    tweet_count = 0
    for tweet in sntwitter.TwitterUserScraper(username).get_items():
        # Check if tweet has media (video)
        if 'media' in tweet.entities and any(m.type == 'video' for m in tweet.media):
            tweet_url = f"https://twitter.com/{username}/status/{tweet.id}"
            download_twitter_video(tweet_url, category)
            tweet_count += 1

            if tweet_count >= n:
                break
        else:
            print(f"Skipping tweet {tweet.id}, no video found.")

    print(f"Downloaded {tweet_count} videos from @{username}")

def download_videos_from_hashtag(hashtag, category, n):
    # Using snscrape to search tweets with media based on a hashtag
    tweet_count = 0
    for tweet in sntwitter.TwitterHashtagScraper(hashtag).get_items():
        # Check if tweet has media (video)
        if 'media' in tweet.entities and any(m.type == 'video' for m in tweet.media):
            tweet_url = f"https://twitter.com/{tweet.user.username}/status/{tweet.id}"
            download_twitter_video(tweet_url, category)
            tweet_count += 1
            
            # Stop after downloading N videos
            if tweet_count >= n:
                break
        else:
            print(f"Skipping tweet {tweet.id}, no video found.")

    print(f"Downloaded {tweet_count} videos from #{hashtag}")

def download_videos_from_username_and_hashtag(username, hashtag, category, n):
    # Using snscrape to search tweets with a specific hashtag from a specific user
    tweet_count = 0
    for tweet in sntwitter.TwitterSearchScraper(f'from:{username} #{hashtag}').get_items():
        # Check if tweet has media (video)
        if 'media' in tweet.entities and any(m.type == 'video' for m in tweet.media):
            tweet_url = f"https://twitter.com/{username}/status/{tweet.id}"
            download_twitter_video(tweet_url, category)
            tweet_count += 1
            
            if tweet_count >= n:
                break
        else:
            print(f"Skipping tweet {tweet.id}, no video found.")

    print(f"Downloaded {tweet_count} videos from @{username} with #{hashtag}")


# Check if the necessary arguments are provided
if len(sys.argv) < 4:
    print("Please provide the Twitter username (or None), hashtag (or None), category, and number of videos (N) as command line arguments.\nEg: python twitter_downloader.py <Username/None> <Hashtag/None> <Category> <N>")
else:
    username = sys.argv[1]
    hashtag = sys.argv[2]
    category = sys.argv[3]
    
    try:
        n = int(sys.argv[4])
        if category and n > 0:
            if username.lower() != "none" and hashtag.lower() != "none":
                # Case: Both username and hashtag are specified
                download_videos_from_username_and_hashtag(username, hashtag, category, n)
            elif username.lower() != "none":
                # Case: Only username is specified
                download_videos_from_username(username, category, n)
            elif hashtag.lower() != "none":
                # Case: Only hashtag is specified
                download_videos_from_hashtag(hashtag, category, n)
            else:
                print("Both username and hashtag cannot be None.")
        else:
            print("Invalid category or number of videos (N) provided.")
    except ValueError:
        print("N must be an integer.")
