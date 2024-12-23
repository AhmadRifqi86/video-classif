#import pyktok as pyk        #import if development stage
#import custom_pyktok.pyktok as pyk #import statement if build
from playwright.sync_api import sync_playwright
import time
import browser_cookie3
import requests
import os
import random

APP_STAGE = os.getenv("APP_STAGE", "devel")

if APP_STAGE == "prod":
    import custom_pyktok.pyktok as pyk
    print("Using custom packaged version of pyktok")
else:
    import pyktok as pyk
    print("Using development version of pyktok")

BACKEND_CHECKER = 'http://backend:5000/video_labels' if APP_STAGE == "prod" else 'http://localhost:5000/video_labels'
VIDEO_DIR = '/app/videos' if APP_STAGE == "prod" else '/home/arifadh/Downloads/tiktok_videos'
pyk.specify_browser('firefox')
print(pyk.__file__)

def is_url_classified(video_url):
    # video_url = loader_data.construct_url(video_url)
    # print("checker, video_url: ",video_url)
    try:
        response = requests.get(BACKEND_CHECKER, params={"url": video_url})
        if response.status_code == 200:
            data = response.json()
            if "url" in data and "labels" in data:
                print(f"URL {video_url} is already classified with label: {data['labels']}")
                return True
            else:
                print(f"URL {video_url} is not classified yet.")
                return False
        else:
            print(f"Failed to check classification status for {video_url}. HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"Error checking classification status for {video_url}: {e}")
        return False

def load_cookies(page):
    cookies = browser_cookie3.firefox(domain_name="tiktok.com")
    for cookie in cookies:
        secure = bool(cookie.secure)
        page.context.add_cookies([{
            "name": cookie.name,
            "value": cookie.value,
            "domain": cookie.domain,
            "path": cookie.path,
            "secure": secure
        }])

def scrape_tiktok_video_links(profile_url):
    with sync_playwright() as p:
        browser = p.firefox.launch(
            headless=False #,
            # args=[
            #     '--no-sandbox',
            #     '--disable-setuid-sandbox',
            #     '--disable-dev-shm-usage'
            # ]
        )
        
        page = browser.new_page()

        # Set headers to mimic a real browser
        page.set_extra_http_headers({
            'Accept-Encoding': 'gzip, deflate, sdch',
            'Accept-Language': 'en-US,en;q=0.8',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive'
        })

        # Go to the TikTok profile page
        page.goto(profile_url)

        # Load cookies and reload the page
        load_cookies(page)
        page.reload()
        time.sleep(5)

        # Check for "Something went wrong" and click the Refresh button if present
        while page.locator("text=Something went wrong").count() > 0:
            print("Detected 'Something went wrong' page.")
            refresh_button = page.locator("button:has-text('Refresh')")  # Locate the Refresh button
            if refresh_button.count() > 0:
                refresh_button.click()  # Click the button
                print("Clicked the Refresh button. Waiting for reload...")
            else:
                print("Refresh button not found. Waiting for the page to reload automatically...")
            time.sleep(10) 

        # Scroll to load more videos
        for _ in range(5):
            print("scroll")
            page.mouse.wheel(0, 10000)
            time.sleep(random.randint(1,6))

        # Extract video links
        video_links = page.eval_on_selector_all(
            "a[href*='/video/']",
            "elements => elements.map(e => e.href)"
        )
        print("extracted vid_links: ",video_links)

        # Close the browser
        browser.close()
        return video_links

def main_old():
    # TikTok profile URL, jadiin apa ya ini? ga mungkin constant
    profile_url = "https://www.tiktok.com/@naiwen88"  #how to retrieve this nigga

    # Extract video links
    video_links = scrape_tiktok_video_links(profile_url)
    vid_links = []

    # Print the video links
    print("Extracted Video Links:")
    for link in video_links:
        if is_url_classified(link):
            continue
        vid_links.append(link)

    print("vid links after filtered: ",vid_links)
    #filter check

    pyk.save_tiktok_multi_urls(vid_links,True,'',1,save_dir=VIDEO_DIR)

def main():
    # File containing TikTok profile URLs (one per line)
    urls_file = "profile_urls.txt"  # Path to your text file

    # Read URLs from the text file
    try:
        with open(urls_file, 'r') as file:
            profile_urls = [line.strip() for line in file.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: The file {urls_file} does not exist.")
        return
    except Exception as e:
        print(f"Error reading {urls_file}: {e}")
        return

    print("Extracted TikTok Profile URLs:")
    print(profile_urls)

    # Iterate over each profile URL
    for profile_url in profile_urls:
        # Extract video links for the profile
        video_links = scrape_tiktok_video_links(profile_url)
        vid_links = []

        print(f"Extracted Video Links for {profile_url}:")
        for link in video_links:
            if is_url_classified(link):
                continue
            vid_links.append(link)

        print(f"Filtered video links for {profile_url}: ", vid_links)

        # Save the filtered video links
        pyk.save_tiktok_multi_urls(vid_links, True, '', 1, save_dir=VIDEO_DIR)

if __name__ == "__main__":
    main()


#pyk.save_tiktok_multi_page('.aicu',ent_type='user',save_video=False,metadata_fn='tiktok.csv',save_dir=all_config.VIDEO_DIR)
