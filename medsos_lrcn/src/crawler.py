import pyktok as pyk
from playwright.sync_api import sync_playwright
import time
import browser_cookie3
import requests

BACKEND_CHECKER = 'http://localhost:5000/video_labels'
VIDEO_DIR = '/home/arifadh/Downloads/tiktok_videos/'
pyk.specify_browser('firefox')
#print(pyk.__file__)

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
        browser = p.firefox.launch(headless=False)  # Launch browser
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
            page.mouse.wheel(0, 10000)
            time.sleep(2)

        # Extract video links
        video_links = page.eval_on_selector_all(
            "a[href*='/video/']",
            "elements => elements.map(e => e.href)"
        )

        # Close the browser
        browser.close()
        return video_links

def main():
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


if __name__ == "__main__":
    main()


#pyk.save_tiktok_multi_page('.aicu',ent_type='user',save_video=False,metadata_fn='tiktok.csv',save_dir=all_config.VIDEO_DIR)
