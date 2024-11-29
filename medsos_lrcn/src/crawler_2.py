from playwright.sync_api import sync_playwright
import time
import browser_cookie3
import requests
import os
import logging

APP_STAGE = os.getenv("APP_STAGE", "devel")
print("APP_STAGE: ",APP_STAGE)

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

logging.basicConfig(level=logging.INFO)


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


def scrape_tiktok_video_links(profile_url):
    try:
        with sync_playwright() as p:
            browser = p.firefox.launch(
                headless=True, 
                # Add these options to mimic more realistic browser behavior
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--disable-site-isolation-trials'
                ]
            )
            
            # Create a new context with specific settings
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                # Disable geolocation and other permissions
                geolocation=None,
                # Optional: Add more browser context configurations
                extra_http_headers={
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
            
            page = context.new_page()
            
            # Configure page navigation and interaction
            page.set_default_timeout(30000)  # 30 seconds timeout
            page.set_viewport_size({'width': 1920, 'height': 1080})
            
            logging.info(f"Navigating to: {profile_url}")
            
            # Enhanced navigation with error handling
            try:
                response = page.goto(profile_url, wait_until='networkidle', timeout=30000)
                logging.info(f"Page load status: {response.status}")
            except Exception as nav_error:
                logging.error(f"Navigation error: {nav_error}")
                return []
            
            # Wait for network idle and potential dynamic content
            page.wait_for_load_state('networkidle', timeout=10000)
            
            # Advanced scrolling technique
            logging.info("Starting scroll to load more content")
            for _ in range(5):
                page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
                time.sleep(2)
            
            # More robust video link extraction
            try:
                video_links = page.evaluate('''() => {
                    const links = Array.from(
                        document.querySelectorAll('a[href*="/video/"]')
                    ).map(a => a.href);
                    return links;
                }''')
                
                logging.info(f"Extracted {len(video_links)} video links")
                
                # Optional: Screenshot for debugging
                #page.screenshot(path='/app/debug_screenshot.png')
                
                return video_links
            
            except Exception as extract_error:
                logging.error(f"Video link extraction error: {extract_error}")
                return []
            
            finally:
                browser.close()
    
    except Exception as overall_error:
        logging.error(f"Overall scraping error: {overall_error}")
        return []

def main():
    profile_url = "https://www.tiktok.com/@nusaaroom"
    video_links = scrape_tiktok_video_links(profile_url)
    print("Extracted Video Links:", video_links)
    vid_links = []

    # Print the video links
    print("Extracted Video Links:")
    for link in video_links:
        if is_url_classified(link):
            continue
        vid_links.append(link)

    print("vid links after filtered: ",vid_links)
    #filter check

    #pyk.save_tiktok_multi_urls(vid_links,True,'',1,save_dir=VIDEO_DIR)


if __name__ == "__main__":
    main()