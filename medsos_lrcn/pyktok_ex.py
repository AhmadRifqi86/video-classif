import pyktok as pyk
from playwright.sync_api import sync_playwright
import time
import all_config
import browser_cookie3


pyk.specify_browser('firefox')
print(pyk.__file__)

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

# TikTok profile URL
profile_url = "https://www.tiktok.com/@gisel_laaa"

# Extract video links
video_links = scrape_tiktok_video_links(profile_url)

# Print the video links
print("Extracted Video Links:")
for link in video_links:
    print(link)

pyk.save_tiktok_multi_urls(video_links[:5],True,'tiktok.csv',1,save_dir=all_config.VIDEO_DIR)
#pyk.save_tiktok_multi_page('.aicu',ent_type='user',save_video=False,metadata_fn='tiktok.csv',save_dir=all_config.VIDEO_DIR)
