import pyktok as pyk
from playwright.sync_api import sync_playwright
import time


pyk.specify_browser('firefox')
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
        # Check if the "Something Goes Wrong" message appears
        if page.locator("text=Something went wrong").count() > 0:
            print("Detected 'Something Goes Wrong' page. Refreshing...")
            page.reload()  # Refresh the page
            time.sleep(5)  # Wait for the page to load again

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
profile_url = "https://www.tiktok.com/@.aicu"

# Extract video links
video_links = scrape_tiktok_video_links(profile_url)

# Print the video links
print("Extracted Video Links:")
for link in video_links:
    print(link)

pyk.save_tiktok_multi_urls(video_links,True,'tiktok.csv',1)
#pyk.save_tiktok_multi_page('.aicu',ent_type='user',save_video=False,metadata_fn='tiktok.csv')
