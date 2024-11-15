import pyktok as pyk

pyk.specify_browser('firefox')
urls=[
    "https://www.tiktok.com/@naiwen88/video/7355136826299960583",
    "https://www.tiktok.com/@naiwen88/video/7355136598129888520",
    "https://www.tiktok.com/@naiwen88/video/7355136203613687058"
]

pyk.save_tiktok_multi_urls(urls,True,'tiktok_example.csv',1)
#pyk.save_tiktok_multi_page('.aicu',ent_type='user',save_video=False,metadata_fn='tiktok.csv')
