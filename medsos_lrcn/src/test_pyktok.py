
import pyktok as pyk

pyk.specify_browser('firefox')
print(pyk.__file__)
tiktok_videos = ['https://www.tiktok.com/@balb444l/video/7444471595550248247']
pyk.save_tiktok_multi_urls(tiktok_videos,True,'tiktok_data.csv',1,save_dir="/home/arifadh/Downloads/tiktok_videos")