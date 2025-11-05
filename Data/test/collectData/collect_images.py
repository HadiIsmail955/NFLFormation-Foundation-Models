from googleapiclient.discovery import build
import requests, os
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# === CONFIG ===
API_KEY = "AIzaSyCXI238VELpKhQevCaFbHi5_DronfXOAfU"           
CX = "c1eb50c06defd48a7"   
QUERIES = [
    "shotgun formation football",
    "i formation football",
    "pistol formation football",
    "singleback formation football",
    "empty set formation football"
]
NUM_IMAGES_PER_QUERY = 20
SAVE_DIR = "formation_images"

# === BUILD CLIENT ===
service = build("customsearch", "v1", developerKey=API_KEY)

os.makedirs(SAVE_DIR, exist_ok=True)

def download_image(url, filepath):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content))
            img.convert("RGB").save(filepath, "JPEG")
            return True
    except Exception:
        pass
    return False

# === MAIN LOOP ===
for query in QUERIES:
    print(f"\n🔍 Searching for: {query}")
    folder = os.path.join(SAVE_DIR, query.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)
    
    start_index = 1
    downloaded = 0
    while downloaded < NUM_IMAGES_PER_QUERY:
        res = service.cse().list(
            q=query,
            cx=CX,
            searchType="image",
            num=min(10, NUM_IMAGES_PER_QUERY - downloaded),
            start=start_index
        ).execute()

        if "items" not in res:
            break
        
        for item in res["items"]:
            link = item["link"]
            fname = os.path.join(folder, f"{downloaded+1:03d}.jpg")
            if download_image(link, fname):
                downloaded += 1
        start_index += 10

    print(f"✅ Collected {downloaded} images for '{query}'")
