import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import quote
from PIL import Image
from io import BytesIO

# === CONFIGURATION ===
OUTPUT_DIR = "card_grader/image_process/images"
METADATA_PATH = "card_grader/image_process/metadata.csv"
NUM_PAGES = 2  # Number of pages per grade to scrape

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_image_urls(grade, num_pages=1):
    base_url = "https://www.ebay.com/sch/i.html?_nkw={query}&_sop=12&_pgn={page}"
    image_urls = []
    listings = []

    for page in range(1, num_pages + 1):
        query = quote(f"psa {grade} pokemon card")
        url = base_url.format(query=query, page=page)
        print(f"📄 Scraping grade {grade} - page {page}: {url}")

        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            items = soup.select("li.s-item")

            for item in items:
                img_tag = item.select_one("img")
                title_tag = item.select_one(".s-item__title")
                link_tag = item.select_one("a.s-item__link")

                if not img_tag or not img_tag.get("src"):
                    continue

                image_url = img_tag["src"]
                title = title_tag.text if title_tag else ""
                url = link_tag["href"] if link_tag else ""

                listings.append({"image_url": image_url, "title": title, "grade": grade, "url": url})
        except Exception as e:
            print(f"❌ Failed to scrape page {page} for grade {grade}: {e}")

    return listings

def download_images_and_save_metadata(all_listings):
    metadata = []
    for i, listing in enumerate(all_listings):
        try:
            response = requests.get(listing["image_url"], timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")

            # Generate safe filename
            safe_title = "".join(c for c in listing["title"] if c.isalnum() or c in (" ", "_")).rstrip()
            filename = f"{listing['grade']}_{safe_title[:80].strip().replace(' ', '_')}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            img.save(filepath, "JPEG")

            metadata.append({"filepath": filepath, "title": listing["title"], "grade": listing["grade"], "url": listing["url"]})
        except Exception as e:
            print(f"❌ Failed to download image {listing['image_url']}: {e}")

    df = pd.DataFrame(metadata)
    df.to_csv(METADATA_PATH, index=False)
    print(f"\n✅ Metadata saved to {METADATA_PATH}")

if __name__ == "__main__":
    all_listings = []
    for grade in range(1, 11):
        all_listings.extend(fetch_image_urls(grade, NUM_PAGES))
    download_images_and_save_metadata(all_listings)
