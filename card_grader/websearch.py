import os
import re
import csv
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from urllib.request import urlretrieve
import subprocess

SAVE_DIR = "card_grader/image_process/images"
CSV_FILE = "card_grader/image_process/metadata.csv"
PREPROCESS_PATH = "card_grader/image_process/preprocess.py"

os.makedirs(SAVE_DIR, exist_ok=True)

def scrape_ebay_cards(search_term="psa pokemon card", max_pages=3):
    results = []
    base_url = "https://www.ebay.com/sch/i.html?_nkw={}&_sop=12&_pgn={}"

    for page in range(1, max_pages + 1):
        url = base_url.format(quote(search_term), page)
        print(f"\n📄 Scraping page {page}: {url}")
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, "html.parser")
            items = soup.select("li.s-item")
        except Exception as e:
            print(f"❌ Failed to fetch or parse page {page}: {e}")
            continue

        found = False

        for item in items:
            title_tag = item.select_one(".s-item__title")
            link_tag = item.select_one("a.s-item__link")
            img_tag = item.select_one("img")

            if not title_tag or not img_tag:
                continue

            title = title_tag.get_text(strip=True)
            link = link_tag.get("href") if link_tag else None

            # Fallback order for image sources
            image_url = (
                img_tag.get("src") or
                img_tag.get("data-src") or
                img_tag.get("data-img-src")
            )

            grade_match = re.search(r'\bpsa\s?(\d{1,2})\b', title, re.IGNORECASE)
            if not grade_match or not image_url:
                continue

            grade = int(grade_match.group(1))
            filename = f"{grade}_" + re.sub(r'[^a-zA-Z0-9]+', '_', title)[:50] + ".jpg"
            filepath = os.path.join(SAVE_DIR, filename)

            # Attempt to use higher resolution
            image_url = re.sub(r"s-l\d+\.webp", "s-l1600.webp", image_url)
            image_url = re.sub(r"s-l\d+\.jpg", "s-l1600.jpg", image_url)

            try:
                urlretrieve(image_url, filepath)
                print(f"💾 Saved: {os.path.relpath(filepath)}")
                results.append((filepath, grade, title, link))
                found = True
            except Exception as e:
                print(f"❌ Failed to save {image_url}: {e}")

        if not found:
            print("⚠️ No listings found.")

        time.sleep(1)

    return results

def save_metadata(results):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "grade", "title", "url"])
        writer.writerows(results)
    print(f"\n✅ Metadata saved to {CSV_FILE}")

def run_preprocess():
    print("\n🚀 Running preprocess.py...")
    result = subprocess.run(["python3", PREPROCESS_PATH])
    if result.returncode != 0:
        print("❌ Preprocessing failed.")

if __name__ == "__main__":
    data = scrape_ebay_cards()
    if data:
        save_metadata(data)
        run_preprocess()
    else:
        print("⚠️ No data scraped.")
