# ---- app/scraping.py ----
import requests
from bs4 import BeautifulSoup
from app.utils import clean_text
from app import config

async def scrape_url(url: str, depth: int, visited: set) -> list:
    if depth < 1 or url in visited:
        return []
    visited.add(url)
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    raw_paras = [p.get_text(strip=True) for p in soup.find_all('p')]
    paras = [(clean_text(p), url, depth) for p in raw_paras if clean_text(p)]

    links = [a['href'] for a in soup.find_all('a', href=True)]
    for href in set(links):
        if href.startswith('/'):
            href = config.BASE_DOMAIN + href
        if href.startswith(config.BASE_DOMAIN):
            paras += await scrape_url(href, depth - 1, visited)

    return paras