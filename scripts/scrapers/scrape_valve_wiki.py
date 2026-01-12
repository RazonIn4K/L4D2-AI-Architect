#!/usr/bin/env python3
"""
Valve Developer Wiki Scraper

Collects L4D2 VScript documentation, Director options, and scripting examples
from the Valve Developer Wiki.

Usage:
    python scrape_valve_wiki.py --output data/raw/valve_wiki
"""

import os
import sys
import json
import time
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser

import subprocess

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "requests", "beautifulsoup4", "lxml"], check=True)
    import requests
    from bs4 import BeautifulSoup

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, validate_url, safe_write_json, safe_write_jsonl

# Allowed domains for Valve Wiki (SSRF prevention)
WIKI_ALLOWED_DOMAINS: Set[str] = {
    "developer.valvesoftware.com",
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
WIKI_BASE = "https://developer.valvesoftware.com"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "valve_wiki"

# Seed URLs for L4D2 scripting content
SEED_URLS = [
    "/wiki/L4D2_Vscripts",
    "/wiki/Left_4_Dead_2/Scripting/Script_Functions",
    "/wiki/Left_4_Dead_2/Scripting/Director_Scripts",
    "/wiki/L4D2_Vscript_Examples",
    "/wiki/Left_4_Dead_2/Scripting/Expanded_Mutation_System",
    "/wiki/Left_4_Dead_2/Scripting/Expanded_Mutation_System/Creating_a_Simple_Mutation",
    "/wiki/Vscript_Fundamentals",
    "/wiki/List_of_L4D2_Cvars",
    "/wiki/Left_4_Dead_2_Level_Creation",
    "/wiki/AI_Director",
    "/wiki/Navigation_Meshes",
    "/wiki/Script",
]

# URL patterns to follow
FOLLOW_PATTERNS = [
    r"/wiki/L4D2",
    r"/wiki/Left_4_Dead_2",
    r"/wiki/Vscript",
    r"/wiki/VScript",
    r"/wiki/AI_Director",
    r"/wiki/Navigation",
    r"/wiki/Script",
    r"/wiki/Squirrel",
]


@dataclass
class WikiPage:
    """Represents a scraped wiki page."""
    url: str
    title: str
    content: str
    code_blocks: List[str]
    sections: List[Dict[str, str]]
    links: List[str]
    categories: List[str]
    collected_at: str


class WikiScraper:
    """Scraper for Valve Developer Wiki."""
    
    def __init__(self, delay: float = 1.0):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "L4D2-AI-Architect/1.0 (Educational Research)"
        self.delay = delay
        self.visited: Set[str] = set()
        self.stats = {
            "pages_scraped": 0,
            "code_blocks_found": 0,
            "errors": 0,
        }
    
    def _normalize_url(self, url: str) -> str:
        """Normalize a URL to its canonical form."""
        if url.startswith("/"):
            url = urljoin(WIKI_BASE, url)
        # Remove anchors
        url = url.split("#")[0]
        return url
    
    def _should_follow(self, url: str) -> bool:
        """Check if a URL should be followed."""
        if not url.startswith(WIKI_BASE):
            return False
        
        path = urlparse(url).path
        
        # Skip non-wiki pages
        if not path.startswith("/wiki/"):
            return False
        
        # Skip special pages
        skip_patterns = [
            "/wiki/Special:",
            "/wiki/File:",
            "/wiki/Talk:",
            "/wiki/User:",
            "/wiki/Category:",
            "/w/index.php",
        ]
        for pattern in skip_patterns:
            if pattern in url:
                return False
        
        # Check if matches any follow pattern
        for pattern in FOLLOW_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                return True
        
        return False
    
    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[str]:
        """Extract code blocks from the page."""
        code_blocks = []
        
        # Pre tags
        for pre in soup.find_all("pre"):
            code = pre.get_text().strip()
            if code and len(code) > 20:  # Skip tiny snippets
                code_blocks.append(code)
        
        # Code tags within content
        for code in soup.find_all("code"):
            text = code.get_text().strip()
            if text and len(text) > 50:
                code_blocks.append(text)
        
        # Syntax highlighted blocks (various classes used on the wiki)
        for highlighted in soup.find_all(class_=re.compile(r"(mw-highlight|source|code)")):
            code = highlighted.get_text().strip()
            if code and len(code) > 20:
                code_blocks.append(code)
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for block in code_blocks:
            if block not in seen:
                seen.add(block)
                unique.append(block)
        
        return unique
    
    def _extract_sections(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract page sections with headers."""
        sections = []
        current_section = {"title": "Introduction", "content": ""}
        
        content_div = soup.find("div", {"id": "mw-content-text"})
        if not content_div:
            content_div = soup.find("div", class_="mw-parser-output")
        
        if not content_div:
            return sections
        
        for element in content_div.children:
            if not hasattr(element, "name"):
                continue
            
            # Check for headers
            if element.name in ["h1", "h2", "h3", "h4"]:
                # Save previous section
                if current_section["content"].strip():
                    sections.append(current_section)
                
                # Start new section
                header_text = element.get_text().strip()
                # Clean up header (remove [edit] links)
                header_text = re.sub(r"\[edit\]", "", header_text).strip()
                current_section = {"title": header_text, "content": ""}
            
            # Add content
            elif element.name in ["p", "pre", "ul", "ol", "dl", "table"]:
                text = element.get_text().strip()
                if text:
                    current_section["content"] += text + "\n\n"
        
        # Save last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        return sections
    
    def _extract_categories(self, soup: BeautifulSoup) -> List[str]:
        """Extract page categories."""
        categories = []
        cat_div = soup.find("div", {"id": "catlinks"})
        if cat_div:
            for link in cat_div.find_all("a"):
                cat = link.get_text().strip()
                if cat and "Category" not in cat:
                    categories.append(cat)
        return categories
    
    def _extract_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract internal wiki links."""
        links = []
        content_div = soup.find("div", {"id": "mw-content-text"})
        if not content_div:
            return links
        
        for anchor in content_div.find_all("a", href=True):
            href = anchor["href"]
            full_url = self._normalize_url(href)
            if self._should_follow(full_url):
                links.append(full_url)
        
        return list(set(links))
    
    def scrape_page(self, url: str) -> Optional[WikiPage]:
        """Scrape a single wiki page."""
        url = self._normalize_url(url)
        
        if url in self.visited:
            return None
        
        self.visited.add(url)
        logger.info(f"Scraping: {url}")

        try:
            # Validate URL to prevent SSRF attacks
            validated_url = validate_url(url, WIKI_ALLOWED_DOMAINS)
            response = self.session.get(validated_url, timeout=30)
            validate_url(response.url, WIKI_ALLOWED_DOMAINS)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "lxml")
            
            # Extract title
            title_elem = soup.find("h1", {"id": "firstHeading"})
            title = title_elem.get_text().strip() if title_elem else "Unknown"
            
            # Extract main content
            content_div = soup.find("div", {"id": "mw-content-text"})
            content = content_div.get_text() if content_div else ""
            
            # Clean content
            content = re.sub(r"\n{3,}", "\n\n", content)
            content = content.strip()
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(soup)
            self.stats["code_blocks_found"] += len(code_blocks)
            
            # Extract sections
            sections = self._extract_sections(soup)
            
            # Extract links for crawling
            links = self._extract_links(soup)
            
            # Extract categories
            categories = self._extract_categories(soup)
            
            self.stats["pages_scraped"] += 1
            
            return WikiPage(
                url=url,
                title=title,
                content=content,
                code_blocks=code_blocks,
                sections=sections,
                links=links,
                categories=categories,
                collected_at=datetime.utcnow().isoformat(),
            )
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            self.stats["errors"] += 1
            return None
    
    def crawl(self, max_pages: int = 200) -> List[WikiPage]:
        """Crawl wiki starting from seed URLs."""
        max_pages = int(max_pages)
        if max_pages < 1:
            return []
        max_pages = min(max_pages, 5000)

        pages = []
        queue = [self._normalize_url(url) for url in SEED_URLS]
        
        while queue and len(pages) < max_pages:
            url = queue.pop(0)
            
            if url in self.visited:
                continue
            
            page = self.scrape_page(url)
            if page:
                pages.append(page)
                
                # Add new links to queue
                for link in page.links:
                    if link not in self.visited:
                        queue.append(link)
            
            # Rate limiting
            time.sleep(self.delay)
            
            # Progress logging
            if len(pages) % 10 == 0:
                logger.info(f"Progress: {len(pages)} pages, {len(queue)} in queue")
        
        return pages


def save_results(pages: List[WikiPage], output_dir: Path, project_root: Path) -> None:
    """Save scraped pages to disk with path traversal protection."""
    # Validate output directory
    safe_output_dir = safe_path(str(output_dir), project_root, create_parents=True)

    # Save as JSONL using safe_write_jsonl
    page_items = [asdict(page) for page in pages]
    jsonl_path = safe_write_jsonl(
        str(safe_output_dir / "valve_wiki.jsonl"),
        page_items,
        project_root
    )
    logger.info(f"Saved {len(pages)} pages to {jsonl_path}")

    # Save code blocks separately for easier processing
    code_items = []
    for page in pages:
        for code in page.code_blocks:
            entry = {
                "source_url": page.url,
                "source_title": page.title,
                "code": code,
                "categories": page.categories,
            }
            code_items.append(entry)

    code_path = safe_write_jsonl(
        str(safe_output_dir / "code_blocks.jsonl"),
        code_items,
        project_root
    )
    logger.info(f"Saved code blocks to {code_path}")

    # Save statistics using safe_write_json
    stats = {
        "total_pages": len(pages),
        "total_code_blocks": sum(len(p.code_blocks) for p in pages),
        "total_sections": sum(len(p.sections) for p in pages),
        "pages_by_category": {},
    }

    for page in pages:
        for cat in page.categories:
            stats["pages_by_category"][cat] = stats["pages_by_category"].get(cat, 0) + 1

    safe_write_json(
        str(safe_output_dir / "scrape_stats.json"),
        stats,
        project_root
    )


def main():
    parser = argparse.ArgumentParser(description="Scrape L4D2 docs from Valve Developer Wiki")
    parser.add_argument("--max-pages", type=int, default=200,
                        help="Maximum pages to scrape")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between requests (seconds)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = WikiScraper(delay=args.delay)
    
    # Run crawl
    logger.info("Starting Valve Wiki scrape...")
    start_time = time.time()
    
    try:
        max_pages = int(args.max_pages)
    except (TypeError, ValueError):
        logger.error("Invalid --max-pages")
        sys.exit(2)

    if max_pages < 1 or max_pages > 5000:
        logger.error("--max-pages must be between 1 and 5000")
        sys.exit(2)

    pages = scraper.crawl(max_pages=max_pages)

    # Save results with path validation
    project_root = Path(__file__).parent.parent.parent
    output_dir = safe_path(args.output, project_root, create_parents=True)
    save_results(pages, output_dir, project_root)
    
    # Print summary
    elapsed = time.time() - start_time
    logger.info("=" * 50)
    logger.info("SCRAPE COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info(f"Pages scraped: {scraper.stats['pages_scraped']}")
    logger.info(f"Code blocks found: {scraper.stats['code_blocks_found']}")
    logger.info(f"Errors: {scraper.stats['errors']}")


if __name__ == "__main__":
    main()
