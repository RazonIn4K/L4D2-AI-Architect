#!/usr/bin/env python3
"""
GitHub SourceMod Plugins Scraper

Collects SourcePawn (.sp) and include (.inc) files from GitHub repositories
tagged with 'sourcemod', 'sourcemod-plugins', 'sourcepawn', or 'l4d2'.

Usage:
    python scrape_github_plugins.py --token YOUR_GITHUB_TOKEN --max-repos 500
"""

import os
import sys
import json
import time
import base64
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system(f"{sys.executable} -m pip install requests")
    import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scraper.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
GITHUB_API_BASE = "https://api.github.com"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "github_plugins"
SEARCH_TOPICS = [
    "sourcemod-plugins",
    "sourcemod",
    "sourcepawn",
    "l4d2",
    "left4dead2",
    "l4d2-plugins",
]

# File extensions to collect
TARGET_EXTENSIONS = {".sp", ".inc", ".nut", ".nuc"}


@dataclass
class CodeFile:
    """Represents a collected code file."""
    repo_name: str
    repo_url: str
    file_path: str
    file_name: str
    extension: str
    content: str
    language: str
    stars: int
    forks: int
    description: Optional[str]
    license: Optional[str]
    collected_at: str


class GitHubScraper:
    """Scraper for GitHub SourceMod repositories."""
    
    def __init__(self, token: Optional[str] = None, rate_limit_wait: int = 60):
        self.session = requests.Session()
        self.rate_limit_wait = rate_limit_wait
        
        if token:
            self.session.headers["Authorization"] = f"token {token}"
            logger.info("Using authenticated GitHub API")
        else:
            logger.warning("No token provided - rate limits will be strict (60 req/hr)")
        
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        self.session.headers["User-Agent"] = "L4D2-AI-Architect-Scraper"
        
        # Statistics
        self.stats = {
            "repos_scanned": 0,
            "files_collected": 0,
            "bytes_collected": 0,
            "errors": 0,
            "rate_limit_hits": 0,
        }
    
    def _check_rate_limit(self, response: requests.Response) -> None:
        """Check and handle rate limiting."""
        remaining = int(response.headers.get("X-RateLimit-Remaining", 1))
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        
        if remaining == 0:
            wait_time = max(reset_time - time.time(), 0) + 5
            logger.warning(f"Rate limit hit! Waiting {wait_time:.0f}s...")
            self.stats["rate_limit_hits"] += 1
            time.sleep(wait_time)
        elif remaining < 10:
            logger.info(f"Rate limit low: {remaining} remaining")
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a rate-limit-aware request."""
        try:
            response = self.session.get(url, params=params, timeout=30)
            self._check_rate_limit(response)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                logger.warning("Forbidden - likely rate limited")
                time.sleep(self.rate_limit_wait)
                return None
            elif response.status_code == 404:
                return None
            else:
                logger.error(f"Request failed: {response.status_code} - {url}")
                self.stats["errors"] += 1
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            self.stats["errors"] += 1
            return None
    
    def search_repositories(self, topic: str, max_repos: int = 100) -> List[Dict]:
        """Search for repositories by topic."""
        repos = []
        page = 1
        per_page = min(100, max_repos)
        
        while len(repos) < max_repos:
            url = f"{GITHUB_API_BASE}/search/repositories"
            params = {
                "q": f"topic:{topic}",
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            }
            
            data = self._make_request(url, params)
            if not data or "items" not in data:
                break
            
            items = data["items"]
            if not items:
                break
            
            repos.extend(items)
            logger.info(f"Found {len(items)} repos for topic '{topic}' (page {page})")
            
            page += 1
            time.sleep(1)  # Be nice to GitHub
        
        return repos[:max_repos]
    
    def get_repo_tree(self, owner: str, repo: str, branch: str = "main") -> Optional[List[Dict]]:
        """Get the file tree of a repository."""
        # Try main branch first, then master
        for try_branch in [branch, "master", "main"]:
            url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/git/trees/{try_branch}"
            params = {"recursive": "1"}
            
            data = self._make_request(url, params)
            if data and "tree" in data:
                return data["tree"]
        
        return None
    
    def get_file_content(self, owner: str, repo: str, path: str) -> Optional[str]:
        """Get the content of a file from a repository."""
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
        
        data = self._make_request(url)
        if not data:
            return None
        
        # Handle file content
        if data.get("encoding") == "base64" and data.get("content"):
            try:
                content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                return content
            except Exception as e:
                logger.error(f"Failed to decode {path}: {e}")
                return None
        
        return None
    
    def collect_from_repo(self, repo_info: Dict) -> List[CodeFile]:
        """Collect all target files from a repository."""
        owner = repo_info["owner"]["login"]
        repo = repo_info["name"]
        full_name = repo_info["full_name"]
        
        logger.info(f"Scanning {full_name}...")
        self.stats["repos_scanned"] += 1
        
        # Get file tree
        tree = self.get_repo_tree(owner, repo)
        if not tree:
            logger.warning(f"Could not get tree for {full_name}")
            return []
        
        # Filter for target files
        target_files = [
            item for item in tree
            if item["type"] == "blob" and 
            Path(item["path"]).suffix.lower() in TARGET_EXTENSIONS
        ]
        
        if not target_files:
            return []
        
        logger.info(f"Found {len(target_files)} target files in {full_name}")
        
        collected = []
        for file_info in target_files:
            path = file_info["path"]
            extension = Path(path).suffix.lower()
            
            # Get file content
            content = self.get_file_content(owner, repo, path)
            if not content:
                continue
            
            # Skip very small or very large files
            if len(content) < 50 or len(content) > 500000:
                continue
            
            # Determine language
            language = "sourcepawn" if extension in {".sp", ".inc"} else "squirrel"
            
            code_file = CodeFile(
                repo_name=full_name,
                repo_url=repo_info["html_url"],
                file_path=path,
                file_name=Path(path).name,
                extension=extension,
                content=content,
                language=language,
                stars=repo_info.get("stargazers_count", 0),
                forks=repo_info.get("forks_count", 0),
                description=repo_info.get("description"),
                license=repo_info.get("license", {}).get("name") if repo_info.get("license") else None,
                collected_at=datetime.utcnow().isoformat(),
            )
            
            collected.append(code_file)
            self.stats["files_collected"] += 1
            self.stats["bytes_collected"] += len(content)
            
            # Rate limiting
            time.sleep(0.5)
        
        return collected
    
    def run(self, max_repos: int = 500) -> List[CodeFile]:
        """Run the full scraping process."""
        all_repos = []
        seen_repos = set()
        
        # Search all topics
        for topic in SEARCH_TOPICS:
            logger.info(f"Searching topic: {topic}")
            repos = self.search_repositories(topic, max_repos // len(SEARCH_TOPICS))
            
            for repo in repos:
                if repo["full_name"] not in seen_repos:
                    all_repos.append(repo)
                    seen_repos.add(repo["full_name"])
        
        logger.info(f"Found {len(all_repos)} unique repositories")
        
        # Sort by stars (most popular first)
        all_repos.sort(key=lambda x: x.get("stargazers_count", 0), reverse=True)
        
        # Collect files
        all_files = []
        for repo in all_repos[:max_repos]:
            try:
                files = self.collect_from_repo(repo)
                all_files.extend(files)
                
                # Periodic save
                if len(all_files) % 100 == 0:
                    logger.info(f"Progress: {len(all_files)} files collected")
                    
            except Exception as e:
                logger.error(f"Error processing {repo['full_name']}: {e}")
                self.stats["errors"] += 1
        
        return all_files


def save_results(files: List[CodeFile], output_dir: Path) -> None:
    """Save collected files to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL (one file per line)
    jsonl_path = output_dir / "github_plugins.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for code_file in files:
            f.write(json.dumps(asdict(code_file), ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(files)} files to {jsonl_path}")
    
    # Save statistics
    stats_path = output_dir / "scrape_stats.json"
    stats = {
        "total_files": len(files),
        "total_bytes": sum(len(f.content) for f in files),
        "by_extension": {},
        "by_language": {},
        "top_repos": [],
    }
    
    for f in files:
        stats["by_extension"][f.extension] = stats["by_extension"].get(f.extension, 0) + 1
        stats["by_language"][f.language] = stats["by_language"].get(f.language, 0) + 1
    
    # Top repos by file count
    repo_counts = {}
    for f in files:
        repo_counts[f.repo_name] = repo_counts.get(f.repo_name, 0) + 1
    
    stats["top_repos"] = sorted(repo_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved statistics to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape SourceMod plugins from GitHub")
    parser.add_argument("--token", type=str, default=os.environ.get("GITHUB_TOKEN"),
                        help="GitHub API token (or set GITHUB_TOKEN env var)")
    parser.add_argument("--max-repos", type=int, default=500,
                        help="Maximum repositories to scan")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    
    args = parser.parse_args()
    
    if not args.token:
        logger.warning("No GitHub token provided. Rate limits will be very strict!")
        logger.info("Get a token at: https://github.com/settings/tokens")
    
    # Initialize scraper
    scraper = GitHubScraper(token=args.token)
    
    # Run collection
    logger.info("Starting GitHub scrape...")
    start_time = time.time()
    
    files = scraper.run(max_repos=args.max_repos)
    
    # Save results
    output_dir = Path(args.output)
    save_results(files, output_dir)
    
    # Print summary
    elapsed = time.time() - start_time
    logger.info("=" * 50)
    logger.info("SCRAPE COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info(f"Repos scanned: {scraper.stats['repos_scanned']}")
    logger.info(f"Files collected: {scraper.stats['files_collected']}")
    logger.info(f"Bytes collected: {scraper.stats['bytes_collected']:,}")
    logger.info(f"Errors: {scraper.stats['errors']}")
    logger.info(f"Rate limit hits: {scraper.stats['rate_limit_hits']}")


if __name__ == "__main__":
    main()
