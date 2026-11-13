"""Web loaders for Prome RAG Plugins."""

from loaders.web.crawler import WebsiteCrawler
from loaders.web.fetcher import Fetcher
from loaders.web.robots import RobotsChecker
from loaders.web.url_loader import URLLoader

__all__ = [
    "Fetcher",
    "RobotsChecker",
    "URLLoader",
    "WebsiteCrawler",
]
