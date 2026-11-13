"""Loader capabilities description."""

from __future__ import annotations

from loaders.contracts.models import ContentType


class LoaderCapabilities:
    """Describe what a loader can do."""

    def __init__(
        self,
        supports_local_files: bool = True,
        supports_local_directories: bool = True,
        supports_urls: bool = True,
        supports_crawling: bool = False,
        max_file_size: int | None = None,
        max_directory_files: int | None = None,
        allowed_extensions: tuple[str, ...] = (),
        supported_content_types: tuple[ContentType, ...] = (),
    ):
        self.supports_local_files = supports_local_files
        self.supports_local_directories = supports_local_directories
        self.supports_urls = supports_urls
        self.supports_crawling = supports_crawling
        self.max_file_size = max_file_size
        self.max_directory_files = max_directory_files
        self.allowed_extensions = allowed_extensions
        self.supported_content_types = supported_content_types
