"""Filters for directory and file operations in Prome RAG Plugins loaders."""

from __future__ import annotations


class ExtensionFilter:
    """Filter files by extension allowlist/blocklist."""

    def __init__(self, allowed_extensions: set[str] | None = None, blocked_extensions: set[str] | None = None):
        self.allowed_extensions = {ext.lower() for ext in (allowed_extensions or set())}
        self.blocked_extensions = {ext.lower() for ext in (blocked_extensions or set())}

    def accept(self, extension: str) -> bool:
        """Check if an extension should be accepted.

        Returns True if the extension is allowed (and not blocked).
        """
        ext = extension.lower()
        # If there are allowed extensions, only those are accepted
        if self.allowed_extensions:
            return ext in self.allowed_extensions and ext not in self.blocked_extensions
        # Otherwise block the blocked extensions
        return ext not in self.blocked_extensions


class ExclusionFilter:
    """Filter files by name patterns (glob-style)."""

    def __init__(self, patterns: list[str] | None = None):
        self.patterns = patterns or []
        # Compile simple patterns for efficiency
        self._compiled = [self._compile_pattern(p) for p in self.patterns]

    def _compile_pattern(self, pattern: str) -> str:
        """Convert a glob pattern to a match-friendly string."""
        # Simple conversion: * matches anything, ? matches single char
        # For production, use fnmatch, but keep it simple here
        return pattern

    def is_excluded(self, name: str) -> bool:
        """Check if a filename matches exclusion patterns."""
        for pattern in self._compiled:
            if self._matches(pattern, name):
                return True
        return False

    @staticmethod
    def _matches(pattern: str, name: str) -> bool:
        """Check if a name matches a pattern."""
        import fnmatch

        return fnmatch.fnmatch(name, pattern)


class HiddenFilePolicy:
    """Policy for handling hidden files (those starting with '.')."""

    EXCLUDE = "exclude"
    INCLUDE = "include"
    ASK = "ask"


class SymlinkPolicy:
    """Policy for handling symlinks."""

    REJECT = "reject"  # Exclude all symlinks
    FOLLOW = "follow"  # Resolve and follow symlinks
    PERMISSIVE = "permissive"  # Allow symlinks within permitted roots
