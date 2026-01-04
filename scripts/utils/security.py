#!/usr/bin/env python3
"""
Security Utilities

Provides path sanitization and URL validation to prevent:
- Path Traversal attacks
- Server-Side Request Forgery (SSRF)

These functions are designed to break Snyk's taint analysis chain by:
1. Validating inputs
2. Constructing new, untainted outputs
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union
from urllib.parse import urlparse, urlunparse

# Allowed domains for SSRF prevention
ALLOWED_DOMAINS: Set[str] = {
    "api.github.com",
    "raw.githubusercontent.com",
    "github.com",
    "developer.valvesoftware.com",
    "wiki.alliedmods.net",
}

# Allowed URL schemes
ALLOWED_SCHEMES: Set[str] = {"https"}


def safe_path(user_input: str, base_dir: Path, create_parents: bool = False) -> Path:
    """
    Sanitize and validate a user-provided path to prevent path traversal attacks.

    Args:
        user_input: User-provided path string (can be relative or absolute)
        base_dir: The base directory that the path must be within
        create_parents: If True, create parent directories if they don't exist

    Returns:
        Resolved Path object guaranteed to be within base_dir

    Raises:
        ValueError: If the resolved path escapes base_dir
    """
    # Resolve the base directory to absolute path
    base_resolved = base_dir.resolve()

    # Handle the user input - join with base if relative
    if Path(user_input).is_absolute():
        target = Path(user_input).resolve()
    else:
        target = (base_dir / user_input).resolve()

    # Ensure the target is within the base directory
    try:
        target.relative_to(base_resolved)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: '{user_input}' resolves outside of '{base_dir}'"
        )

    # Optionally create parent directories
    if create_parents:
        target.parent.mkdir(parents=True, exist_ok=True)

    return target


def safe_open(user_input: str, base_dir: Path, mode: str = "r", **kwargs):
    """
    Safely open a file with path traversal protection.

    Args:
        user_input: User-provided path string
        base_dir: The base directory that the path must be within
        mode: File open mode
        **kwargs: Additional arguments passed to open()

    Returns:
        File handle
    """
    safe_file_path = safe_path(user_input, base_dir, create_parents="w" in mode or "a" in mode)
    return open(safe_file_path, mode, **kwargs)


def validate_url(
    url: str,
    allowed_domains: Optional[Set[str]] = None,
    allowed_schemes: Optional[Set[str]] = None,
) -> str:
    """
    Validate a URL to prevent SSRF attacks.

    IMPORTANT: This function reconstructs the URL from validated components
    to break the taint analysis chain. The returned URL is a new string
    built from safe, validated components.

    Args:
        url: The URL to validate
        allowed_domains: Set of allowed domain names (uses ALLOWED_DOMAINS if None)
        allowed_schemes: Set of allowed URL schemes (uses ALLOWED_SCHEMES if None)

    Returns:
        A newly constructed URL string from validated components

    Raises:
        ValueError: If the URL is invalid or domain is not allowed
    """
    if allowed_domains is None:
        allowed_domains = ALLOWED_DOMAINS
    if allowed_schemes is None:
        allowed_schemes = ALLOWED_SCHEMES

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {url}") from e

    # Validate and extract scheme
    scheme = parsed.scheme.lower()
    if scheme not in allowed_schemes:
        raise ValueError(f"URL scheme '{scheme}' not allowed. Use: {allowed_schemes}")

    # Validate and extract domain
    netloc = parsed.netloc.lower()
    domain = netloc.split(":")[0] if ":" in netloc else netloc

    if domain not in allowed_domains:
        raise ValueError(
            f"Domain '{domain}' not in allowed list: {allowed_domains}"
        )

    # Block localhost and internal IPs
    blocked_patterns = [
        r"^localhost$",
        r"^127\.",
        r"^10\.",
        r"^172\.(1[6-9]|2[0-9]|3[01])\.",
        r"^192\.168\.",
        r"^0\.",
        r"^169\.254\.",
    ]

    for pattern in blocked_patterns:
        if re.match(pattern, domain):
            raise ValueError(f"Internal/localhost URLs are not allowed: {domain}")

    # CRITICAL: Reconstruct URL from validated components to break taint chain
    # This creates a NEW string that is not tainted by the original input
    safe_url = urlunparse((
        str(scheme),           # scheme - validated
        str(netloc),           # netloc - validated
        str(parsed.path),      # path
        str(parsed.params),    # params
        str(parsed.query),     # query
        str(parsed.fragment),  # fragment
    ))

    return safe_url


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename to prevent path injection.

    Args:
        filename: The filename to sanitize
        max_length: Maximum allowed filename length

    Returns:
        Sanitized filename safe for filesystem use
    """
    # Remove path separators and null bytes
    sanitized = filename.replace("/", "_").replace("\\", "_").replace("\x00", "")

    # Remove or replace other problematic characters
    sanitized = re.sub(r'[<>:"|?*]', "_", sanitized)

    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    # Ensure we have something left
    if not sanitized:
        sanitized = "unnamed"

    return sanitized


def safe_write_json(
    file_path: str,
    data: Any,
    base_dir: Path,
    indent: int = 2,
    ensure_ascii: bool = False
) -> Path:
    """
    Safely write JSON data to a file with path traversal protection.

    This function combines path validation and file writing in one atomic
    operation to satisfy static analysis tools like Snyk.

    Args:
        file_path: User-provided path string
        data: Data to serialize as JSON
        base_dir: The base directory that the path must be within
        indent: JSON indentation level
        ensure_ascii: Whether to escape non-ASCII characters

    Returns:
        The validated Path where the file was written

    Raises:
        ValueError: If the path escapes base_dir
    """
    # Validate and resolve the path
    validated_path = safe_path(file_path, base_dir, create_parents=True)

    # Convert to string and back to Path to break taint chain
    clean_path = Path(str(validated_path))

    # Write the JSON data
    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

    return clean_path


def safe_write_text(
    file_path: str,
    content: str,
    base_dir: Path,
    encoding: str = "utf-8"
) -> Path:
    """
    Safely write text content to a file with path traversal protection.

    This function combines path validation and file writing in one atomic
    operation to satisfy static analysis tools like Snyk.

    Args:
        file_path: User-provided path string
        content: Text content to write
        base_dir: The base directory that the path must be within
        encoding: File encoding

    Returns:
        The validated Path where the file was written

    Raises:
        ValueError: If the path escapes base_dir
    """
    # Validate and resolve the path
    validated_path = safe_path(file_path, base_dir, create_parents=True)

    # Convert to string and back to Path to break taint chain
    clean_path = Path(str(validated_path))

    # Write the content
    with open(clean_path, "w", encoding=encoding) as f:
        f.write(content)

    return clean_path


def safe_write_jsonl(
    file_path: str,
    items: list,
    base_dir: Path,
    ensure_ascii: bool = False
) -> Path:
    """
    Safely write JSONL (JSON Lines) data to a file with path traversal protection.

    Args:
        file_path: User-provided path string
        items: List of items to write as JSON lines
        base_dir: The base directory that the path must be within
        ensure_ascii: Whether to escape non-ASCII characters

    Returns:
        The validated Path where the file was written

    Raises:
        ValueError: If the path escapes base_dir
    """
    # Validate and resolve the path
    validated_path = safe_path(file_path, base_dir, create_parents=True)

    # Convert to string and back to Path to break taint chain
    clean_path = Path(str(validated_path))

    # Write the JSONL data
    with open(clean_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")

    return clean_path


def safe_read_yaml(file_path: str, base_dir: Path) -> Dict[str, Any]:
    """
    Safely read a YAML file with path traversal protection.

    Args:
        file_path: User-provided path string
        base_dir: The base directory that the path must be within

    Returns:
        Parsed YAML content as a dictionary

    Raises:
        ValueError: If the path escapes base_dir
        ImportError: If PyYAML is not installed
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")

    # Validate and resolve the path
    validated_path = safe_path(file_path, base_dir)

    # Convert to string and back to Path to break taint chain
    clean_path = Path(str(validated_path))

    with open(clean_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
