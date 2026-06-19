from __future__ import annotations

from urllib.parse import urlparse

from browser_agent.safety.types import SafetyDecision


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _matches(pattern: str, host: str) -> bool:
    """Match a host pattern against a hostname using suffix matching.

    'example.com' matches 'example.com' and 'www.example.com'.
    '.example.com' matches only subdomains of example.com.
    """
    if pattern.startswith("."):
        return host.endswith(pattern) or host == pattern[1:]
    return host == pattern or host.endswith("." + pattern)


def check_navigation(
    url: str, allow_hosts: list[str], block_hosts: list[str]
) -> SafetyDecision | None:
    """Return a deny decision if navigation violates the lists, else None.

    Uses suffix-based domain matching: 'example.com' matches 'www.example.com'
    but 'mail' does not match 'gmail.com'.
    """
    host = host_of(url)
    if not host:
        return SafetyDecision(allow=False, reason=f"could not parse host from {url!r}")

    for blocked in block_hosts:
        if _matches(blocked, host):
            return SafetyDecision(allow=False, reason=f"host {host!r} is on the blocklist")

    if allow_hosts and not any(_matches(allowed, host) for allowed in allow_hosts):
        return SafetyDecision(allow=False, reason=f"host {host!r} is not on the allowlist")

    return None
