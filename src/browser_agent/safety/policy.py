from __future__ import annotations

from urllib.parse import urlparse

from browser_agent.safety.types import SafetyDecision


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def check_navigation(
    url: str, allow_hosts: list[str], block_hosts: list[str]
) -> SafetyDecision | None:
    """Return a deny decision if navigation violates the lists, else None.

    Stub policy: substring match on host. Phase 4 swaps in category-based
    block lists (financial/adult/sensitive) by default.
    """
    host = host_of(url)
    if not host:
        return SafetyDecision(allow=False, reason=f"could not parse host from {url!r}")

    for blocked in block_hosts:
        if blocked in host:
            return SafetyDecision(allow=False, reason=f"host {host!r} is on the blocklist")

    if allow_hosts and not any(allowed in host for allowed in allow_hosts):
        return SafetyDecision(allow=False, reason=f"host {host!r} is not on the allowlist")

    return None
