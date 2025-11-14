"""
classifier.py
~~~~~~~~~~~~~~~~

This module implements a simple rule‑based classifier for Gmail messages.  It assigns
each email to one of the following categories: Primary, Social, Promotions,
Updates, Forums or Work.  The heuristics are based on Gmail’s own description
of how it sorts messages into categories【617583219031976†L24-L35】, plus a custom
work category.

To add or modify rules, adjust the keyword and domain lists below.  The classifier
expects basic message information: the sender’s address, the subject line,
the message snippet and optionally the list of headers.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


# Known social domains (social networking and media‑sharing sites)
SOCIAL_DOMAINS = {
    "facebook.com",
    "twitter.com",
    "linkedin.com",
    "instagram.com",
    "pinterest.com",
    "snapchat.com",
    "tiktok.com",
    "tumblr.com",
    "reddit.com",
    "discord.com",
    "slack.com",
    "meetup.com",
}

# Free email providers for personal accounts
FREE_EMAIL_PROVIDERS = {
    "gmail.com",
    "googlemail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "live.com",
    "msn.com",
    "aol.com",
    "icloud.com",
    "protonmail.com",
    "yandex.com",
    "zoho.com",
}

# Keywords indicating promotional content
PROMOTION_KEYWORDS = {
    "sale", "deal", "discount", "offer", "promo", "coupon",
    "subscribe", "subscription", "save", "free", "limited time",
    "special", "clearance", "bargain", "order now", "unlimited",
}

# Keywords indicating updates (receipts, confirmations, reminders)
UPDATES_KEYWORDS = {
    "invoice", "receipt", "order", "confirmation", "confirm",
    "shipping", "shipment", "tracking", "itinerary", "bill",
    "payment", "statement", "notice", "reminder", "ticket",
    "appointment", "reservation", "booking", "schedule",
}

# Keywords indicating forum or mailing‑list digests
FORUM_KEYWORDS = {
    "digest", "discussion", "forum", "mailing list", "newsletter",
    "community", "thread", "re: [", "[list]", "[forum]",
}

# Social keywords (for subject or snippet matching)
SOCIAL_KEYWORDS = {
    "new follower", "friend request", "mentioned you", "tagged you",
    "commented", "liked your", "shared", "join us",
}


def extract_domain(email_address: str) -> str:
    """Return the domain part of an email address in lower case."""
    if not email_address:
        return ""
    parts = email_address.split("@")
    if len(parts) != 2:
        return ""
    return parts[1].lower().strip()


def has_any_keyword(text: str, keywords: set[str]) -> bool:
    """Return True if any keyword appears in the given text (case‑insensitive)."""
    text_lower = (text or "").lower()
    return any(keyword in text_lower for keyword in keywords)


def classify_email(
    from_address: Optional[str],
    subject: Optional[str],
    snippet: Optional[str],
    headers: Optional[Dict[str, str]] = None,
) -> str:
    """Classify an email into one of the supported categories.

    Parameters
    ----------
    from_address : str
        The sender’s email address (can be None).
    subject : str
        The subject line of the email (can be None).
    snippet : str
        A short snippet of the email body (can be None).
    headers : dict or None
        Optional mapping of header names to values.  Used for detecting
        `List‑Id` (forums).

    Returns
    -------
    str
        One of 'Social', 'Promotions', 'Updates', 'Forums', 'Work' or 'Primary'.
    """
    # Normalize inputs
    subject = subject or ""
    snippet = snippet or ""
    domain = extract_domain(from_address or "")

    # 1. Social: check domain list and keywords
    if domain in SOCIAL_DOMAINS:
        return "Social"
    if has_any_keyword(subject, SOCIAL_KEYWORDS) or has_any_keyword(snippet, SOCIAL_KEYWORDS):
        return "Social"

    # 2. Promotions: check for marketing keywords or marketing domains
    if has_any_keyword(subject, PROMOTION_KEYWORDS) or has_any_keyword(snippet, PROMOTION_KEYWORDS):
        return "Promotions"
    # common marketing domains patterns
    if any(domain.endswith(suffix) for suffix in {"mailchimp.com", "sendgrid.net", "emarketing.com"}):
        return "Promotions"

    # 3. Updates: keywords typical of confirmations and receipts
    if has_any_keyword(subject, UPDATES_KEYWORDS) or has_any_keyword(snippet, UPDATES_KEYWORDS):
        return "Updates"

    # 4. Forums: presence of List‑Id header or forum keywords
    if headers:
        list_id = headers.get("List-Id") or headers.get("List-id") or headers.get("List-ID")
        if list_id:
            return "Forums"
    if has_any_keyword(subject, FORUM_KEYWORDS) or has_any_keyword(snippet, FORUM_KEYWORDS):
        return "Forums"

    # 5. Work: heuristically label messages from non‑free domains as work
    if domain and domain not in FREE_EMAIL_PROVIDERS:
        return "Work"

    # 6. Primary: default category for personal messages and anything else
    return "Primary"


__all__ = ["classify_email"]