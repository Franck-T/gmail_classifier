"""
main.py
~~~~~~~

Command‑line application that connects to the Gmail API, reads a selection of
messages and classifies them using the rule‑based classifier defined in
``classifier.py``.  The script authenticates via OAuth2, fetches the most
recent 25 messages and prints a summary table showing each subject and the
assigned category.  Authentication tokens are stored in ``token.json`` for
reuse on subsequent runs.

Follow the setup instructions in ``README.md`` before running this script.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from tabulate import tabulate

# Use the AI‑powered classifier
from ai_classifier import classify_email


# If modifying these scopes, delete the file token.json.  We only need
# read‑only access to Gmail messages.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_credentials(credentials_path: str = "credentials.json", token_path: str = "token.json") -> Credentials:
    """Load or obtain OAuth2 credentials for Gmail API access.

    If a valid token exists in ``token_path`` it will be used.  Otherwise
    this function runs the OAuth flow to obtain new credentials.
    """
    creds: Credentials | None = None
    # Load existing token if available
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    # If there are no valid credentials available, perform the OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Missing credentials file at {credentials_path}.  "
                    f"Follow the README to download OAuth client credentials from the Google Cloud console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            #creds = flow.run_local_server(port=0)
            creds = flow.run_console()
        # Save the credentials for the next run
        with open(token_path, "w") as token_file:
            token_file.write(creds.to_json())
    return creds


def get_headers_map(headers_list: List[Dict[str, str]]) -> Dict[str, str]:
    """Convert a list of Gmail message headers into a dictionary mapping names to values."""
    return {h.get("name"): h.get("value") for h in headers_list}


def fetch_and_classify_messages(service, max_results: int = 25) -> List[Dict[str, str]]:
    """Fetch the latest Gmail messages and classify them.

    Parameters
    ----------
    service : googleapiclient.discovery.Resource
        An authorized Gmail API service instance.
    max_results : int
        Maximum number of messages to fetch.

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries with keys: subject, from, snippet, category.
    """
    results: List[Dict[str, str]] = []
    # List message IDs
    message_list = service.users().messages().list(userId="me", maxResults=max_results).execute()
    messages = message_list.get("messages", [])
    for msg in messages:
        msg_id = msg["id"]
        # Fetch message metadata and snippet
        msg_data = service.users().messages().get(
            userId="me",
            id=msg_id,
            format="full",
        ).execute()
        payload = msg_data.get("payload", {})
        headers_list = payload.get("headers", [])
        headers_map = get_headers_map(headers_list)
        subject = headers_map.get("Subject", "")
        sender = headers_map.get("From", "")
        snippet = msg_data.get("snippet", "")
        # Compute category
        category = classify_email(sender, subject, snippet, headers_map)
        results.append({
            "subject": subject,
            "from": sender,
            "snippet": snippet,
            "category": category,
        })
    return results


def main() -> None:
    """Entry point: authenticate and classify messages."""
    try:
        creds = get_credentials()
        service = build("gmail", "v1", credentials=creds)
        messages = fetch_and_classify_messages(service)
        # Build a table for display
        table = [(i + 1, msg["category"], msg["subject"][:60]) for i, msg in enumerate(messages)]
        print(tabulate(table, headers=["#", "Category", "Subject (truncated)"], tablefmt="grid"))
    except HttpError as error:
        print(f"An error occurred: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()