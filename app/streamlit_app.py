# streamlit_app.py
import os
import json
import base64
from typing import Optional, Dict, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# CONFIG
# =========================
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels",
]

# Copy–paste (console-style) OAuth:
# We'll build an auth URL and let the user paste the "code" back.
OAUTH_CLIENT_SECRETS_PATH = os.environ.get("GOOGLE_CLIENT_SECRETS", "credentials.json")

# Token store backend: filesystem (fs) or S3 (s3)
TOKEN_BACKEND = os.environ.get("TOKEN_BACKEND", "fs")  # "fs" | "s3"
TOKEN_DIR = os.environ.get("TOKEN_DIR", "tokens")      # fs root directory

# S3 config (only used if TOKEN_BACKEND == "s3")
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "tokens/")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


# =========================
# TOKEN STORES
# =========================
class TokenStore:
    def save(self, key: str, data: str): ...
    def load(self, key: str) -> Optional[str]: ...
    def key_for_email(self, email: str) -> str:
        safe = email.lower().strip().replace("@", "_at_")
        return f"{safe}.json"

class FileTokenStore(TokenStore):
    def __init__(self, root_dir: str):
        self.root = root_dir
        os.makedirs(self.root, exist_ok=True)
    def save(self, key: str, data: str):
        with open(os.path.join(self.root, key), "w") as f:
            f.write(data)
    def load(self, key: str) -> Optional[str]:
        path = os.path.join(self.root, key)
        return open(path).read() if os.path.exists(path) else None
    def list_emails(self) -> List[str]:
        emails: List[str] = []
        if os.path.isdir(self.root):
            for fn in os.listdir(self.root):
                if fn.endswith(".json"):
                    emails.append(fn.replace("_at_", "@").replace(".json", ""))
        return sorted(emails)

class S3TokenStore(TokenStore):
    def __init__(self, bucket: str, prefix: str = ""):
        import boto3
        self.s3 = boto3.client("s3", region_name=AWS_REGION)
        self.bucket = bucket
        self.prefix = prefix or ""
    def save(self, key: str, data: str):
        self.s3.put_object(Bucket=self.bucket, Key=self.prefix + key, Body=data)
    def load(self, key: str) -> Optional[str]:
        import botocore
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.prefix + key)
            return obj["Body"].read().decode("utf-8")
        except self.s3.exceptions.NoSuchKey:
            return None
        except botocore.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                return None
            raise
    # Listing from S3 is optional; keep it simple here.


def get_store() -> TokenStore:
    if TOKEN_BACKEND == "s3":
        if not S3_BUCKET:
            st.error("S3_BUCKET not set but TOKEN_BACKEND=s3"); st.stop()
        return S3TokenStore(S3_BUCKET, S3_PREFIX)
    return FileTokenStore(TOKEN_DIR)


# =========================
# MODEL (cached)
# =========================
@st.cache_resource
def get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return model.encode(texts)


# =========================
# GMAIL HELPERS
# =========================
def build_installed_flow() -> InstalledAppFlow:
    if not os.path.exists(OAUTH_CLIENT_SECRETS_PATH):
        st.error("Missing credentials JSON. Set GOOGLE_CLIENT_SECRETS or place credentials.json next to this file.")
        st.stop()
    flow = InstalledAppFlow.from_client_secrets_file(OAUTH_CLIENT_SECRETS_PATH, SCOPES)
    # Use out-of-band copy–paste code flow (no https redirect required)
    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
    return flow

def make_authorization_url() -> str:
    flow = build_installed_flow()
    # Keep this flow object for token exchange later
    st.session_state["oauth_flow"] = flow
    auth_url, _ = flow.authorization_url(
        prompt="consent",
        access_type="offline",
        include_granted_scopes="true"
    )
    return auth_url

def exchange_code_for_tokens_with_flow(code: str) -> Optional[Credentials]:
    flow: InstalledAppFlow = st.session_state.get("oauth_flow")
    if not flow:
        st.warning("OAuth flow is not initialized. Click the login link again.")
        return None
    try:
        flow.fetch_token(code=code.strip())
        return flow.credentials
    except Exception as e:
        st.error(f"Auth failed: {e}")
        return None

def gmail_service(creds: Credentials):
    return build("gmail", "v1", credentials=creds)

def get_authed_email(creds: Credentials) -> str:
    svc = gmail_service(creds)
    prof = svc.users().getProfile(userId="me").execute()
    return prof["emailAddress"]

def list_labels_map(svc) -> Dict[str, str]:
    labels = svc.users().labels().list(userId="me").execute().get("labels", [])
    out: Dict[str, str] = {}
    for l in labels:
        name = l.get("name", "")
        typ = l.get("type", "")
        if typ == "user" or (typ == "system" and name.startswith("CATEGORY_")):
            out[name] = l["id"]
    return out

def pretty_label(raw_name: str) -> str:
    mapping = {
        "CATEGORY_PERSONAL": "Primary",
        "CATEGORY_SOCIAL": "Social",
        "CATEGORY_PROMOTIONS": "Promotions",
        "CATEGORY_UPDATES": "Updates",
        "CATEGORY_FORUMS": "Forums",
    }
    if raw_name.startswith("CATEGORY_"):
        return mapping.get(raw_name, raw_name.replace("CATEGORY_", "").title())
    return raw_name

def parse_body(payload: dict) -> str:
    if not payload:
        return ""
    if "parts" not in payload or not payload["parts"]:
        data = payload.get("body", {}).get("data")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        return ""
    text_chunks, html_fallback = [], []
    def walk(p):
        if "parts" in p and p["parts"]:
            for sub in p["parts"]:
                walk(sub)
            return
        data = p.get("body", {}).get("data")
        if not data: return
        decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        mt = p.get("mimeType", "")
        if mt == "text/plain": text_chunks.append(decoded)
        elif mt == "text/html": html_fallback.append(decoded)
    for part in payload["parts"]:
        walk(part)
    if text_chunks: return "\n".join(text_chunks)
    if html_fallback: return html_fallback[0]
    return ""

def fetch_recent(svc, max_results: int, query: str) -> List[dict]:
    args = dict(userId="me", labelIds=["INBOX"], maxResults=max_results)
    if query: args["q"] = query
    res = svc.users().messages().list(**args).execute()
    items = res.get("messages", [])
    out: List[dict] = []
    for it in items:
        msg = svc.users().messages().get(userId="me", id=it["id"]).execute()
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        out.append({
            "id": it["id"],
            "subject": headers.get("Subject", "(no subject)"),
            "from": headers.get("From", "(unknown)"),
            "snippet": msg.get("snippet", ""),
            "body": parse_body(msg.get("payload", {})),
        })
    return out

def apply_label(svc, message_id: str, label_id: str):
    svc.users().messages().modify(
        userId="me", id=message_id, body={"addLabelIds": [label_id]}
    ).execute()


# =========================
# CLASSIFICATION (labels as classes)
# =========================
def classify_to_existing_labels(
    texts: List[str],
    label_display_map: Dict[str, Tuple[str, str]]  # {display_name -> (label_id, raw_name)}
) -> List[str]:
    """Return predicted DISPLAY label for each text using embeddings against label display names."""
    model = get_model()
    display_names = list(label_display_map.keys())
    label_embs = embed_texts(model, display_names)
    text_embs = embed_texts(model, texts)
    sims = cosine_similarity(text_embs, label_embs)  # [n_texts x n_labels]
    best_idx = sims.argmax(axis=1)
    return [display_names[i] for i in best_idx]


# =========================
# STREAMLIT WIZARD (copy–paste OAuth)
# =========================
def main():
    st.set_page_config(page_title="Gmail AI Auto-Labeller", page_icon="📧", layout="wide")
    st.title("📧 Gmail AI Auto-Labeller —  Token Store")

    store = get_store()

    # ---- Step 1: Authenticate (copy the code and paste) ----
    st.header("Step 1 · Authenticate with Google (copy–paste code)")
    if "auth_url" not in st.session_state:
        st.session_state["auth_url"] = make_authorization_url()

    st.markdown(f"**1)** Click to open Google consent: **[🔐 Authorize Gmail access]({st.session_state['auth_url']})**")
    st.markdown("**2)** After granting access, Google shows a **code**. Copy it, then paste below and submit.")

    with st.form("auth_form"):
        st.text_input("Paste the Google authorization code here:", key="auth_code_input")
        submitted = st.form_submit_button("Submit Code")

    if submitted:
        code = (st.session_state.get("auth_code_input") or "").strip()
        if not code:
            st.warning("Please paste the code first.")
        else:
            creds = exchange_code_for_tokens_with_flow(code)
            if creds and creds.valid:
                try:
                    email = get_authed_email(creds)
                    store.save(store.key_for_email(email), creds.to_json())
                    st.success(f"✅ Connected as: {email}")
                except Exception as e:
                    st.error(f"Failed to save token: {e}")
            else:
                st.error("Could not complete OAuth exchange.")

    st.divider()

    # ---- Step 2: Choose connected account ----
    st.header("Step 2 · Choose connected account")
    suggested = ""
    if isinstance(store, FileTokenStore):
        emails = store.list_emails()
        if emails:
            suggested = emails[0]
    email_input = st.text_input("Your Gmail address", value=suggested)

    creds_loaded: Optional[Credentials] = None
    if email_input:
        raw = store.load(store.key_for_email(email_input))
        if raw:
            try:
                creds_loaded = Credentials.from_authorized_user_info(json.loads(raw), SCOPES)
                if creds_loaded and creds_loaded.expired and creds_loaded.refresh_token:
                    creds_loaded.refresh(Request())
                    store.save(store.key_for_email(email_input), creds_loaded.to_json())
                st.success(f"Using stored token for {email_input}")
            except Exception as e:
                st.error(f"Failed to load/refresh token: {e}")
        else:
            st.info("No stored token for this email yet. Complete Step 1 to authorize.")

    if not (creds_loaded and creds_loaded.valid):
        st.stop()

    svc = gmail_service(creds_loaded)

    # ---- Step 3: Read labels (used as classes) ----
    st.header("Step 3 · Read your Gmail labels (used as AI classes)")
    labels_map = list_labels_map(svc)  # {raw -> id}
    label_display_map: Dict[str, Tuple[str, str]] = {
        pretty_label(raw): (lid, raw) for raw, lid in labels_map.items()
    }
    st.write("Labels:", list(label_display_map.keys()))

    # ---- Step 4: Fetch + classify + apply ----
    st.header("Step 4 · Fetch recent emails, classify, and auto-label")
    colA, colB = st.columns(2)
    max_results = colA.number_input("Max results", 1, 200, 25)
    query = colB.text_input("Gmail query (e.g., newer_than:7d, is:unread)", "newer_than:7d")

    if st.button("Fetch Emails"):
        try:
            msgs = fetch_recent(svc, max_results, query)
            st.session_state["msgs"] = msgs
            st.success(f"Fetched {len(msgs)} emails.")
        except Exception as e:
            st.error(f"Failed to fetch emails: {e}")

    msgs = st.session_state.get("msgs", [])
    if msgs:
        texts = [(m["subject"] + " " + m["snippet"] + " " + m["body"]).strip() for m in msgs]
        preds_display: List[str] = classify_to_existing_labels(texts, label_display_map)

        auto_apply = st.checkbox("Apply labels automatically", value=True)
        rows = []
        for m, pred_disp in zip(msgs, preds_display):
            label_id, _raw = label_display_map[pred_disp]
            applied = False
            if auto_apply:
                try:
                    apply_label(svc, m["id"], label_id)
                    applied = True
                except Exception as e:
                    st.error(f"Failed to apply label: {e}")
            rows.append({
                "subject": m["subject"],
                "from": m["from"],
                "predicted_label": pred_disp,
                "applied": applied
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        if not df.empty:
            st.bar_chart(df["predicted_label"].value_counts())


if __name__ == "__main__":
    main()
