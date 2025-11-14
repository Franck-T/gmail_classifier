# Gmail Email Classification App

This repository contains a small sample application that connects to a Gmail account, reads emails and classifies them into categories such as **Primary**, **Promotions**, **Social**, **Updates**, **Forums** and **Work**.  Unlike earlier versions that relied on simple heuristics, this version uses a free AI model (`sentence-transformers/all-MiniLM-L6-v2`) to compute semantic embeddings of emails and category descriptions.  It chooses the category whose description is most similar to the email according to cosine similarity.  The entire application is implemented in Python.  It includes a command‑line script and a Streamlit web application for interactive exploration.  A Dockerfile is provided to containerise the app.

## Background

Gmail provides "Category Tabs" to help users manage their inboxes.  According to Google's documentation, when you use the **Default** inbox type, Gmail automatically sorts incoming emails into the following categories: **Primary** (personal and important emails), **Social** (messages from social networks), **Promotions** (deals and other promotional emails), **Updates** (automated confirmations and receipts) and **Forums** (messages from online groups and mailing lists)【25304394913824†L40-L50】.  The Google Support site reinforces that these five categories are the only supported categories and that you cannot create your own【617583219031976†L24-L35】【617583219031976†L46-L48】.  Our application uses these categories as the basis for its classification logic.  In addition, we provide a custom **Work** category that attempts to separate work‑related correspondence from personal messages.

## Project structure

```
gmail_classifier/
├── README.md                — this file
├── python/
│   ├── ai_classifier.py     — AI‑based email classifier using sentence embeddings
│   ├── main.py              — command‑line script to fetch and classify messages
│   ├── streamlit_app.py     — Streamlit web application for classification and visualisation
│   ├── requirements.txt     — Python dependencies
├── Dockerfile               — Container definition to run the Streamlit app
```

## Setting up the Gmail API

Before running either the Python or TypeScript code, you must set up a Google Cloud project and enable the Gmail API:

1. **Create a Google Cloud project** and enable the Gmail API via the [Google Cloud Console](https://console.cloud.google.com/).  Follow the "Enable the API" and "Configure OAuth consent" sections in Google’s Python quickstart【197099469592622†L417-L456】.  Be sure to add `https://www.googleapis.com/auth/gmail.readonly` to the list of OAuth scopes.
2. **Download your OAuth credentials**: create OAuth client credentials (type “Desktop application”) and download the `credentials.json` file.
3. **Place `credentials.json` into both the `python/` and `ts/` directories**.  The first time you run the code, the application will prompt you to authorize access and will store an access token (`token.json`) for subsequent runs【197099469592622†L441-L459】.

## Running the Python script

The Python module demonstrates how to list and classify Gmail messages.  It uses `google-api-python-client` and the rule‑based classifier defined in `classifier.py`.

### Installation

```bash
cd gmail_classifier/python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

Place your downloaded `credentials.json` in the `python/` directory.  Then run:

```bash
python main.py
```

The script will:

* Perform the OAuth flow to obtain authorization.  A web browser will open for you to log into the Gmail account and grant read‑only access【197099469592622†L441-L459】.
* List the 25 most recent emails, fetch their subjects, senders and snippets and classify each one using semantic embeddings.  The classification is performed by the AI model described in the next section.
* Print a summary table showing the message subject and the assigned category.

## Streamlit web application

In addition to the command‑line script, a Streamlit application (`streamlit_app.py`) provides a simple web interface for authenticating with Gmail, fetching messages and visualising the classification results.

### Installation and running locally

First install the dependencies:

```bash
cd gmail_classifier/python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Next, place your downloaded `credentials.json` in the `python/` directory.  Then run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The app will open a browser window.  Click the button to fetch and classify the 25 most recent emails.  On first use you will be prompted to authorize the application.  The results are displayed in an interactive table and a bar chart summarises the distribution of categories.

## Classification using AI embeddings

The classifier no longer relies on manually curated heuristics.  It uses the `sentence-transformers/all-MiniLM-L6-v2` model from the open‑source [sentence-transformers](https://www.sbert.net/) library to embed both emails and brief descriptions of each category.  For example, **Promotions** is described as “Deals, offers, advertisements and other promotional emails,” while **Primary** is described as “Personal and important emails from people you know.”  Each email is embedded by concatenating the sender’s address, subject and snippet.  Cosine similarity between the email embedding and each category embedding determines the label with the highest semantic similarity.  This approach leverages the semantics of the text rather than rigid keyword matching.

## Dockerisation

A `Dockerfile` is provided at the root of the `gmail_classifier/` directory.  It builds a container image based on Python, installs all dependencies (including the AI model and Streamlit) and copies the application code.  At runtime the container launches the Streamlit server.

To build and run the container locally:

```bash
cd gmail_classifier
# Build the image (this may take several minutes because it downloads the AI model and dependencies)
docker build -t gmail-classifier:latest .

# Run the container, exposing the Streamlit port and mounting your OAuth credentials
docker run -it --rm -p 8501:8501 \
  -v $(pwd)/credentials.json:/app/python/credentials.json \ # mount your credentials
  -v $(pwd)/token.json:/app/python/token.json \ # optional: persist tokens
  gmail-classifier:latest
```

You can then access the app at `http://localhost:8501`.  Click the button to fetch and classify your emails.  The first time you run the container, it will prompt you to authorise access to your Gmail account.  `token.json` stores the access and refresh tokens for subsequent runs【197099469592622†L441-L459】.

## Caveats

* This sample performs client‑side authorization and is meant for demonstration and personal use only.  For production applications you should implement a secure server‑side authorization flow and protect your credentials.
* Gmail does not allow you to create arbitrary custom categories【617583219031976†L46-L48】.  The **Work** category here is a custom classifier implemented within the app and does not integrate with Gmail’s built‑in tabs.
* Because the classifier uses embeddings and cosine similarity rather than Gmail’s internal machine‑learning model, results may not perfectly match the categories Gmail would assign.  Experiment with the category descriptions or consider fine‑tuning a model if higher accuracy is required.

---
If you run into issues using the Gmail API, consult Google’s official [Python quickstart guide](https://developers.google.com/workspace/gmail/api/quickstart/python) for details on enabling the API, obtaining OAuth credentials and understanding the `token.json` storage mechanism【197099469592622†L417-L456】.