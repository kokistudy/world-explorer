# Toy world explorer

Interactive Streamlit app for small petition networks: edit people, arrows, who signed, then compute pivotality, criticality, and contingency scores (same functions as the experiment model).

## Run locally

```bash
python3 -m pip install -r requirements.txt
streamlit run explore_toy.py
```

Or `python3 explore_toy.py`.

## Put it on GitHub + Streamlit Community Cloud

1. Create a new GitHub repository and upload everything in this folder (keep these files at the **root** of the repo).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and deploy:
   - **Main file path:** `explore_toy.py`
   - Python environment will install `requirements.txt` automatically.

Do not use Vercel: Streamlit needs a long-running server.
