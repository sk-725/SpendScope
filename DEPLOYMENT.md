# 🚀 SpendScope Deployment Guide

This guide walks you through uploading SpendScope to GitHub and deploying it online so anyone can use your dashboard.

---

## 1. Push SpendScope to GitHub

SpendScope is already initialized as a local git repository on the `main` branch with a clean initial commit.

### Steps to push:
1. Go to [github.com/new](https://github.com/new) and log in.
2. Enter the repository name: **`SpendScope`**
3. Choose **Public** (or Private if you prefer).
4. **Leave "Add a README file", ".gitignore", and license UNCHECKED** (we already created them locally).
5. Click **"Create repository"**.
6. Run the following commands in your terminal:

```bash
cd c:\projects\SpendScope

# Link your local repo to GitHub (replace <YOUR_USERNAME> with your GitHub username)
git remote add origin https://github.com/<YOUR_USERNAME>/SpendScope.git

# Set main branch and push
git branch -M main
git push -u origin main
```

---

## 2. Deploy to Streamlit Community Cloud (Recommended — 100% Free)

Streamlit Community Cloud is the fastest, easiest, and completely free way to host Streamlit apps directly from GitHub.

1. Go to [share.streamlit.io](https://share.streamlit.io/) and click **"Sign in with GitHub"**.
2. Click **"New app"**.
3. Select your repository: **`<YOUR_USERNAME>/SpendScope`**.
4. Set branch: **`main`**.
5. Set main file path: **`app.py`**.
6. Click **"Deploy!"**.
7. In ~60 seconds, your app will be live at `https://<your-app-name>.streamlit.app`!

---

## 3. Deploy to Render (Free Web Service)

SpendScope includes a `render.yaml` blueprint and `Procfile`.

1. Go to [render.com](https://render.com/) and sign up with GitHub.
2. Click **"New +"** ➔ **"Web Service"**.
3. Connect your **`SpendScope`** repository.
4. Render will automatically detect `requirements.txt` and `render.yaml`.
5. Settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
6. Click **"Create Web Service"**.

---

## 4. Run with Docker (Local or Cloud)

SpendScope includes a production `Dockerfile`.

```bash
# Build the Docker image
docker build -t spendscope:latest .

# Run the container on port 8501
docker run -p 8501:8501 spendscope:latest
```

Then visit `http://localhost:8501` in your browser.
