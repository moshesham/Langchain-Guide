# Meta Data Engineer Candidate Study Guide & Interview Blueprint

An interactive web application built with Python and Streamlit, rendered client-side on GitHub Pages via **stlite** (WebAssembly), and wrapped in Jekyll format. This application features a robust CI/CD pipeline using GitHub Actions.

## Setup Instructions

1. **Initialize Git Repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Meta DE Prep App"
   ```

2. **Push to GitHub:**
   ```bash
   git remote add origin <your-github-repo-url>
   git branch -M main
   git push -u origin main
   ```

3. **Activate GitHub Pages:**
   - Navigate to your repository on GitHub.
   - Go to **Settings** > **Pages**.
   - Under **Source**, select **GitHub Actions** (the `.github/workflows/deploy.yml` file handles the deployment).
   - Once the action completes, your interactive Streamlit application will be live!

## Architecture

- **Frontend UI / App Logic:** Python 3.11+, Streamlit framework.
- **Static Hosting Platform:** GitHub Pages with Jekyll format (`_config.yml`).
- **WebAssembly Client Runtime:** `@stlite/mountable` (Serverless Streamlit executing pure Python inside the browser without requiring a backend server).
- **CI/CD Automation:** GitHub Actions (`.github/workflows/deploy.yml`).
- **Interactive Visualization Libraries:** Pydeck, Plotly Express, Pandas, Graphviz, Altair.
