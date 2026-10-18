# LangChain & Production AI Systems Engineering Blueprint

An interactive, production-grade web application built for AI Engineers to master **LangChain**, **RAG Pipelines**, **Vector Database Trade-offs**, **Agentic Workflows (LangGraph)**, and **LLM Scaling Economics**.

Rendered client-side via **stlite** (WebAssembly Pyodide runtime) and paired with an Express full-stack proxy supporting real-time Google Search grounded research.

## Features
- **Interactive Mermaid.js RAG Sequence Diagram:** Visualizing query expansion, dense embedding, vector retrieval, cross-encoder reranking, and structured output parsing with interactive node inspection.
- **Scaling Cost & Token Calculator:** Real-time token consumption simulation and LLM pricing comparison across query volumes with semantic caching economics.
- **Vector Database Configurator:** Multi-dimensional trade-off matrix and memory footprint analysis comparing Pinecone, FAISS, Chroma, Qdrant, Milvus, and pgvector.
- **LangChain LCEL Deep Dive:** Composition patterns, streaming, fallback runnables, and production state machines.
- **AI Architecture Radar:** Real-time search-grounded research on LLM engineering and distributed AI systems.

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
