# 🧠 Cluster Workflow App with Llama3 Integration

This is an interactive **Streamlit-based clustering assistant** that enables users to upload a dataset, analyze it for clustering suitability, apply multiple clustering algorithms, and generate **interpretable summaries** — all without coding.

It supports **K-Means**, **K-Prototypes**, **GMM**, and **DBSCAN**, with dynamic preprocessing tailored to your selected method. Additionally, it features **Llama3-powered executive summaries** for dataset diagnostics.

---

## 🚀 Features

- **📁 Upload CSV/Excel** and preview your dataset.
- **📊 Auto-diagnostics** including nulls, dtypes, shape, duplicates, and column summaries.
- **🔍 AI-powered dataset summarization** using Llama3 (via `ollama`).
- **🧠 Clustering Support**:
  - K-Means
  - K-Prototypes (for mixed categorical + numerical data)
  - Gaussian Mixture Model (GMM)
  - DBSCAN (density-based)
- **🎯 Dynamic Feature Selection** with categorical encoding and numeric scaling.
- **📋 Cluster Summary Table** with user-defined statistics (mean, count, percentage, etc.).
- **📄 Export Llama3 summaries as downloadable PDFs.**
- **🔁 Seamless navigation** across app stages with session memory.

---

## 📦 Dependencies

Install dependencies via `pip install -r requirements.txt`.

