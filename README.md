# Renewal Intelligence NBA (Local, Parquet-first)

Pipeline:
Synthetic data → Join/features → KG → Retrieval → LLM reasoning (Ollama) → Recommendation ranking → Chatbot API/UI

## 1) Setup (Conda)
```bash
conda env create -f environment.yml
conda activate renewal-intel