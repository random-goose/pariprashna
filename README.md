# Mahabharata RAG: Hybrid Dense-Sparse Retrieval with Grounded Verification

### Key Technical Contributions
1. **Hybrid Retrieval**: Combines semantic embeddings from `Qwen/Qwen3-Embedding-0.6B` with exact-lexical indexing using `BM25Okapi`, fused using Reciprocal Rank Fusion ($k=60$).
2. **Context Continuity Extension**: Expands matched verse hits with adjacent intra-chapter verses to capture surrounding narrative arcs while strictly retaining fine-grained verse index citations.
3. **Low-Footprint Generation**: Executes `Qwen3.5-9B-UD-Q8_K_XL` (or `Q4_K_XL`) in quantized GGUF format via `llama-cpp-python` with strict `<answer>` extraction guarantees under 16GB VRAM.

---

## Dataset & Embeddings

The dataset comprises 73,659 bilingual (Sanskrit + English) verses across the complete 18 Parvas of the Mahabharata.

| File                                                     | Description                                            | Download Link                                                                                                                                                           |
| :------------------------------------------------------- | :----------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mahabharat_with_qwen3-0.6B_embeddings.csv`              | Full bilingual verse corpus with metadata & embeddings | [**Download via Google Drive**](https://drive.google.com/file/d/1QCfWzmO3dRP0V2IEIVc7Yu2tAz36PCYt/view?usp=drive_link) |
| `mahabharat_with_qwen3-0.6B_embeddings.csv.embcache.npy` | Pre-normalized `float32` (73659, 1024) embedding cache | [**Download via Google Drive**](https://drive.google.com/your-drive-link-here](https://drive.google.com/file/d/16dKeIhW51ek0YKfpm9soRPjO-1xtFuoL/view?usp=sharing))    |
| `meta.parquet`                                           | Pre-extracted text & parva/chapter metadata table      | Auto-generated on first run                                                                                                                                             |

Place the downloaded `.csv` and `.embcache.npy` files into the `./data` folder before starting the server.

---

## Installation & Setup

### 1. Clone Repository & Setup Virtualenv
```bash
git clone [https://github.com/random-goose/pariprashna](https://github.com/random-goose/pariprashna)
cd mahabharata-hybrid-rag

python3 -m venv venv
source venv/bin/activate
````

### 2. Install Dependencies

Bash

```
pip install -r requirements.txt
```

> **CUDA Acceleration Note:** Ensure `llama-cpp-python` is compiled with CUDA support for GPU offloading:
> 
>   
> CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

## Usage

### Running the Web Interface

Bash

```
python app.py --data_dir ./data --device cuda --n_ctx 4096 --port 7860
```

### Parameter Reference

|**Argument**|**Type**|**Default**|**Description**|
|---|---|---|---|
|`--data_dir`|`str`|`./data`|Directory containing corpus files and embedding cache|
|`--device`|`str`|`cuda`|Hardware target for the sentence embedder (`cuda` or `cpu`)|
|`--n_ctx`|`int`|`4096`|Context window allocation for the LLM KV cache|
|`--port`|`int`|`7860`|Port to bind the Gradio web server|
|`--share`|`flag`|`False`|Create a temporary public Gradio URL|
