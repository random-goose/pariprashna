"""
Mahabharata RAG Chat: Hybrid Retrieval-Augmented Generation over Classical Sanskrit Epics
Pipeline: Query -> Instruction Dense (Qwen3) + BM25 -> Reciprocal Rank Fusion (RRF) -> Context Expansion -> LLM
"""

import argparse
import contextlib
import html
import io
import os
import re
import sys
import threading
from packaging.version import Version
import gradio as gr
from huggingface_hub import hf_hub_download
from jinja2 import BaseLoader, Environment
from llama_cpp import Llama
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Global State & Defaults
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM = """You are a Mahabharata excerpt-answering tool.

Output format (produce exactly this, nothing before it):
<answer>
One paragraph, 2-5 sentences, built only from the numbered excerpts below. Each sentence ends with its supporting number(s), e.g. [1] or [2][5]. Skip excerpts that don't fit the question.
</answer>

Procedure:
1. Read the question once.
2. Read the excerpts once, mark which numbers are relevant.
3. Write the <answer> block directly using only the marked excerpts.
4. Stop after the closing </answer> tag.

If no excerpt answers the question, output <answer>One sentence naming what the excerpts do cover instead.</answer>

Begin your response immediately with <answer> — no text before it."""

CITATION_CSS = """
<style>
.mb-cites {
  --ink: #14213d; --ink-soft: #4a5568; --rule: #e2ddd2;
  --card: #fdfcf9; --ochre: #a9741a;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
.mb-cite {
  background: var(--card); border: 1px solid var(--rule);
  border-left: 3px solid var(--ochre); border-radius: 3px;
  padding: 12px 14px; margin-bottom: 10px;
}
.mb-cite-head { display: flex; align-items: baseline; gap: 8px; margin-bottom: 8px; }
.mb-cite-rank {
  font-variant-numeric: tabular-nums; font-weight: 700;
  font-size: 0.78rem; color: var(--ochre); flex: none;
}
.mb-cite-loc {
  font-size: 0.68rem; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--ink-soft); line-height: 1.4;
}
.mb-cite-sa {
  font-family: "Noto Sans Devanagari", "Nirmala UI", sans-serif;
  font-size: 0.95rem; line-height: 1.85; color: var(--ink); margin-bottom: 8px;
}
.mb-cite-en { font-size: 0.88rem; line-height: 1.6; color: var(--ink); }
.mb-cite-scores {
  margin-top: 9px; padding-top: 7px; border-top: 1px dashed var(--rule);
  font-size: 0.7rem; font-variant-numeric: tabular-nums;
  color: var(--ink-soft); display: flex; gap: 14px; flex-wrap: wrap;
}
.mb-empty {
  padding: 22px 14px; border: 1px dashed var(--rule); border-radius: 3px;
  font-size: 0.85rem; color: var(--ink-soft); text-align: center;
}
@media (prefers-color-scheme: dark) {
  .mb-cites {
    --ink: #eceae4; --ink-soft: #a5a196;
    --rule: #3a3a3a; --card: #1e1e1e; --ochre: #d9a441;
  }
}
</style>
"""

stop_event = threading.Event()
model_lock = threading.Lock()


def tokenize(s):
    return re.findall(r"\w+", str(s).lower())


def _cell(row, col, default=""):
    val = row.get(col, default)
    return default if pd.isna(val) else val


@contextlib.contextmanager
def quiet_llama():
    """Suppress llama.cpp C-level logging during generation."""
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


# ---------------------------------------------------------------------------
# Mahabharata RAG Pipeline Class
# ---------------------------------------------------------------------------
class MahabharataRAG:
    def __init__(self, data_dir: str, embed_device: str = "cuda", n_ctx: int = 4096, hnsw_ef: int = 128):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, "mahabharat_with_qwen3-0.6B_embeddings.csv")
        self.emb_cache = f"{self.csv_path}.embcache.npy"
        self.meta_cache = os.path.join(data_dir, "meta.parquet")
        self.pkl_cache = os.path.join(data_dir, "meta.pkl")

        self.text_col, self.sa_col, self.emb_col = "en", "sa", "en_embedding"
        self.parva_col, self.chapter_col, self.verse_col = "parva", "chapter", "id"

        self.embed_model_name = "Qwen/Qwen3-Embedding-0.6B"
        self.embed_device = embed_device
        self.hnsw_ef = hnsw_ef
        self.n_ctx = n_ctx

        self.model_repo = "unsloth/Qwen3.5-9B-MTP-GGUF"
        self.model_file = "Qwen3.5-9B-UD-Q8_K_XL.gguf"

        self._load_embedder()
        self._load_corpus()
        self._build_indexes()
        self._load_llm()

    def _load_embedder(self):
        print(f"[*] Loading embedding model ({self.embed_model_name}) on {self.embed_device}...")
        self.embedder = SentenceTransformer(
            self.embed_model_name,
            device=self.embed_device,
            model_kwargs={"torch_dtype": "auto"},
            tokenizer_kwargs={"padding_side": "left"},
        )
        try:
            self.emb_dim = self.embedder.get_embedding_dimension()
        except AttributeError:
            self.emb_dim = self.embedder.get_sentence_embedding_dimension()
        print(f"[*] Embedder ready: dim={self.emb_dim}")

    def _load_corpus(self):
        if not os.path.exists(self.emb_cache):
            raise FileNotFoundError(
                f"Embedding cache missing at {self.emb_cache}. "
                "Download the precomputed embeddings or generate them from the source dataset."
            )
        print(f"[*] Loading precomputed embeddings from {self.emb_cache}...")
        self.embs = np.load(self.emb_cache).astype(np.float32)
        self.embs /= np.linalg.norm(self.embs, axis=1, keepdims=True) + 1e-12
        self.N = self.embs.shape[0]

        if os.path.exists(self.meta_cache):
            print(f"[*] Loading metadata from {self.meta_cache}...")
            self.df = pd.read_parquet(self.meta_cache)
        elif os.path.exists(self.pkl_cache):
            print(f"[*] Loading metadata from {self.pkl_cache}...")
            self.df = pd.read_pickle(self.pkl_cache)
        else:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(
                    f"Corpus CSV not found at {self.csv_path}. Please place it in data_dir."
                )
            print(f"[*] Parsing metadata from {self.csv_path} (skipping embedding column)...")
            header = pd.read_csv(self.csv_path, nrows=0).columns.tolist()
            keep = [c for c in header if c != self.emb_col]
            self.df = pd.read_csv(self.csv_path, usecols=keep)
            try:
                self.df.to_parquet(self.meta_cache, index=False)
            except Exception:
                self.df.to_pickle(self.pkl_cache)

        assert len(self.df) == self.N, f"Row count mismatch: df={len(self.df)} vs embs={self.N}"
        print(f"[*] Corpus loaded: {self.N:,} verses.")

    def _build_indexes(self):
        import hnswlib
        print("[*] Initializing HNSW vector index...")
        self.index = hnswlib.Index(space="cosine", dim=self.emb_dim)
        self.index.init_index(max_elements=self.N, ef_construction=200, M=16)
        self.index.add_items(self.embs, np.arange(self.N))
        self.index.set_ef(self.hnsw_ef)

        print("[*] Building BM25 sparse index...")
        corpus_tokens = [tokenize(t) for t in self.df[self.text_col].fillna("")]
        self.bm25 = BM25Okapi(corpus_tokens)
        print("[*] Hybrid retrieval indexes built successfully.")

    def _load_llm(self):
        print(f"[*] Downloading/verifying LLM GGUF ({self.model_file})...")
        model_path = hf_hub_download(repo_id=self.model_repo, filename=self.model_file)
        print(f"[*] Loading llama.cpp model from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=self.n_ctx,
            verbose=False,
        )
        env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
        self.chat_template = env.from_string(self.llm.metadata["tokenizer.chat_template"])
        print("[*] LLM and chat template initialized.")

    def embed_query(self, query: str, task: str = "Given a query, retrieve relevant passages that answer the query"):
        v = self.embedder.encode([query], prompt=f"Instruct: {task}\nQuery:", normalize_embeddings=True)
        return v.astype(np.float32)

    def dense_search(self, query: str, k: int = 50):
        k = min(k, self.N)
        self.index.set_ef(max(self.hnsw_ef, k))
        labels, dists = self.index.knn_query(self.embed_query(query), k=k)
        return list(labels[0]), (1.0 - dists[0])

    def sparse_search(self, query: str, k: int = 50):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return list(top), scores[top]

    @staticmethod
    def rrf(rank_lists, k: int = 60):
        fused = {}
        for ids in rank_lists:
            for rank, doc_id in enumerate(ids):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])

    def neighbor_text(self, doc_id: int, window: int):
        row = self.df.iloc[doc_id]
        if window <= 0:
            return str(_cell(row, self.text_col))
        lo, hi = max(0, doc_id - window), min(self.N - 1, doc_id + window)
        parts = []
        for j in range(lo, hi + 1):
            r = self.df.iloc[j]
            if _cell(r, self.chapter_col) != _cell(row, self.chapter_col):
                continue
            if _cell(r, self.parva_col) != _cell(row, self.parva_col):
                continue
            parts.append(str(_cell(r, self.text_col)))
        return " ".join(parts)

    def retrieve(self, query: str, top_k_each: int = 50, top_n: int = 8, window: int = 1):
        d_ids, d_scores = self.dense_search(query, top_k_each)
        s_ids, s_scores = self.sparse_search(query, top_k_each)
        d_map, s_map = dict(zip(d_ids, d_scores)), dict(zip(s_ids, s_scores))

        hits = []
        for rank, (doc_id, score) in enumerate(self.rrf([d_ids, s_ids])[:top_n], 1):
            row = self.df.iloc[doc_id]
            hits.append({
                "rank": rank,
                "doc_id": int(doc_id),
                "rrf": float(score),
                "dense": float(d_map.get(doc_id, float("nan"))),
                "bm25": float(s_map.get(doc_id, float("nan"))),
                "parva": str(_cell(row, self.parva_col, "?")),
                "chapter": str(_cell(row, self.chapter_col, "?")),
                "verse": str(_cell(row, self.verse_col, "?")),
                "en": str(_cell(row, self.text_col)),
                "sa": str(_cell(row, self.sa_col)),
                "context": self.neighbor_text(doc_id, window),
            })
        return hits

    def render_prompt(self, messages, add_generation_prompt=True, enable_thinking=False):
        return self.chat_template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=None,
            add_vision_id=False,
        )


# ---------------------------------------------------------------------------
# UI Helpers & Handlers
# ---------------------------------------------------------------------------
def render_citations(hits, show_sanskrit=True):
    if not hits:
        return (CITATION_CSS + '<div class="mb-cites"><div class="mb-empty">'
                'Ask a question to retrieve verses from the corpus.</div></div>')

    parts = [CITATION_CSS, '<div class="mb-cites">']
    for h in hits:
        d = "—" if np.isnan(h["dense"]) else f"{h['dense']:.4f}"
        b = "—" if np.isnan(h["bm25"]) else f"{h['bm25']:.2f}"
        parts += [
            '<div class="mb-cite">',
            '<div class="mb-cite-head">',
            f'<span class="mb-cite-rank">[{h["rank"]}]</span>',
            '<span class="mb-cite-loc">',
            f'{html.escape(h["parva"])} &middot; Chapter {html.escape(h["chapter"])}',
            f' &middot; Verse {html.escape(h["verse"])}</span>',
            '</div>',
        ]
        if show_sanskrit and h["sa"].strip():
            parts.append(f'<div class="mb-cite-sa">{html.escape(h["sa"])}</div>')
        parts += [
            f'<div class="mb-cite-en">{html.escape(h["en"])}</div>',
            '<div class="mb-cite-scores">',
            f'<span>rrf {h["rrf"]:.5f}</span><span>dense {d}</span>',
            f'<span>bm25 {b}</span><span>row {h["doc_id"]}</span></div>',
            '</div>',
        ]
    parts.append('</div>')
    return "".join(parts)


def build_context_block(hits, char_budget=4000):
    blocks, used = [], 0
    for h in hits:
        entry = (f"[{h['rank']}] {h['parva']}, Chapter {h['chapter']}, "
                 f"Verse {h['verse']}\n{h['context'] or h['en']}")
        if used + len(entry) > char_budget:
            break
        blocks.append(entry)
        used += len(entry)
    return "\n\n".join(blocks)


def normalize_history(history):
    clean = []
    for item in history or []:
        if isinstance(item, dict):
            role, content = item.get("role"), item.get("content")
            if role in ("system", "user", "assistant") and isinstance(content, str):
                clean.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            u, b = item
            if u: clean.append({"role": "user", "content": str(u)})
            if b: clean.append({"role": "assistant", "content": str(b)})
    return clean


def make_chatbot(**kwargs):
    gradio_ver = Version(gr.__version__)
    if Version("4.27") <= gradio_ver < Version("6.0"):
        kwargs["type"] = "messages"
    return gr.Chatbot(**kwargs)


def build_demo(rag: MahabharataRAG):
    with gr.Blocks(title="Mahabharata RAG Chat") as demo:
        gr.Markdown(
            f"# Mahabharata RAG System\n"
            f"Hybrid retrieval across **{rag.N:,} verses** combining Dense (Qwen3-Embedding-0.6B) "
            f"and Sparse (BM25Okapi) fusion via Reciprocal Rank Fusion (RRF)."
        )
        hits_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=5):
                chatbot = make_chatbot(label="Conversation", height=560)
                msg = gr.Textbox(label="Question", placeholder="How did Karna die?", lines=3, autofocus=True)
                with gr.Row():
                    send_btn = gr.Button("Ask", variant="primary")
                    preview_btn = gr.Button("Retrieve only")
                    stop_btn = gr.Button("Stop")
                    clear_btn = gr.Button("Clear")
                status = gr.Markdown("Ready.")

            with gr.Column(scale=4):
                gr.Markdown("### Source Citations")
                show_sanskrit = gr.Checkbox(value=True, label="Show Sanskrit")
                citations = gr.HTML(render_citations([], True))

            with gr.Column(scale=2):
                gr.Markdown("### Retrieval & Inference Config")
                use_rag = gr.Checkbox(value=True, label="Use retrieval")
                top_n = gr.Slider(1, 20, value=6, step=1, label="Verses in context")
                top_k_each = gr.Slider(10, 200, value=50, step=10, label="Candidates / retriever")
                window = gr.Slider(0, 3, value=1, step=1, label="Context window (adjacent verses)")
                char_budget = gr.Slider(1000, 12000, value=4000, step=500, label="Context char budget")
                temperature = gr.Slider(0.0, 2.0, value=0.3, step=0.05, label="Temperature")
                max_tokens = gr.Slider(64, 2048, value=768, step=64, label="Max tokens")
                system_prompt = gr.Textbox(label="System prompt", value=DEFAULT_SYSTEM, lines=10)

        def add_user_message(message, history):
            history = normalize_history(history)
            if message and message.strip():
                history = history + [{"role": "user", "content": message}]
            return "", history

        def retrieval_step(history, use_rag, top_k_each, top_n, window):
            history = normalize_history(history)
            if not use_rag:
                return [], render_citations([], True), "Retrieval disabled — using internal LLM knowledge."
            if not history or history[-1]["role"] != "user":
                return [], render_citations([], True), "No query detected."
            try:
                hits = rag.retrieve(history[-1]["content"], int(top_k_each), int(top_n), int(window))
            except Exception as e:
                return [], render_citations([], True), f"Retrieval failed: {e}"
            if not hits:
                return [], render_citations([], True), "No matching verses found."
            return hits, render_citations(hits, True), f"Retrieved {len(hits)} verses from {rag.N:,}."

        def bot_reply(history, hits, use_rag, temperature, max_tokens, system_prompt, char_budget):
            stop_event.clear()
            history = normalize_history(history)
            if not history or history[-1]["role"] != "user":
                yield history
                return

            question = history[-1]["content"]
            prior = history[:-1][-4:]
            messages = []
            if system_prompt and system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(prior)

            if use_rag and hits:
                ctx = build_context_block(hits, int(char_budget))
                messages.append({"role": "user", "content": f"Verse excerpts:\n\n{ctx}\n\n---\n\nQuestion: {question}"})
            else:
                messages.append({"role": "user", "content": question})

            base = history
            yield base + [{"role": "assistant", "content": ""}]
            response = ""

            try:
                with model_lock:
                    prompt = rag.render_prompt(messages, enable_thinking=False)
                    stream = rag.llm.create_completion(
                        prompt=prompt,
                        temperature=float(temperature),
                        max_tokens=int(max_tokens),
                        stream=True,
                    )
                    for chunk in stream:
                        if stop_event.is_set():
                            break
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        piece = choices[0].get("text")
                        if piece:
                            response += piece
                            yield base + [{"role": "assistant", "content": response}]

                if stop_event.is_set() and response:
                    yield base + [{"role": "assistant", "content": response + "\n\n_(generation halted)_"}]
            except Exception as e:
                yield base + [{"role": "assistant", "content": f"**Generation Error:** `{e}`"}]

        # Triggers
        for trigger in (send_btn.click, msg.submit):
            trigger(
                fn=add_user_message, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
            ).then(
                fn=retrieval_step, inputs=[chatbot, use_rag, top_k_each, top_n, window],
                outputs=[hits_state, citations, status]
            ).then(
                fn=bot_reply, inputs=[chatbot, hits_state, use_rag, temperature, max_tokens, system_prompt, char_budget],
                outputs=chatbot
            )

        preview_btn.click(
            fn=lambda q, k, n, w: (rag.retrieve(q, int(k), int(n), int(w)), render_citations(rag.retrieve(q, int(k), int(n), int(w)), True), "Previewed verses.") if q.strip() else ([], render_citations([], True), "Type query first."),
            inputs=[msg, top_k_each, top_n, window],
            outputs=[hits_state, citations, status]
        )
        show_sanskrit.change(fn=lambda h, s: render_citations(h, s), inputs=[hits_state, show_sanskrit], outputs=citations, queue=False)
        stop_btn.click(fn=lambda: stop_event.set() or "Generation halted.", outputs=status, queue=False)
        clear_btn.click(fn=lambda: stop_event.set() or ([], [], render_citations([], True), "Cleared."), outputs=[chatbot, hits_state, citations, status], queue=False)

    return demo


# ---------------------------------------------------------------------------
# CLI & Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Mahabharata Hybrid RAG Runner")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing embeddings and corpus files.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Embedder inference device.")
    parser.add_argument("--n_ctx", type=int, default=4096, help="LLM KV context window length.")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio share link.")
    args = parser.parse_args()

    rag = MahabharataRAG(data_dir=args.data_dir, embed_device=args.device, n_ctx=args.n_ctx)
    demo = build_demo(rag)
    demo.queue().launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()