# Hệ thống hỏi đáp Nghị định 116/2020/NĐ-CP dựa trên RAG

## Giới thiệu

Đây là dự án xây dựng hệ thống hỏi đáp pháp luật tập trung vào **Nghị định 116/2020/NĐ-CP** về chính sách hỗ trợ sinh viên sư phạm. Dữ liệu gồm nghị định chính, **Nghị định 60/2025/NĐ-CP** (văn bản sửa đổi), và **Luật Giáo dục 2019** để bổ sung định nghĩa và ngữ cảnh pháp lý. Bộ câu hỏi–trả lời về Nghị định 116 được dùng làm tài liệu tham khảo, không thay thế nguồn pháp lý chính thức.

Theo định hướng RAG (Retrieval-Augmented Generation), hệ thống nhận câu hỏi, truy xuất các đoạn pháp lý liên quan, rồi sinh câu trả lời bằng tiếng Việt có trích dẫn nguồn từ **mô hình LLM**.

---

## Kiến trúc tổng quát — Pipeline hoàn chỉnh

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE / BUILD PHASE                            │
│                                                                          │
│  Tệp Markdown pháp lý (data/markdown/, data/raw/qa/)                     │
│          ↓                                                               │
│  scripts/legal_chunker.py  →  data/processed/legal_chunks.jsonl          │
│          ↓                                                               │
│  scripts/validate_legal_chunks.py  →  kiểm tra JSONL (618 chunks)        │
│          ↓                                                               │
│  scripts/import_legal_data.py  →  PostgreSQL (documents, legal_chunks)   │
│          ↓                                                               │
│  scripts/index_embeddings.py  →  BGE-M3 embeddings + HNSW index          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                         ONLINE / QUERY PHASE                             │
│                                                                          │
│  Câu hỏi tiếng Việt của người dùng                                       │
│          ↓                                                               │
│  RAGPipeline.run() — scripts/rag_pipeline.py                             │
│          ↓                                                               │
│  HybridRetriever (HNSW + BM25 + RRF)  →  top-40 candidates               │
│          ↓                                                               │
│  CrossEncoderReranker (bge-reranker-v2-m3)  →  top-10 chunks             │
│          ↓                                                               │
│  format_evidence_blocks()  →  numbered evidence [1]…[n] + metadata       │
│          ↓                                                               │
│  build_user_prompt()  →  NGUON PHAP LY + CAU HOI                         │
│          ↓                                                               │
│  LLMClient.generate()  →  câu trả lời tiếng Việt                         │
│          ↓                                                               │
│  RAGResponse(answer, sources, retrieval_method, retrieved_count)         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Công nghệ chính

| Thành phần | Công nghệ |
|---|---|
| Cơ sở dữ liệu | PostgreSQL + pgvector |
| Mô hình nhúng | `BAAI/bge-m3` (mặc định) và 4 mô hình khác |
| Chỉ mục vector | HNSW (`vector_cosine_ops`, m=16, ef_construction=64) |
| Tìm kiếm từ khóa | BM25 (`pg_trgm` + PostgreSQL full-text) |
| Fusion & Reranking | Hybrid RRF + Cross-Encoder `BAAI/bge-reranker-v2-m3` |
| Mô hình sinh | LLM (cấu hình qua `.env`) |
| Ngôn ngữ | Python |
| Kết nối DB | `psycopg` / `psycopg2` |

---

## Quy trình chạy

### Bước 0 — Kích hoạt môi trường (tùy chọn)

Ví dụ nếu dùng Conda:

```powershell
conda activate chatbot
```

### Bước 1–4: Build Phase (Offline)

**1. Chunking dữ liệu pháp lý:**

```powershell
python scripts/legal_chunker.py
```

**2. Kiểm tra JSONL:**

```powershell
python scripts/validate_legal_chunks.py
```

**3. Import vào PostgreSQL:**

```powershell
python scripts/import_legal_data.py
```

> Cần PostgreSQL đang chạy và `DATABASE_URL` đã cấu hình trong `.env`.

**4. Sinh embeddings & tạo chỉ mục HNSW:**

```powershell
python scripts/index_embeddings.py --model bge-m3
```

### Bước 5: Đánh giá Retrieval (tùy chọn)

```powershell
# HNSW đơn thuần
$env:EMBEDDING_MODEL = "bge-m3"
python scripts/test_retrieval.py --top-k 10 --ef-search 80

# Hybrid RRF
python scripts/test_retrieval.py --method hybrid --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80

# Hybrid + Cross-Encoder Rerank (cấu hình benchmark tốt nhất)
python scripts/test_retrieval.py --method hybrid --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80 --rerank --rerank-model BAAI/bge-reranker-v2-m3 --rerank-top-k 10
```

Kết quả ghi vào `logs/`.

### Bước 6: Chạy RAG Pipeline đầu cuối

Bằng file python:

```python
from scripts.rag_pipeline import build_pipeline, run_query

# Khởi tạo pipeline (load model embedding + BM25 corpus — ~30-60s lần đầu)
pipeline = build_pipeline()

# Truy vấn
response = run_query(pipeline, "Sinh viên sư phạm được hỗ trợ bao nhiêu?")

print(response.answer)
```

Hoặc trực tiếp bằng lệnh:

```powershell
python -c "
from scripts.rag_pipeline import build_pipeline, run_query
p = build_pipeline()
r = run_query(p, 'Sinh viên sư phạm được hỗ trợ bao nhiêu?')
print(r.answer)
"
```

> Cần các biến LLM API key và model được cấu hình trong `.env` (xem `.env.example`).

---

## Cấu hình mặc định pipeline (`scripts/config.py`)

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `CANDIDATE_K` | 40 | Số ứng viên mỗi retriever (HNSW, BM25) trước RRF |
| `TOP_K` | 20 | Kết quả sau RRF, trước reranking |
| `RRF_K` | 60 | Hằng số RRF |
| `EF_SEARCH` | 80 | HNSW ef_search |
| `RERANK` | True | Bật Cross-Encoder reranking mặc định |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-Encoder model |
| `RERANK_TOP_K` | 10 | Số chunks đưa vào tầng Generation |

---

## Phương pháp truy xuất — Metadata-aware Embedding

Mỗi đoạn pháp lý được biểu diễn kèm tiền tố cấu trúc trước khi đưa vào BGE-M3:

```text
[Document] <document_title>
[Chapter]  <chapter>
[Article]  <article>
[Clause]   <clause>
[Point]    <point>
[Content]  <nội dung gốc>
```

Câu hỏi được nhúng nguyên văn, không biến đổi. Chi tiết: [Lịch sử phát triển Retrieval](docs/retrieval-development-history.md).

---

## Kết quả đánh giá Retrieval — EvalQueryV2

### Benchmark tốt nhất — Hybrid RRF + Cross-Encoder Rerank (2026-10-06)

Cấu hình: `candidate-k=40`, `top-k=20`, `rrf-k=60`, `ef_search=80`, reranker `BAAI/bge-reranker-v2-m3`, `rerank-top-k=10`.

| Model | Alias | Hit@3 | Hit@10 | Recall@10 | MRR | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| `BAAI/bge-m3` | `bge-m3` | **0.8421** | 0.9053 | **0.6169** | **0.7568** | **0.5713** |
| `mainguyen9/vietlegal-harrier-0.6b` | `vietlegal-harrier` | 0.8316 | **0.9368** | 0.6124 | 0.7299 | 0.5626 |
| `mainguyen9/vietlegal-e5` | `vietlegal-e5` | **0.8421** | 0.9158 | 0.5944 | 0.7362 | 0.5468 |
| `jinaai/jina-embeddings-v3-hf` | `jina-v3` | 0.7895 | 0.8632 | 0.5385 | 0.7028 | 0.5318 |
| `darklethelong/vnlegal-lal` | `vnlegal-lal` | 0.7789 | 0.8526 | 0.5176 | 0.6875 | 0.5023 |

**Nhận xét:**
- `bge-m3` mạnh nhất tổng thể: Recall@10 (0.6169), MRR (0.7568), nDCG@10 (0.5713) → **mô hình mặc định**.
- `vietlegal-harrier` có Hit@10 cao nhất (0.9368), phù hợp khi ưu tiên coverage.
- Chi tiết đầy đủ: [Đánh giá Retrieval](docs/retrieval-evaluation.md).

**Lệnh tái lập:**
```powershell
$env:EMBEDDING_MODEL="bge-m3"; python scripts/test_retrieval.py --method hybrid --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80 --rerank --rerank-model BAAI/bge-reranker-v2-m3 --rerank-top-k 10
```

---

## Tầng Generation — Grounded Answer với LLM

`scripts/rag_pipeline.py` điều phối toàn bộ pipeline từ câu hỏi đến câu trả lời:

1. **Validate** câu hỏi đầu vào.
2. **Retrieve** chunks liên quan qua HybridRetriever (HNSW + BM25 + RRF).
3. **Rerank** với Cross-Encoder `bge-reranker-v2-m3` → top-10 chunks.
4. **Format** chunks thành numbered evidence blocks `[S1]…[Sn]` kèm metadata pháp lý.
5. **Build prompt** kết hợp `NGUỒN PHÁP LÝ` + câu hỏi.
6. **Generate** câu trả lời tiếng Việt có trích dẫn `[1], [2]…` qua LLM client (mặc định `GeminiClient`).
7. **Return** `RAGResponse(answer, sources, retrieval_method, retrieved_count)`.

### Biến môi trường cần thiết

| Biến | Mô tả |
|---|---|
| `LLM_API_KEY` | API key của LLM provider (ví dụ Gemini, OpenRouter…) |
| `LLM_MODEL` | Tên model LLM (ví dụ `gemini-2.0-flash`) |
| `DATABASE_URL` | PostgreSQL connection URL |
| `EMBEDDING_MODEL` | Alias mô hình nhúng (mặc định `bge-m3`) |

> Tên biến cụ thể phụ thuộc vào LLM client được sử dụng. Xem `.env.example` để biết chi tiết.

### Trích dẫn và Grounding

- Mỗi chunk được đánh nhãn `[1]`, `[2]`… trong prompt. LLM được yêu cầu trích dẫn theo đúng nhãn và ghi **Nguồn tham khảo** ở cuối.
- Metadata pháp lý (điều, khoản, điểm, tên văn bản) hiển thị trong header mỗi evidence block.
- Hệ thống **không tự suy diễn** điều khoản khi metadata thiếu — thiết kế cố ý tránh hallucination.

Chi tiết: [Generation Layer](docs/generation.md).

---

## Cấu trúc thư mục

```text
init.sql                     ← schema PostgreSQL (nguồn sự thật)
.env.example                 ← mẫu cấu hình biến môi trường
requirements.txt
README.md
docs/                        ← tài liệu chi tiết
scripts/
├── config.py                ← cấu hình pipeline tập trung
├── environment.py           ← load biến môi trường (.env)
├── database.py              ← kết nối PostgreSQL
├── retrieval_types.py       ← hợp đồng RetrievalResult dùng chung
├── rag_pipeline.py          ← pipeline RAG đầu cuối (Retrieval → Generation)
├── data_pipeline/           ← chunking, validation
├── embeddings/              ← embedding backend, model registry
├── evaluation/              ← dataset, matching, metrics, runner, reporting
├── generator/
│   ├── gemini.py            ← Gemini API wrapper
│   └── openrouter.py        ← OpenRouter API wrapper (dự phòng)
├── indexing/                ← embedding_index, import_data
├── retrievers/              ← bm25, hnsw, hybrid, reranker
├── legal_chunker.py         ← CLI wrapper chunking
├── import_legal_data.py     ← CLI wrapper import_data
├── validate_legal_chunks.py ← CLI wrapper validation
├── index_embeddings.py      ← CLI wrapper indexing
└── test_retrieval.py        ← CLI đánh giá HNSW, BM25, Hybrid (EvalQueryV2)
tests/
├── test_embedding.py        ← unit tests embedding
├── test_retrieval.py        ← unit tests HNSW retrieval
├── test_bm25_retriever.py   ← unit tests BM25
├── test_hybrid_retriever.py ← unit tests RRF, Hybrid, evaluation integration
└── test_generation.py       ← unit tests Generation (Gemini mock)
logs/
├── retrieval_*.txt          ← báo cáo đánh giá kèm nhãn thời gian
└── <model>_<timestamp>.txt  ← báo cáo benchmark từng mô hình embedding
data/
├── markdown/                ← tài liệu pháp lý gốc (.md)
├── raw/qa/                  ← bộ câu hỏi–trả lời tham khảo
└── processed/
    └── legal_chunks.jsonl   ← 618 chunks đã xử lý
```

---

## Tài liệu chi tiết

| Tài liệu | Nội dung |
|---|---|
| [Cài đặt & Cấu hình](docs/setup.md) | Yêu cầu môi trường, Conda, pip, biến `.env` |
| [Chuẩn bị dữ liệu](docs/data-preparation.md) | Nguồn pháp lý, pipeline chunking, JSONL |
| [Cơ sở dữ liệu & Import](docs/database-and-import.md) | Schema `init.sql`, import, xác minh DB |
| [Embedding & Indexing](docs/embedding-and-indexing.md) | BGE-M3, HNSW, cấu hình, kết quả indexing |
| [Chuyển đổi mô hình Embedding](docs/embedding-model-switching.md) | Hướng dẫn chuyển đổi giữa các mô hình embedding |
| [Retrieval](docs/retrieval.md) | Kiến trúc, module, cách dùng, unit test |
| [BM25 Retrieval](docs/retrieval-bm25.md) | Chi tiết BM25, cấu hình, so sánh với HNSW |
| [Hybrid Search](docs/retrieval-hybrid.md) | HNSW + BM25, RRF, cấu hình và diagnostics |
| [Lịch sử phát triển Retrieval](docs/retrieval-development-history.md) | Text-only → metadata-aware → hybrid |
| [Đánh giá Retrieval](docs/retrieval-evaluation.md) | Phương pháp luận, chỉ số, benchmark 5 mô hình, phân tích chuyên sâu |
| [Generation Layer](docs/generation.md) | Gemini wrapper, RAG pipeline, grounding, trích dẫn |
