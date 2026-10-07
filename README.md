# Xây dựng hệ thống hỏi đáp Nghị định 116/2020/NĐ-CP dựa trên RAG

## Giới thiệu

Đây là dự án xây dựng hệ thống hỏi đáp pháp luật tập trung vào **Nghị định 116/2020/NĐ-CP**. Dữ liệu gồm nghị định chính, **Nghị định 60/2025/NĐ-CP** (văn bản sửa đổi), và **Luật Giáo dục 2019** để bổ sung định nghĩa và ngữ cảnh pháp lý. Bộ câu hỏi–trả lời về Nghị định 116 được dùng làm tham khảo, không thay thế nguồn pháp lý chính thức.

Theo định hướng RAG (Retrieval-Augmented Generation), hệ thống nhận câu hỏi, truy xuất các đoạn pháp lý liên quan, rồi cung cấp câu trả lời dựa trên nội dung đã truy xuất.

---

## Kiến trúc tổng quát

```text
Tệp Markdown pháp lý
        ↓
scripts/legal_chunker.py  →  data/processed/legal_chunks.jsonl
        ↓
scripts/validate_legal_chunks.py  →  kiểm tra JSONL
        ↓
scripts/import_legal_data.py  →  PostgreSQL (documents, legal_chunks)
        ↓
scripts/index_embeddings.py   →  embedding + HNSW index
        ↓
scripts/test_retrieval.py     →  đánh giá retrieval HNSW, BM25 hoặc Hybrid RRF
```

## Quy trình chạy Python chính

Chạy các lệnh từ thư mục gốc repository trong PowerShell. Nếu sử dụng `conda`, cần kích hoạt môi trường `chatbot` trước mọi lệnh Python:

```powershell
conda activate chatbot
```

1. Tạo chunks từ tài liệu trong `data/markdown/` và `data/raw/qa/`; đầu ra mặc định là `data/processed/legal_chunks.jsonl`:

   ```powershell
   python scripts/legal_chunker.py
   ```

2. Kiểm tra JSONL đã tạo. Bước này chỉ đọc dữ liệu và in báo cáo, không sửa database:

   ```powershell
   python scripts/validate_legal_chunks.py
   ```

3. Import `data/processed/legal_chunks.jsonl` vào PostgreSQL. Lệnh này chạy `init.sql` rồi cập nhật các bảng, nên cần PostgreSQL và `DATABASE_URL` đã cấu hình:

   ```powershell
   python scripts/import_legal_data.py
   ```

4. Sinh embeddings và tạo/cập nhật cột cùng chỉ mục HNSW cho model đã chọn (mặc định `BAAI/bge-m3`). Đây là thao tác ghi database; chỉ chạy sau khi chunks đã import:

   ```powershell
   python scripts/index_embeddings.py --model bge-m3
   ```

5. Chạy bộ đánh giá retrieval HNSW sau khi có embeddings. Kết quả được ghi vào `logs/`:

   ```powershell
   $env:EMBEDDING_MODEL = "bge-m3"
   python scripts/test_retrieval.py --top-k 10 --ef-search 80
   ```
   hoặc thay đổi mô hình trong .env và chạy:
   ```powershell
   python scripts/test_retrieval.py --top-k 10 --ef-search 80
   ```     

Các script cấp cao nhất trong `scripts/` là CLI; package con như `scripts/embeddings/`, `scripts/indexing/`, `scripts/retrievers/` và `scripts/evaluation/` chứa implementation được các CLI import. HNSW, BM25 và Hybrid RRF có thể được đánh giá riêng trên cùng `EvalQueryV2`.

Chạy baseline Hybrid RRF:

```powershell
python scripts/test_retrieval.py --method hybrid --candidate-k 20 --top-k 10 --rrf-k 60 --ef-search 80
```

Thiết kế, cấu hình và benchmark so sánh gần nhất: [Hybrid Search](docs/retrieval-hybrid.md) và [Đánh giá Retrieval](docs/retrieval-evaluation.md).

---

## Công nghệ chính

| Thành phần | Công nghệ |
|---|---|
| Cơ sở dữ liệu | PostgreSQL + pgvector |
| Mô hình nhúng | `BAAI/bge-m3` (mặc định) và 4 mô hình khác |
| Chỉ mục vector | HNSW — Hierarchical Navigable Small World: cấu trúc chỉ mục tìm kiếm vector tương đồng nhanh |
| Tìm kiếm Trigram | `pg_trgm` (đã có sẵn trong schema) |
| Ngôn ngữ | Python (môi trường Conda `chatbot`) |
| Kết nối DB | `psycopg` / `psycopg2` |

---

## Trạng thái hiện tại

| Giai đoạn | Trạng thái |
|---|:---:|
| Thu thập & Xử lý dữ liệu (618 chunks pháp lý) | ✅ Hoàn thành |
| Chunking & Chuẩn hóa cấu trúc (Điều/Khoản/Điểm) | ✅ Hoàn thành |
| Cơ sở dữ liệu PostgreSQL + pgvector (`init.sql`) | ✅ Hoàn thành |
| Vector hóa BGE-M3 — 618/618 chunks, 1024 chiều | ✅ Hoàn thành |
| Chỉ mục HNSW (`vector_cosine_ops`, m=16, ef_construction=64) | ✅ Hoàn thành |
| Module Vector Retrieval (`scripts/retrievers/hnsw.py`) | ✅ Hoàn thành |
| Hybrid Search (HNSW + BM25 + RRF) | ✅ Hoàn thành |
| Đánh giá Retrieval (EvalQueryV2 — 100 câu hỏi, ground truth phân cấp và chia mức) | ✅ Hoàn thành |
| Bộ kiểm thử tự động (embedding, HNSW, BM25, Hybrid) | ✅ Hoàn thành |
| Nhúng tài liệu nhận biết siêu dữ liệu (Metadata-aware) | ✅ Hoàn thành |
| Đánh giá so sánh 5 mô hình embedding | ✅ Hoàn thành |

---

## Phương pháp truy xuất hiện tại

Tầng Retrieval hiện dùng **metadata-aware document embedding**: mỗi đoạn pháp lý được biểu diễn kèm tiền tố cấu trúc (tên văn bản, chương, điều, khoản, điểm) trước khi đưa vào BGE-M3. Câu hỏi được nhúng nguyên văn, không biến đổi.

```text
[Document] <document_title>
[Chapter]  <chapter>
[Article]  <article>
[Clause]   <clause>
[Point]    <point>
[Content]  <nội dung gốc>
```

Chi tiết đầy đủ: [Lịch sử phát triển Retrieval](docs/retrieval-development-history.md).

---

## Kết quả đánh giá Retrieval

### Benchmark mới nhất — EvalQueryV2 2026-10-06 (Hybrid RRF + Cross-Encoder Rerank)

Cấu hình: `candidate-k=40`, `top-k=20`, `rrf-k=60`, `ef_search=80`, reranker `BAAI/bge-reranker-v2-m3`, `rerank-top-k=10`. Dataset `EvalQueryV2` sau update (QA-reference chunks có thể có grade 2/3).

| Model | Alias | Hit@3 | Hit@5 | Hit@10 | Recall@3 | Recall@5 | Recall@10 | Precision@3 | Precision@5 | Precision@10 | MRR | nDCG@3 | nDCG@5 | nDCG@10 | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `BAAI/bge-m3` | `bge-m3` | **0.8421** | 0.8737 | 0.9053 | 0.4263 | 0.5244 | **0.6169** | **0.4702** | **0.3453** | **0.2126** | **0.7568** | **0.5317** | **0.5409** | **0.5713** | 197.27s |
| `mainguyen9/vietlegal-harrier-0.6b` | `vietlegal-harrier` | 0.8316 | **0.9158** | **0.9368** | **0.4344** | **0.5355** | 0.6124 | 0.4632 | 0.3389 | 0.2042 | 0.7299 | 0.5231 | 0.5359 | 0.5626 | 190.04s |
| `mainguyen9/vietlegal-e5` | `vietlegal-e5` | **0.8421** | 0.8947 | 0.9158 | 0.4273 | 0.5091 | 0.5944 | 0.4596 | 0.3347 | 0.2042 | 0.7362 | 0.5105 | 0.5156 | 0.5468 | 193.47s |
| `jinaai/jina-embeddings-v3-hf` | `jina-v3` | 0.7895 | 0.8526 | 0.8632 | 0.4064 | 0.4770 | 0.5385 | 0.4211 | 0.3011 | 0.1811 | 0.7028 | 0.5083 | 0.5117 | 0.5318 | 177.13s |
| `darklethelong/vnlegal-lal` | `vnlegal-lal` | 0.7789 | 0.8211 | 0.8526 | 0.3985 | 0.4683 | 0.5176 | 0.4281 | 0.3011 | 0.1726 | 0.6875 | 0.4890 | 0.4887 | 0.5023 | 183.77s |

**Nhận xét chính:**
- `bge-m3` mạnh nhất tổng thể về Recall@10 (0.6169), Precision@10 (0.2126), MRR (0.7568) và nDCG@10 (0.5713).
- `vietlegal-harrier` có Hit@10 cao nhất (0.9368), phù hợp khi ưu tiên coverage/hit probability.
- `vietlegal-e5` thuộc nhóm trên thứ ba, kết quả khá cân bằng.
- `jina-v3` thấp hơn nhóm dẫn đầu về coverage.
- `vnlegal-lal` thấp nhất ở các metric tổng hợp chính.
- **Không nên so sánh trực tiếp** bảng 2026-10-06 với bảng 2026-10-02 vì khác phương pháp retrieval và khác ground truth semantics.

**Lệnh tái lập (Hybrid + Rerank):**
```powershell
conda activate chatbot
$env:EMBEDDING_MODEL="bge-m3"; python scripts/test_retrieval.py --method hybrid --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80 --rerank --rerank-model BAAI/bge-reranker-v2-m3 --rerank-top-k 10
```

> Chi tiết đầy đủ về phương pháp luận, ground truth phân cấp, phân tích theo category và tất cả caveats: [Đánh giá Retrieval](docs/retrieval-evaluation.md).



---

## Cấu trúc thư mục

```text
init.sql                     ← schema PostgreSQL (nguồn sự thật)
.env.example                 ← mẫu cấu hình biến môi trường
requirements.txt
README.md
docs/                        ← tài liệu chi tiết
scripts/
├── data_pipeline/           ← chunking, validation
├── embeddings/              ← embedding backend, model registry
├── evaluation/              ← dataset, matching, metrics, runner, reporting
├── indexing/                ← embedding_index, import_data
├── retrievers/              ← bm25, hnsw, hybrid
├── environment.py           ← cấu hình môi trường (.env)
├── database.py              ← kết nối PostgreSQL
├── retrieval_types.py       ← hợp đồng RetrievalResult dùng chung
├── legal_chunker.py         ← CLI wrapper cho chunking
├── import_legal_data.py     ← CLI wrapper cho import_data
├── validate_legal_chunks.py ← CLI wrapper cho validation
├── index_embeddings.py      ← CLI wrapper cho indexing
├── test_retrieval.py        ← CLI đánh giá HNSW, BM25, Hybrid (EvalQueryV2)
└── gpu_smoke_test.py
tests/
├── test_embedding.py        ← 30 unit tests
├── test_retrieval.py        ← HNSW unit tests
├── test_bm25_retriever.py   ← BM25 unit tests
└── test_hybrid_retriever.py ← RRF, Hybrid và evaluation integration
logs/
├── retrieval_*.txt          ← báo cáo đánh giá kèm nhãn thời gian
└── <model>_<timestamp>.txt  ← báo cáo benchmark từng mô hình embedding
data/
├── markdown/
├── raw/
└── processed/
    └── legal_chunks.jsonl
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
| [Hybrid Search](docs/retrieval-hybrid.md) | HNSW + BM25, RRF, cấu hình và diagnostics |
| [Lịch sử phát triển Retrieval](docs/retrieval-development-history.md) | Text-only → cải tiến đánh giá → metadata-aware |
| [Đánh giá Retrieval](docs/retrieval-evaluation.md) | Phương pháp luận, định nghĩa chỉ số, điều kiện thực nghiệm, kết quả benchmark 6 mô hình và phân tích chuyên sâu |
