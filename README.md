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
scripts/test_retrieval.py     →  đánh giá retrieval HNSW
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

Các script cấp cao nhất trong `scripts/` là CLI; package con như `scripts/embeddings/`, `scripts/indexing/`, `scripts/retrievers/` và `scripts/evaluation/` chứa implementation được các CLI import. BM25 là retriever implementation riêng, chưa có CLI benchmark riêng trong repository.

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
| Đánh giá Retrieval (EvalQueryV2 — 100 câu hỏi, ground truth phân cấp và chia mức) | ✅ Hoàn thành |
| Bộ kiểm thử tự động — 91/91 tests passed (embedding + retrieval) | ✅ Hoàn thành |
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

Đánh giá thực nghiệm được thực hiện trên tập benchmark chuẩn **`EvalQueryV2`** (100 câu hỏi: 95 câu hỏi hợp lệ, 5 câu hỏi ngoài phạm vi OOS / không hợp lệ) trên toàn bộ 618 chunks pháp lý trong cơ sở dữ liệu NeonDB PostgreSQL (`pgvector`), sử dụng chỉ mục HNSW (`vector_cosine_ops`, `ef_search=80`, `top_k=10`).

### Bảng kết quả tổng hợp 5 mô hình embedding

Nguồn dữ liệu: 5 tệp log benchmark chính thức tại thư mục `logs/` (ngưỡng liên quan nhị phân `grade >= 2`, nDCG có trọng số đa mức `1, 2, 3`):

| Mô hình | Alias | Chiều vector | Hit@10 | Recall@10 | MRR | nDCG@10 | Avg Top-1 Sim |
|---|---|---:|---:|---:|---:|---:|---:|
| `mainguyen9/vietlegal-harrier-0.6b` | `vietlegal-harrier` | 1024 | **0.9053** | **0.6591** | **0.6528** | **0.5653** | 0.601330 |
| `BAAI/bge-m3` | `bge-m3` | 1024 | 0.8842 | 0.6282 | 0.6229 | 0.5457 | 0.698086 |
| `mainguyen9/vietlegal-e5` | `vietlegal-e5` | 1024 | 0.8316 | 0.5502 | 0.5432 | 0.4549 | 0.678649 |
| `jinaai/jina-embeddings-v3-hf` | `jina-v3` | 1024 | 0.7684 | 0.5527 | 0.4983 | 0.4484 | 0.732983 |
| `darklethelong/vnlegal-lal` | `vnlegal-lal` | 1024 | 0.7158 | 0.4647 | 0.4759 | 0.3903 | **0.933826** |
| `BM25 (Sparse)` | `bm25` | N/A | 0.7368 | 0.4696 | 0.4101 | 0.3798 | N/A |

**Nhận xét chính:**
- `mainguyen9/vietlegal-harrier-0.6b` và `BAAI/bge-m3` đạt hiệu năng truy xuất dẫn đầu toàn bảng: Hit@10 đạt trên 88-90%, Recall@10 đạt trên 62-65%, MRR đạt 0.62-0.65.
- `mainguyen9/vietlegal-e5` đạt độ phủ Top-10 khá tốt (83.16%) với MRR 0.5432.
- `darklethelong/vnlegal-lal` có điểm tương đồng Top-1 trung bình cao nhất nhóm (**0.933826**), nhưng các chỉ số truy xuất thực tế lại thấp nhất. Điểm tương đồng cosin tuyệt đối không được dùng làm thước đo xếp hạng giữa các mô hình khác nhau.

**Lệnh chạy đánh giá nhanh qua PowerShell:**
```powershell
conda activate chatbot
$env:EMBEDDING_MODEL = "bge-m3"
python scripts/test_retrieval.py --top-k 10 --ef-search 80
```

> Chi tiết đầy đủ về phương pháp luận, giải thích bản chất từng chỉ số và tham số kỹ thuật, ground truth phân cấp chia mức, phân tích chuyên sâu theo danh mục/độ khó và hướng dẫn tái lập xem tại: [Đánh giá Retrieval](docs/retrieval-evaluation.md).

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
├── retrievers/              ← bm25, hnsw
├── environment.py           ← cấu hình môi trường (.env)
├── database.py              ← kết nối PostgreSQL
├── retrieval_types.py       ← hợp đồng RetrievalResult dùng chung
├── legal_chunker.py         ← CLI wrapper cho chunking
├── import_legal_data.py     ← CLI wrapper cho import_data
├── validate_legal_chunks.py ← CLI wrapper cho validation
├── index_embeddings.py      ← CLI wrapper cho indexing
├── test_retrieval.py        ← CLI đánh giá HNSW (EvalQueryV2, 100 câu hỏi)
└── gpu_smoke_test.py
tests/
├── test_embedding.py        ← 30 unit tests
└── test_retrieval.py        ← 209 unit tests
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
| [Lịch sử phát triển Retrieval](docs/retrieval-development-history.md) | Text-only → cải tiến đánh giá → metadata-aware |
| [Đánh giá Retrieval](docs/retrieval-evaluation.md) | Phương pháp luận, định nghĩa chỉ số, điều kiện thực nghiệm, kết quả benchmark 6 mô hình và phân tích chuyên sâu |
