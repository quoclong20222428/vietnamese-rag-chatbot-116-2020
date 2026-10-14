# Generation Layer

Tài liệu này mô tả tầng LLM Generation đã được thêm vào dự án RAG chatbot Nghị định 116/2020/NĐ-CP.

## Tổng quan kiến trúc

```text
Câu hỏi người dùng (tiếng Việt)
        ↓
scripts/rag_pipeline.py  →  validate input
        ↓
HybridRetriever (HNSW + BM25 + RRF)  →  top-7 legal chunks
        ↓
format_evidence_blocks()  →  numbered evidence blocks [S1]…[Sn]
        ↓
build_user_prompt()  →  NGUỒN PHÁP LÝ + CÂU HỎI
        ↓
GeminiClient.generate()  →  Vietnamese grounded answer
        ↓
RAGResponse(answer, sources, retrieval_method, retrieved_count)
```

## Các file mới

| File | Mục đích |
|---|---|
| `scripts/generator/__init__.py` | Package marker |
| `scripts/generator/gemini.py` | Wrapper Google Gemini API |
| `scripts/rag_pipeline.py` | Pipeline điều phối Retrieval → Generation |
| `tests/test_generation.py` | Unit tests (không cần API key thật) |

## Biến môi trường cần thiết

| Biến | Mô tả |
|---|---|
| `GEMINI_API_KEY` | Google Generative AI API key |
| `GEMINI_MODEL` | Gemini model identifier, ví dụ `gemini-2.0-flash` |
| `DATABASE_URL` | PostgreSQL connection URL (đã có từ trước) |
| `EMBEDDING_MODEL` | Embedding model alias (mặc định `BAAI/bge-m3`) |

Tất cả biến này được load từ `.env` qua `scripts/environment.py` (đã có sẵn).

> [!IMPORTANT]
> Kiểm tra giá trị `GEMINI_MODEL` trong `.env`. Tên model phải khớp với Gemini API
> (ví dụ: `gemini-2.0-flash`, `gemini-1.5-pro`). Giá trị hiện tại `"Gemini 3.6 Flash"`
> có dấu ngoặc kép và có thể không đúng format — xóa dấu ngoặc kép nếu có.

## Cài đặt dependency

```powershell
conda activate chatbot
pip install google-genai>=1.0.0
```

Hoặc cài toàn bộ:

```powershell
pip install -r requirements.txt
```

## Sử dụng pipeline

```python
from scripts.rag_pipeline import build_pipeline, run_query

# Khởi tạo (load model embedding + BM25 corpus — mất ~30-60s lần đầu)
pipeline = build_pipeline()

# Truy vấn
response = run_query(pipeline, "Điều kiện để được hưởng học bổng sư phạm là gì?")

print(response.answer)
print(f"\nSố nguồn truy xuất: {response.retrieved_count}")
for src in response.sources:
    print(f"  {src['label']} {src['citation_hint']}")
```

### Cấu hình nâng cao

```python
pipeline = build_pipeline(
    candidate_k=40,   # ứng viên mỗi retriever trước RRF (mặc định)
    top_k=10,          # số chunks đưa vào mô hình LLM
    ef_search=80,     # HNSW ef_search (mặc định, khớp benchmark)
)
```

## Cách grounding và trích dẫn hoạt động

Mỗi chunk được truy xuất sẽ được định danh bằng nhãn `[1]`, `[2]`, … và đặt
trong phần `NGUỒN PHÁP LÝ` của prompt. Metadata pháp lý (điều, khoản, điểm, tên
văn bản) được hiển thị ngay trên header mỗi khối bằng chứng.

Mô hình LLM được yêu cầu (qua system prompt tiếng Việt) trích dẫn bằng nhãn `[1]`,`[2]`...,
kèm tham chiếu pháp lý cụ thể nhất có trong metadata. Nếu metadata thiếu thông
tin (không có số điều/khoản), hệ thống **không tự động suy diễn** — đây là thiết
kế cố ý để tránh bịa đặt điều khoản pháp lý.

`format_evidence_blocks()` trả về `sources` — danh sách metadata của từng nguồn —
để caller hiển thị song song với câu trả lời.

## Chạy tests

```powershell
conda activate chatbot
python -m pytest tests/test_generation.py -v
```

Tất cả Gemini API calls được mock — không cần API key thật, không phát sinh chi phí.

Chạy toàn bộ test suite:

```powershell
python -m pytest tests/ -v
```

## Giới hạn đã biết

- **Không có conversation memory:** Pipeline xử lý từng câu hỏi độc lập (single-turn).
  Multi-turn cần bổ sung thêm.
- **Cấu hình mặc định:** `build_pipeline()` luôn dùng Hybrid RRF. Nếu chỉ muốn
  HNSW hoặc BM25, truyền retriever tự tạo vào `RAGPipeline(retriever=..., generator=...)`.
- **Chưa có integration test thực:** Tests hiện tại chỉ là unit tests với mock.
  Cần `GEMINI_API_KEY` thật và database live để chạy integration test.
- **GEMINI_MODEL format:** Giá trị trong `.env` có thể cần kiểm tra — xem mục
  biến môi trường ở trên.
