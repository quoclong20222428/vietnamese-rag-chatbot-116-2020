# Hybrid Search — HNSW + BM25 + RRF + Cross-Encoder Re-ranking

Hybrid Search kết hợp truy xuất dense bằng HNSW với truy xuất sparse bằng BM25, sau đó hợp nhất thứ hạng bằng Reciprocal Rank Fusion (RRF). Một bước tái xếp hạng tùy chọn bằng Cross-Encoder (`CrossEncoderReranker`) có thể được áp dụng lên tập candidate sau RRF để tinh chỉnh độ chính xác trước khi đưa kết quả vào LLM. Hai retriever gốc vẫn hoạt động độc lập; Hybrid chỉ gọi chúng và kết hợp candidate.

## Vì sao dùng Hybrid Search

HNSW biểu diễn câu hỏi và chunk thành embedding rồi tìm theo độ gần ngữ nghĩa. Cách này phù hợp với câu hỏi diễn đạt tự nhiên dù không lặp đúng từ trong văn bản.

BM25 tìm theo từ và cụm từ xuất hiện trong nội dung. Cách này hữu ích khi câu hỏi cần khớp chính xác số hiệu hoặc cấu trúc pháp luật như “Điều 5”, “Khoản 2”, “Điểm a”, `116/2020/NĐ-CP` hay `60/2025/NĐ-CP`.

Hai cách tìm kiếm có thể bổ trợ nhau: dense retrieval nhận ra ý nghĩa tương đương, còn sparse retrieval nhạy với từ khóa và định danh chính xác. Trong RAG, Hybrid Search tạo danh sách context cho bước sinh câu trả lời; nó không tự đánh giá tính đúng đắn của câu trả lời do LLM tạo.

## Luồng xử lý

### Luồng cơ bản (Hybrid RRF)

```text
Query
 ├── HNSW dense retrieval ──> candidate_k (=40)
 └── BM25 sparse retrieval ─> candidate_k (=40)
                  ↓
            Candidate Pool
                  ↓
                 RRF
                  ↓
              top_k (=20)
```

### Luồng đầy đủ khuyến nghị (Hybrid RRF + Cross-Encoder Re-ranking)

```text
Query
 ├── HNSW dense retrieval ──> candidate_k (=40)
 └── BM25 sparse retrieval ─> candidate_k (=40)
                  ↓
            Candidate Pool
                  ↓
                 RRF
                  ↓
              top_k (=20)   ← đủ rộng cho Re-ranker
                  ↓
        Cross-Encoder Re-ranker
          (BAAI/bge-reranker-v2-m3)
                  ↓
           rerank_top_k (=10) ← context đưa vào LLM
```

## Reciprocal Rank Fusion

Với chunk `d`, RRF tính:

```text
RRF(d) = Σ 1 / (k + rank(d))
```

- `rank(d)` là thứ hạng 1-based của chunk trong một nguồn.
- `k` là hằng số làm giảm ảnh hưởng của các vị trí đầu; mặc định trong project là `60`.
- Tổng được tính trên các nguồn có chứa chunk đó.

Ví dụ với `k = 60`: chunk A đứng thứ nhất ở cả HNSW và BM25 nhận `1/61 + 1/61`; chunk B chỉ đứng thứ nhất ở BM25 nhận `1/61`. A được ưu tiên vì cả hai nguồn cùng xếp hạng chunk đó cao.

HNSW cosine similarity và BM25 score có thang đo khác nhau. Cộng raw score sẽ để thang điểm của một thuật toán chi phối fusion, vì vậy baseline này chỉ dùng rank.

## Candidate pool và cấu hình

`candidate_k` là số kết quả yêu cầu từ **mỗi** retriever trước fusion. `final_top_k` là số kết quả tối đa giữ lại sau khi sắp xếp RRF. Hai giá trị độc lập; nếu candidate pool nhỏ hơn final top-k, Hybrid trả về số candidate thực tế có được.

Mặc định CLI dùng `candidate_k=20`, `final_top_k=10` và `rrf_k=60` (chế độ Hybrid thuần túy không re-rank). **Cấu hình khuyến nghị cho pipeline RAG đầy đủ** dùng `candidate_k=40`, `top_k=20` (để lấy đủ candidate cho re-ranker) và `rerank_top_k=10`. Có thể đổi riêng từng tham số:

```powershell
conda activate chatbot
$env:EMBEDDING_MODEL = "vietlegal-harrier"

# Hybrid thuần túy (không re-rank)
python scripts/test_retrieval.py --method hybrid --candidate-k 20 --top-k 10 --rrf-k 60 --ef-search 80

# Pipeline đầy đủ khuyến nghị: Hybrid RRF + Cross-Encoder Re-rank
python scripts/test_retrieval.py --method hybrid --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80 --rerank --rerank-top-k 10

# Dùng mô hình re-ranker khác
python scripts/test_retrieval.py --method hybrid --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80 --rerank --rerank-model BAAI/bge-reranker-v2-m3 --rerank-top-k 10
```

`--top-k` giữ nguyên vai trò final top-k sau RRF (số candidate đưa vào re-ranker). `--rerank-top-k` kiểm soát số kết quả cuối trả ra sau Cross-Encoder. `--candidate-k` chỉ áp dụng cho Hybrid. CLI hỗ trợ truy vấn đơn để xem kết quả mà không chạy metric:

```powershell
# Ad-hoc query với re-ranking
python scripts/test_retrieval.py --method hybrid --query "Điều kiện được miễn bồi hoàn là gì?" --candidate-k 40 --top-k 20 --ef-search 80 --rerank --rerank-top-k 10
```

Các lệnh chạy từ thư mục gốc `RAG-chatbot`. HNSW dùng model/cột embedding do `EMBEDDING_MODEL` chọn; BM25 tải corpus `legal_chunks` trong PostgreSQL. Không cần đổi schema hoặc tái lập chỉ mục embedding.

## Cross-Encoder Re-ranking

Sau bước RRF, bước tái xếp hạng (re-ranking) sử dụng mô hình Cross-Encoder để đọc đồng thời cặp (query, chunk) và tính điểm liên quan chính xác hơn. Khác với Bi-Encoder dùng cho HNSW (nhúng query và chunk riêng lẻ), Cross-Encoder đọc cả hai cùng lúc qua một lần forward pass, cho phép mô hình học được sự tương tác chéo giữa từng từ trong query và từng từ trong chunk.

**Mô hình mặc định:** `BAAI/bge-reranker-v2-m3` — mô hình multilingual của BAAI, hỗ trợ tiếng Việt và các ngôn ngữ pháp lý, tải qua `sentence-transformers` đã có sẵn trong môi trường.

**Thông số thống nhất cho dự án:**

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `candidate_k` | `40` | Số chunk lấy từ mỗi retriever (HNSW và BM25) |
| `rrf_k` | `60` | Hằng số RRF — giữ nguyên theo paper gốc |
| `top_k` (sau RRF) | `20` | Số chunk đưa vào Cross-Encoder |
| `rerank_top_k` | `10` | Số chunk cuối trả về cho LLM |
| `ef_search` | `80` | Tham số tìm kiếm HNSW |

**Lý do chọn các giá trị này:**
- `candidate_k=40`: thà bắt nhầm còn hơn bỏ sót ở bước candidate; đảm bảo Recall cao trước khi lọc.
- `top_k=20` sau RRF: đủ rộng để Cross-Encoder có pool chất lượng, nhưng không quá lớn làm chậm inference.
- `rerank_top_k=10`: cân bằng giữa context đủ phong phú cho LLM và tránh hội chứng "Lost in the Middle" khi context quá dài.

**Module:** `scripts/retrievers/reranker.py` — `CrossEncoderReranker` lazy-load model, bảo toàn toàn bộ RRF diagnostics trong `metadata["retrieval_diagnostics"]` để trace.

## Kết quả chung và diagnostics

HNSW, BM25 và Hybrid có thể benchmark riêng trên cùng `EvalQueryV2` bằng CLI `scripts/test_retrieval.py`. Bộ dữ liệu có 100 query, gồm 95 answerable và 5 OOS/invalid; ground truth và nhãn liên quan không bị thay đổi. Các metric vẫn được tính bởi cùng evaluation runner.

- **Hit@K**: tỷ lệ query có ít nhất một chunk grade ≥ 2 trong K kết quả.
- **Recall@K**: tỷ lệ chunk ground-truth grade ≥ 2 được tìm thấy trong K kết quả.
- **Precision@K**: tỷ lệ kết quả liên quan trong danh sách K; project giữ mẫu số K kể cả khi retriever trả ít kết quả.
- **MRR**: trung bình nghịch đảo thứ hạng của chunk liên quan đầu tiên.
- **nDCG@K**: chất lượng thứ hạng có tính mức độ liên quan grade 1–3; chunk liên quan cao được thưởng nhiều hơn khi đứng đầu.

Hit, Recall, Precision và MRR dùng `min_grade=2`; nDCG dùng toàn bộ grade 1–3. OOS/invalid được đưa vào report để xem hành vi truy xuất nhưng không tính vào trung bình metric. Điểm cosine, BM25 và RRF chỉ là điểm nội bộ/diagnostic, không phải metric chất lượng. Khi dùng `--rerank`, report ghi thêm Cross-Encoder score và pre-rerank score/rank vào diagnostics.

Report Hybrid ghi `chunk_id`, final rank, RRF score, rank trong từng nguồn, candidate list từng nguồn và lỗi retrieval riêng lẻ. Phần tổng hợp candidate diagnostics cho biết mỗi nguồn tìm thấy ground-truth candidate ở bao nhiêu query, số query chỉ một nguồn tìm được candidate đúng, và độ giao nhau candidate trung bình. Các số này giúp xác định hai nguồn có bổ trợ nhau trong candidate pool hay không; chúng không thay thế metric cuối cùng sau fusion.

So sánh kết quả và điều kiện chạy mới nhất nằm trong [tài liệu evaluation](retrieval-evaluation.md).

## API Python

```python
from scripts.retrievers.bm25 import BM25Retriever
from scripts.retrievers.hnsw import Retriever
from scripts.retrievers.hybrid import HybridRetriever
from scripts.retrievers.reranker import CrossEncoderReranker

hnsw = Retriever(ef_search=80)
bm25 = BM25Retriever()
hybrid = HybridRetriever(
    {"hnsw": hnsw, "bm25": bm25},
    candidate_k=40,
    rrf_k=60,
)
reranker = CrossEncoderReranker(
    model_name="BAAI/bge-reranker-v2-m3",
    rerank_top_k=10,
)

# Bước 1: Hybrid RRF lấy ra 20 candidates
candidates = hybrid.retrieve("Khoản 2 Điều 5 quy định gì?", top_k=20)
# Bước 2: Cross-Encoder re-rank xuống còn 10 results
results = reranker.rerank("Khoản 2 Điều 5 quy định gì?", candidates)

for result in results:
    diag = result.metadata["retrieval_diagnostics"]
    print(
        result.chunk_id, result.rank,
        f"CE={result.score:.4f}",
        f"pre_rrf={diag['pre_rerank_score']:.6f} (rank {diag['pre_rerank_rank']})",
    )
```

Mỗi kết quả sau re-ranking dùng `RetrievalResult` với `score_type="cross_encoder"`, `retrieval_method="hybrid+rerank"` và `rank` đánh lại từ 1. `retrieval_diagnostics` trong metadata giữ Cross-Encoder score mới, RRF score gốc, rank gốc trước re-ranking và tên mô hình reranker. HNSW/BM25/Hybrid result contract không bị thay đổi.

## Kiểm thử

```powershell
conda activate chatbot
python -m pytest tests/test_hybrid_retriever.py -v
```

Tài liệu liên quan: [Tầng retrieval](retrieval.md), [Đánh giá retrieval](retrieval-evaluation.md).
