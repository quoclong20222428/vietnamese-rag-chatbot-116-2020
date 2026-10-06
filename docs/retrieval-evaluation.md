# Đánh giá Retrieval (Retrieval Evaluation)

> **LƯU Ý:** Kết quả benchmark mới nhất (**Section 7**) được đánh giá trên corpus **618 chunks**, dataset `EvalQueryV2` sau dataset update, với cả **5 embedding models** trên cùng cấu hình Hybrid+Rerank, benchmark ngày **2026-10-06**. Phần benchmark cũ (Section đầu — HNSW/BM25/Hybrid không rerank, ngày 2026-10-02) vẫn được giữ nguyên để tham khảo lịch sử.



Tài liệu này là **nguồn sự thật duy nhất (single canonical source of truth)** về phương pháp luận đánh giá, định nghĩa và ý nghĩa thực tiễn của các chỉ số, cấu trúc ground truth phân cấp (**`EvalQueryV2`**), kết quả benchmark embedding và so sánh retrieval HNSW/BM25/Hybrid RRF mới nhất, cùng hướng dẫn tái lập trên Windows PowerShell.

## Benchmark Lịch sử (2026-10-02): HNSW, BM25 và Hybrid RRF (Không Rerank)

Ba phương pháp được chạy ngày **2026-10-02** trên cùng bộ `EvalQueryV2` và ground truth hiện có. Không thay đổi dataset hay schema.

| Điều kiện | HNSW | BM25 | Hybrid RRF |
|---|---|---|---|
| Dataset / ground truth | 100 query; 95 answerable, 5 OOS/invalid | Như HNSW | Như HNSW |
| Corpus / database | 618 chunks; PostgreSQL + pgvector (`neondb`); HNSW index và 618/618 embeddings | Cùng corpus/database | Cùng corpus/database |
| Embedding model | `mainguyen9/vietlegal-harrier-0.6b` (1024 chiều) | Không dùng | `mainguyen9/vietlegal-harrier-0.6b` (1024 chiều) |
| Retrieval configuration | top-10; `ef_search=80` | top-10; `k1=1.5`, `b=0.75` | HNSW candidate-20 + BM25 candidate-20; final top-10; RRF `k=60`; `ef_search=80` |
| Relevance rule | Hit/Recall/Precision/MRR: grade ≥ 2; nDCG: grade 1–3 | Như HNSW | Như HNSW |

Hai HNSW-based runs dùng model trên NVIDIA GeForce RTX 3050 6GB Laptop GPU. BM25 không cần embedding model. OOS/invalid queries có kết quả trong report nhưng bị loại khỏi các metric trung bình.

### Metric theo cut-off

Các điểm là trung bình trên 95 answerable queries; càng cao càng tốt. Hit đo tỷ lệ query có ít nhất một căn cứ phù hợp; Recall đo phần căn cứ ground-truth tìm thấy; Precision đo mật độ chunk phù hợp trong top-k.

| Method | Hit@3 | Hit@5 | Hit@10 | Recall@3 | Recall@5 | Recall@10 | Precision@3 | Precision@5 | Precision@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HNSW | 0.7895 | 0.8421 | 0.9053 | 0.4707 | 0.5427 | 0.6591 | 0.3474 | 0.2526 | 0.1663 |
| BM25 | 0.4842 | 0.5684 | 0.7368 | 0.3024 | 0.3668 | 0.4696 | 0.1895 | 0.1432 | 0.1011 |
| Hybrid RRF | 0.7053 | 0.7684 | 0.8947 | 0.4304 | 0.5069 | 0.6255 | 0.3053 | 0.2274 | 0.1463 |

| Method | MRR | nDCG@3 | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| HNSW | 0.6528 | 0.5051 | 0.5203 | 0.5653 |
| BM25 | 0.4101 | 0.3112 | 0.3390 | 0.3798 |
| Hybrid RRF | 0.6144 | 0.4768 | 0.4959 | 0.5376 |

MRR là nghịch đảo thứ hạng trung bình của chunk liên quan đầu tiên. nDCG đánh giá thứ tự có xét grade liên quan 1–3. Điểm similarity, BM25 và RRF chỉ dùng để chẩn đoán nội bộ, không nằm trong bảng quality metrics.

### Phân tích kết quả

- Với cấu hình đã chạy, HNSW đứng trên Hybrid ở mọi metric tổng hợp trong bảng; Hybrid đứng trên BM25 ở mọi metric. Ví dụ Recall@5 của Hybrid là `0.5069`, thấp hơn HNSW `0.5427` **0.0358**, nhưng cao hơn BM25 `0.3668` **0.1401**. MRR tương ứng là `0.6144`, `0.6528` và `0.4101`.
- Candidate pool cho thấy hai nguồn có bổ trợ: HNSW top-20 tìm thấy ít nhất một relevant chunk ở 90/95 query, BM25 top-20 ở 80/95; có 13 query chỉ HNSW tìm thấy relevant candidate và 3 query chỉ BM25 tìm thấy. Jaccard overlap candidate trung bình là `0.2368` trên 95 query. Đây là candidate-stage diagnostics, không phải metric kết quả sau fusion.
- BM25 đưa relevant candidate vào pool ở các query **Q017** (miễn bồi hoàn sau khi tốt nghiệp), **Q074** (Điều 12 Nghị định 116 và phần sửa đổi tại Nghị định 60) và **Q078** (cùng chủ đề qua nhiều văn bản), trong khi HNSW top-20 không có relevant chunk.
- Chiều ngược lại, HNSW có relevant candidate mà BM25 top-20 bỏ lỡ ở Q002, Q006, Q008, Q013, Q015, Q019, Q026, Q057, Q065, Q066, Q080, Q081 và Q087.
- Final top-1 của Hybrid trùng rank-1 HNSW ở 46/95 query và rank-1 BM25 ở 42/95; ở 28/95 query, chunk đứng đầu Hybrid không phải rank-1 của nguồn nào. Hai số trùng nguồn có thể giao nhau khi hai retriever cùng xếp một chunk ở đầu.
- Kết quả này cho thấy candidate complement nhưng chưa cho thấy RRF cải thiện hơn HNSW ở cấu hình đã benchmark. Chưa thể suy rộng kết luận này sang candidate size, RRF k hoặc embedding model khác.

### Lệnh tái lập

Chạy từ thư mục gốc project trong PowerShell; cùng `EvalQueryV2`, `min_grade=2`, database và model được giữ nguyên cho cả ba lệnh:

```powershell
conda activate chatbot
$env:EMBEDDING_MODEL = "vietlegal-harrier"
python scripts/test_retrieval.py --method hnsw --top-k 10 --ef-search 80
python scripts/test_retrieval.py --method bm25 --top-k 10
python scripts/test_retrieval.py --method hybrid --candidate-k 20 --top-k 10 --rrf-k 60 --ef-search 80
```

Report đầy đủ từng query được ghi trong `logs/`. Các giá trị trên chỉ phản ánh đúng điều kiện chạy được liệt kê ở đây.

---

## 1. Mục đích và Phạm vi đánh giá

### 1.1. Mục đích đánh giá
Hệ thống RAG (Retrieval-Augmented Generation) phục vụ tra cứu văn bản quy phạm pháp luật giáo dục (trọng tâm là **Nghị định 116/2020/NĐ-CP**, văn bản sửa đổi bổ sung **Nghị định 60/2025/NĐ-CP**, và **Luật Giáo dục 2019**). Tầng Retrieval nhận câu hỏi tiếng Việt, tìm các chunk pháp lý qua HNSW dense search, BM25 sparse search hoặc Hybrid RRF, rồi cung cấp chúng làm ngữ cảnh cho mô hình ngôn ngữ lớn (LLM).

Đánh giá Retrieval nhằm mục đích:
1. Đo lường khách quan năng lực định vị và thu hồi đúng, đủ các căn cứ pháp lý cần thiết từ cơ sở dữ liệu.
2. So sánh hiệu năng thực nghiệm giữa các mô hình embedding khác nhau trong cùng điều kiện chỉ mục HNSW và corpus chuẩn.
3. Nhận diện các điểm nghẽn, hiện tượng trôi dạt ngữ nghĩa (semantic drift) và các dạng lỗi truy xuất điển hình để định hướng tối ưu hóa hệ thống.

### 1.2. Khái niệm "Chất lượng Retrieval" trong dự án
Trong hệ thống này, "Chất lượng Retrieval" được định nghĩa dựa trên hai tiêu chí cốt lõi:
* **Độ bao phủ căn cứ pháp lý (Coverage / Recall)**: Khả năng tìm thấy đầy đủ các điều, khoản, điểm pháp lý chứa thông tin giải quyết câu hỏi, đặc biệt là các câu hỏi đa điều khoản hoặc liên văn bản.
* **Độ chính xác thứ hạng (Ranking Precision)**: Khả năng ưu tiên đưa các đoạn trích trả lời trực tiếp (Core Answer) lên vị trí đầu tiên (Rank 1, Top-3), giảm thiểu các đoạn trích nhiễu hoặc thông tin hỗ trợ thứ yếu chiếm vị trí ưu tiên.

### 1.3. Phân định rõ ràng các khái niệm đánh giá

> [!CAUTION]
> **Phân biệt giữa Chất lượng Retrieval và Độ chính xác câu trả lời cuối cùng:**
> Các chỉ số trong tài liệu này đo lường chất lượng của tầng Retrieval đối chiếu với ground truth phân cấp của các chuyên viên pháp lý. Chúng **KHÔNG** đồng nghĩa với độ chính xác câu trả lời cuối cùng của Chatbot (Generation Accuracy).

Cần phân định rành mạch 4 khái niệm kỹ thuật:
1. **Chất lượng truy xuất (Retrieval Quality)**: Tỷ lệ các đoạn pháp lý liên quan được tìm thấy trong tập kết quả trả về Top-K (đo bằng `Hit@K`, `Recall@K`, `Precision@K`).
2. **Chất lượng xếp hạng (Ranking Quality)**: Mức độ ưu tiên đưa đoạn pháp lý quan trọng nhất lên đầu danh sách (đo bằng `MRR`, `nDCG@K`).
3. **Điểm nội bộ của retriever**: Cosine similarity của HNSW, BM25 score và RRF score phản ánh thang xếp hạng riêng của thuật toán tương ứng. Đây là **tín hiệu chẩn đoán**, không phải metric chất lượng, và không được so sánh hoặc cộng trực tiếp giữa các phương pháp.
4. **Độ chính xác sinh câu trả lời (Generation / End-to-End Accuracy)**: Mức độ chuẩn xác, đầy đủ và trung thực (faithfulness) của văn bản câu trả lời do LLM sinh ra dựa trên ngữ cảnh đã truy xuất. Retrieval là tiền đề cần thiết: nếu Retrieval thất bại, LLM gần như chắc chắn sẽ hallucinate (bịa đặt) hoặc từ chối trả lời; tuy nhiên, Retrieval tốt vẫn có thể dẫn đến câu trả lời kém nếu LLM suy luận sai.

---

## 2. Thuật ngữ và Khái niệm cơ bản

Để giúp người đọc dễ tiếp cận tài liệu dù chưa quen thuộc với các thuật ngữ Information Retrieval (IR), dưới đây là giải thích chi tiết các khái niệm nền tảng:

* **Query (Truy vấn / Câu hỏi)**: Câu hỏi bằng ngôn ngữ tự nhiên tiếng Việt do người dùng nhập vào hệ thống (ví dụ: *"Sinh viên sư phạm được hỗ trợ tiền đóng học phí và sinh hoạt phí bao nhiêu một tháng?"*).
* **Answerable Query (Truy vấn có thể trả lời)**: Câu hỏi mà nội dung căn cứ pháp lý để giải quyết đã tồn tại trong cơ sở dữ liệu của hệ thống. Bộ benchmark `EvalQueryV2` có **95 câu hỏi** loại này.
* **Out-of-Scope Query - OOS (Truy vấn ngoài phạm vi)**: Câu hỏi hỏi về các văn bản pháp luật hoặc lĩnh vực không nằm trong phạm vi cơ sở dữ liệu của chatbot (ví dụ: hỏi về Luật Lao động 2019, Bộ luật Hình sự).
* **Invalid Query (Truy vấn không hợp lệ / Nhiễu)**: Câu hỏi chứa ký tự ngẫu nhiên vô nghĩa (ví dụ: `asdfghjkl qwerty 12345`) hoặc câu hỏi hoàn toàn lạc đề (ví dụ: giá vàng, thời tiết, tỷ giá ngoại tệ).
* **Chunk (Đoạn trích dữ liệu)**: Đơn vị văn bản nhỏ nhất được chia cắt từ tài liệu gốc, lưu trữ trong cơ sở dữ liệu và được biểu diễn bằng một vector embedding (toàn bộ corpus hiện có **618 chunks**).
* **Ground Truth (Chân lý mặt đất / Dữ liệu đối sánh chuẩn)**: Tập hợp các chunk pháp lý thực tế đã được chuyên viên pháp lý xác thực là nguồn thông tin chính xác và đầy đủ để trả lời cho từng câu hỏi.
* **Relevant Chunk (Chunk liên quan)**: Chunk được xác định là có liên quan đến câu hỏi. Trong hệ thống này, chunk được coi là liên quan khi có điểm mức độ liên quan `grade >= 2`.
* **Graded Relevance (Mức độ liên quan phân cấp / chia mức)**: Cơ chế chấm điểm độ phù hợp của chunk theo thang điểm nhiều mức (từ 0 đến 3) thay vì chỉ nhị phân (Đúng/Sai).

---

## 3. Ground Truth và Thang đo Mức độ liên quan (`EvalQueryV2`)

Tập benchmark **`EvalQueryV2`** (`scripts/evaluation/eval_queries_v2.py`) là nguồn sự thật duy nhất và mới nhất của dự án, khắc phục toàn diện các hạn chế của tập dữ liệu thử nghiệm ban đầu.

### 3.1. Cấu trúc phân cấp văn bản pháp luật
Trong cơ sở dữ liệu, các chunk pháp lý được tổ chức theo hệ thống thứ bậc chuẩn mực của văn bản quy phạm pháp luật Việt Nam:

```text
Văn bản (Document)
  └── Chương (Chapter)
        └── Điều (Article)
              └── Khoản (Clause)
                    └── Điểm (Point)
```

* **Văn bản (`Document`)**: Tên định danh chính thức của văn bản (ví dụ: *Nghị định 116/2020/NĐ-CP*, *Nghị định 60/2025/NĐ-CP*, *Luật Giáo dục 2019*).
* **Chương (`Chapter`)**: Nhóm các điều luật có cùng nhóm quan hệ điều chỉnh (ví dụ: *Chương II: Chính sách hỗ trợ*).
* **Điều (`Article`)**: Đơn vị quy phạm độc lập xác định quyền, nghĩa vụ hoặc định nghĩa (ví dụ: *Điều 4. Mức hỗ trợ và thời gian hỗ trợ*).
* **Khoản (`Clause`)**: Phân đoạn nội dung có đánh số 1, 2, 3... trong một Điều, quy định các trường hợp cụ thể.
* **Điểm (`Point`)**: Tiểu mục liệt kê chi tiết a, b, c... trong một Khoản.

### 3.2. Thang đo 4 mức độ liên quan (Graded Scale)

Mỗi câu hỏi trong `EvalQueryV2` định nghĩa thuộc tính `relevant_chunks: dict[str, int]`, liên kết định danh chunk với điểm số liên quan:

| Mức điểm | Tên gọi | Định nghĩa & Ý nghĩa thực tiễn | Ví dụ minh họa |
|:---:|---|---|---|
| **`3`** | **Rất liên quan (Core / Direct Answer)** | Đoạn trích chứa thông tin trả lời trực tiếp, đầy đủ cho câu hỏi. Đây là căn cứ pháp lý quan trọng nhất mà chatbot bắt buộc phải tìm thấy để trả lời chính xác. | Câu hỏi: *"Điều 4 ND116 quy định mức hỗ trợ sinh hoạt phí là bao nhiêu?"* → Chunk `116-2020-ND-CP-dieu-4-khoan-1` và QA chunk `hoi-dap-nghi-dinh-116-2020-nd-cp-qa-026` (trả lời trực tiếp) đạt **grade 3**. |
| **`2`** | **Liên quan (Supporting Context)** | Đoạn trích cung cấp thông tin bổ trợ có giá trị cao, ví dụ: thẩm quyền phê duyệt, thời hạn nộp hồ sơ, công thức tính toán, trường hợp miễn trừ. Chunk QA có thể đạt grade 2 nếu cung cấp bằng chứng hỗ trợ quan trọng. | Chunk `hoi-dap-nghi-dinh-116-2020-nd-cp-qa-031` giải thích các trường hợp miễn bồi hoàn đạt **grade 2** trong các câu hỏi liên quan. |
| **`1`** | **Liên quan yếu (Weak / Contextual)** | Đoạn trích mang tính chất ngữ cảnh rộng hoặc định nghĩa khái niệm nền tảng trong cùng văn bản, có liên quan gián tiếp nhưng không trực tiếp giải quyết vấn đề. | Chunk `116-2020-ND-CP-dieu-1` về phạm vi điều chỉnh đối với câu hỏi về hồ sơ đạt **grade 1**. |
| **`0`** | **Không liên quan (Irrelevant)** | Đoạn trích về chủ đề khác hoặc điều khoản không liên quan. Chunk QA không được liệt kê trong `relevant_chunks` của một câu hỏi cụ thể thì mặc định có grade 0 đối với câu hỏi đó. | Điều khoản về trách nhiệm của UBND tỉnh đối với câu hỏi về thủ tục cá nhân sinh viên nộp hồ sơ đạt **grade 0**. |

### 3.3. Quy tắc đối sánh và Nguồn pháp lý

> [!IMPORTANT]
> **QA-reference chunks có thể có grade 2 hoặc 3.** Đây là điểm thay đổi quan trọng so với design ban đầu. Hàm `_is_legal_source()` trong `matching.py` chỉ phân biệt `legal_text` vs `qa` cho mục đích hiển thị và routing — **không** ép grade về 0 trong evaluation.

1. **Chunks QA có thể được tính là relevant**: Hàm `_is_legal_source()` (trong `scripts/evaluation/matching.py`) phân loại chunk theo `content_type`. Tuy nhiên, đây chỉ là classifier cho mục đích hiển thị ("Source: QA" vs structural legal metadata). Trong evaluation metrics, **grade của chunk QA được lấy trực tiếp từ `relevant_chunks` dict trong dataset** — không có cơ chế tự động ép grade về 0. Các chunk QA như `hoi-dap-nghi-dinh-116-2020-nd-cp-qa-*` được gán grade 2 hoặc 3 trong nhiều câu hỏi nếu chúng thực sự cung cấp bằng chứng hỗ trợ trực tiếp.
2. **Quy tắc nhị phân hóa (`min_grade = 2`)**:
   * Khi tính toán các chỉ số nhị phân (`Hit@K`, `Recall@K`, `Precision@K`, `MRR`), hệ thống áp dụng ngưỡng:
     $$\text{Relevant} \iff \text{grade} \ge 2$$
   * Các chunk có `grade = 1` được coi là thông tin phụ trợ, không được tính vào số lượng chunk thỏa mãn để đảm bảo đánh giá khắt khe chất lượng truy xuất pháp lý. Áp dụng cho cả legal text lẫn QA chunks.
3. **Quy tắc tính toán đa mức (`nDCG@K`)**:
   * Toàn bộ các mức điểm **`1`, `2`, `3`** được đưa vào công thức nDCG có trọng số, giúp ghi nhận công bằng việc mô hình tìm thấy cả thông tin trực tiếp lẫn thông tin ngữ cảnh hỗ trợ. Điều này bao gồm cả QA chunks được gán grade 1/2/3.
4. **`relevant_documents` và `_QA116`**: Field `relevant_documents` trong mỗi query chứa `_QA116` khi có ít nhất một QA chunk được liệt kê trong `relevant_chunks`. Đây là metadata cho mục đích truy vết, không được evaluation code sử dụng trực tiếp để tính metric.

---

## 4. Các Chỉ số Đánh giá Chi tiết (Metrics)

Phần này giải thích chi tiết từng chỉ số đo lường, bao gồm: bản chất đo lường, khoảng giá trị, đơn vị, công thức tính toán, diễn giải thực tiễn và ví dụ cụ thể.

### 4.1. `Hit@K` (Tỷ lệ tìm thấy trong Top-K)

* **Bản chất đo lường**: Xác định xem hệ thống có tìm thấy *ít nhất một* đoạn trích liên quan (`grade >= 2`) trong K kết quả đầu tiên hay không.
* **Khoảng giá trị / Đơn vị**: `0.0` đến `1.0` (tương ứng `0%` đến `100%`).
* **Ý nghĩa 1 đơn vị**:
  * Đối với một câu hỏi đơn lẻ: Chỉ nhận giá trị nhị phân `1.0` (Thành công - có ít nhất 1 chunk liên quan) hoặc `0.0` (Thất bại - không có chunk liên quan nào).
  * Đối với toàn bộ tập đánh giá: Là tỷ lệ phần trăm số câu hỏi được giải quyết thành công ở ngưỡng K. Ví dụ: $\text{Hit@5} = 0.70$ nghĩa là có **khoảng 70%** số câu hỏi có ít nhất một đoạn pháp lý đúng nằm trong Top-5 kết quả trả về.
* **Công thức tính**:
  $$\text{Hit@K} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left( \exists c \in \text{Top-K}_i : \text{grade}(c) \ge 2 \right)$$
  *(với $N = 95$ là tổng số câu hỏi có thể trả lời, $\mathbb{I}$ là hàm chỉ thị).*
* **Chiều hướng mong muốn**: Giá trị càng cao càng tốt (tiến gần 1.0).
* **Ví dụ thực tế**:
  > Câu hỏi về cách làm tròn số tháng công tác (Điều 2 Khoản 1 NĐ 116).
  > Hệ thống trả về 10 kết quả. Nếu chunk `116-2020-ND-CP-dieu-2-khoan-1` (grade 3) xuất hiện ở Rank 2, thì với $K \ge 2$ ($K=3, 5, 10$), $\text{Hit@K} = 1.0$.

### 4.2. `Recall@K` (Độ bao phủ / Tỷ lệ thu hồi trong Top-K)

* **Bản chất đo lường**: Tỷ lệ giữa số lượng chunk liên quan (`grade >= 2`) được tìm thấy trong Top-K so với *tổng số chunk liên quan kỳ vọng* của câu hỏi đó.
* **Khoảng giá trị / Đơn vị**: `0.0` đến `1.0` (`0%` đến `100%`).
* **Ý nghĩa 1 đơn vị**:
  * Thể hiện mức độ "thu hồi trọn vẹn" các căn cứ pháp lý.
  * Chỉ số này cực kỳ quan trọng đối với các câu hỏi phức tạp đòi hỏi căn cứ từ nhiều điều khoản (ví dụ: vừa xét điều kiện bồi hoàn theo Điều 6 vừa tính tiền bồi hoàn theo Điều 8).
* **Công thức tính**:
  $$\text{Recall@K} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\{c \in \text{Top-K}_i : \text{grade}(c) \ge 2\}|}{|\{c \in \text{GroundTruth}_i : \text{grade}(c) \ge 2\}|}$$
* **Chiều hướng mong muốn**: Giá trị càng cao càng tốt (tiến gần 1.0).
* **Ví dụ thực tế**:
  > Câu hỏi: *"Quy định về bồi hoàn kinh phí hỗ trợ gồm những trường hợp nào và chi phí tính ra sao?"*
  > Ground truth kỳ vọng 2 chunk có `grade >= 2`:
  > 1. `116-2020-ND-CP-dieu-6-khoan-1` (Đối tượng bồi hoàn - grade 3)
  > 2. `116-2020-ND-CP-dieu-8-khoan-1` (Chi phí bồi hoàn - grade 3)
  > Trong Top-5 kết quả trả về: Hệ thống tìm thấy chunk Điều 6 ở Rank 1, nhưng chunk Điều 8 nằm ở Rank 7.
  > Khi đó:
  > * Tại $K = 5$: Chỉ tìm thấy 1/2 chunk $\implies \text{Recall@5} = 0.50$ (50%).
  > * Tại $K = 10$: Tìm thấy cả 2 chunk $\implies \text{Recall@10} = 1.00$ (100%).

### 4.3. `Precision@K` (Độ chính xác danh sách Top-K)

* **Bản chất đo lường**: Tỷ lệ giữa số chunk thực sự liên quan (`grade >= 2`) nằm trong Top-K so với tổng số $K$ kết quả mà hệ thống trả về.
* **Khoảng giá trị / Đơn vị**: `0.0` đến `1.0`.
* **Ý nghĩa 1 đơn vị**:
  * Đo lường "mật độ thông tin hữu ích" trong ngữ cảnh gửi tới LLM.
  * Nếu $\text{Precision@5} = 0.40$, nghĩa là trong 5 chunk trả về, trung bình có 2 chunk ($40\%$) là thông tin pháp lý đúng, còn 3 chunk ($60\%$) là thông tin phụ trợ, chunk QA hoặc không liên quan.
* **Công thức tính**:
  $$\text{Precision@K} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\{c \in \text{Top-K}_i : \text{grade}(c) \ge 2\}|}{K}$$
* **Chiều hướng mong muốn**: Giá trị càng cao càng tốt. Lưu ý rằng khi $K$ tăng, Precision thường giảm do số lượng chunk liên quan thực tế của một câu hỏi pháp lý thường chỉ có từ 1 đến 3 chunks.

### 4.4. `MRR` (Mean Reciprocal Rank - Thứ hạng nghịch đảo trung bình)

* **Bản chất đo lường**: Đánh giá vị trí xuất hiện của **chunk liên quan đầu tiên** (`grade >= 2`).
* **Khoảng giá trị / Đơn vị**: `0.0` đến `1.0`.
* **Ý nghĩa 1 đơn vị**:
  * Phản ánh năng lực đưa kết quả đúng lên đỉnh danh sách (Rank 1):
    * Chunk đúng đầu tiên ở **Rank 1**: Điểm Reciprocal Rank $= 1/1 = \mathbf{1.0}$
    * Chunk đúng đầu tiên ở **Rank 2**: Điểm Reciprocal Rank $= 1/2 = \mathbf{0.5}$
    * Chunk đúng đầu tiên ở **Rank 3**: Điểm Reciprocal Rank $= 1/3 \approx \mathbf{0.333}$
    * Chunk đúng đầu tiên ở **Rank 5**: Điểm Reciprocal Rank $= 1/5 = \mathbf{0.200}$
    * Chunk đúng đầu tiên ở **Rank 10**: Điểm Reciprocal Rank $= 1/10 = \mathbf{0.100}$
    * Không tìm thấy chunk đúng nào trong Top-10: Điểm $= \mathbf{0.0}$
* **Công thức tính**:
  $$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$$
  *(với $\text{rank}_i$ là thứ hạng của chunk có $\text{grade} \ge 2$ xuất hiện sớm nhất trong truy vấn $i$).*
* **Chiều hướng mong muốn**: Càng cao càng tốt (tiến gần 1.0). MRR cao chứng tỏ hệ thống thường xuyên xếp đoạn trích quan trọng nhất ngay tại Rank 1 hoặc Rank 2.

### 4.5. `nDCG@K` (Normalized Discounted Cumulative Gain)

* **Bản chất đo lường**: Thước đo toàn diện nhất về chất lượng xếp hạng có trọng số, thưởng cho hệ thống khi xếp các chunk có độ liên quan cao (`grade = 3`) lên trước các chunk có độ liên quan thấp hơn (`grade = 2, 1`), và phạt nặng nếu đẩy kết quả quan trọng xuống các vị trí thấp.
* **Khoảng giá trị / Đơn vị**: `0.0` đến `1.0`. Điểm `1.0` là chất lượng xếp hạng hoàn hảo tuyệt đối (Ideal Ranking).
* **Cơ chế chiết khấu thứ hạng logarithm (Logarithmic Ranking Discount)**:
  * Trong thực tế, người dùng (và mô hình LLM) luôn chú ý nhiều nhất đến các vị trí đầu tiên. Càng trôi xuống dưới danh sách, giá trị thông tin đóng góp càng suy giảm.
  * Hệ thống áp dụng hệ số phạt $\log_2(i + 1)$ cho vị trí thứ $i$:
    * Vị trí 1: Chia cho $\log_2(1+1) = 1.0$ (Giữ nguyên toàn bộ giá trị điểm)
    * Vị trí 3: Chia cho $\log_2(3+1) = 2.0$ (Giá trị điểm bị giảm một nửa)
    * Vị trí 7: Chia cho $\log_2(7+1) = 3.0$ (Giá trị điểm chỉ còn 1/3)
* **Công thức tính toán**:
  1. Tính độ lợi tích lũy chiết khấu thực tế (**DCG@K**):
     $$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{\text{grade}_i} - 1}{\log_2(i + 1)}$$
     *(Mức grade $3 \implies 2^3 - 1 = 7$; grade $2 \implies 2^2 - 1 = 3$; grade $1 \implies 2^1 - 1 = 1$; grade $0 \implies 0$).*
  2. Tính độ lợi tích lũy lý tưởng (**IDCG@K**):
     Sắp xếp toàn bộ các chunk ground truth của câu hỏi theo thứ tự điểm giảm dần ($3, 3, 2, 1...$) rồi tính DCG lý tưởng ở ngưỡng K.
  3. Chuẩn hóa (**nDCG@K**):
     $$\text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
* **Chiều hướng mong muốn**: Càng cao càng tốt (tiến gần 1.0).

### 4.6. `Average Top-1 Similarity` (Điểm tương đồng Cosin Top-1 trung bình)

* **Bản chất đo lường**: Trung bình cộng độ tương đồng cosin giữa vector truy vấn và vector của chunk xếp hạng cao nhất (Rank 1) trên toàn bộ 95 câu hỏi.
* **Khoảng giá trị / Đơn vị**: Lý thuyết từ `-1.0` đến `1.0`; trong thực nghiệm tìm kiếm vector văn bản pháp luật thường dao động từ `0.40` đến `0.95`.
* **Ý nghĩa & Cảnh báo sử dụng**:
  * Là **tín hiệu chẩn đoán kỹ thuật**: Cho biết khoảng cách phân bố hình học bên trong không gian vector của một mô hình cụ thể.
  * **Tuyệt đối không dùng để xếp hạng mô hình**: Thang đo khoảng cách giữa các mô hình khác nhau là không tương đương. Mô hình có điểm tương đồng cao chưa chắc đã truy xuất chính xác hơn (xem phân tích đối chứng chi tiết tại Mục 11).

---

## 5. Các Tham số Kỹ thuật và Điều kiện Thực nghiệm

Để đảm bảo tính khách quan và khả năng tái lập, toàn bộ các tham số thử nghiệm được cố định chuẩn hóa:

### 5.1. Tham số chỉ mục vector HNSW
Thuật toán HNSW (Hierarchical Navigable Small World) xây dựng đồ thị nhiều tầng để tìm kiếm vector láng giềng gần nhất:
* **`M` (hoặc `m = 16`)**:
  * **Khái niệm**: Số lượng liên kết hai chiều tối đa được tạo cho mỗi điểm vector trên mỗi tầng của đồ thị HNSW.
  * **Ý nghĩa thực tế**: Giá trị `16` đảm bảo mỗi đoạn văn bản trong cơ sở dữ liệu được kết nối chặt chẽ với 16 đoạn văn bản gần nhất về mặt ngữ nghĩa. Giá trị này giúp cân bằng hoàn hảo giữa dung lượng RAM sử dụng và độ liên kết của đồ thị.
* **`ef_construction = 64`**:
  * **Khái niệm**: Kích thước hàng đợi ưu tiên động được sử dụng trong quá trình *xây dựng chỉ mục (Indexing time)*.
  * **Ý nghĩa thực tế**: Khi thêm một vector mới vào database, thuật toán đánh giá 64 điểm láng giềng tiềm năng để tìm ra các cạnh nối tối ưu nhất. Giá trị 64 mang lại chất lượng đồ thị cao mà không làm chậm quá trình import dữ liệu.
* **`ef_search = 80`**:
  * **Khái niệm**: Kích thước hàng đợi ưu tiên động được sử dụng trong quá trình *truy vấn tìm kiếm (Query/Search time)*.
  * **Ý nghĩa thực tế**: Khi nhận câu hỏi của người dùng, thuật toán HNSW duyệt qua 80 ứng viên tiềm năng trên đồ thị trước khi chọn ra Top-K kết quả cuối cùng. `ef_search = 80` cao hơn đáng kể so với mức mặc định thông thường (40), giúp tối đa hóa khả năng thu hồi (Recall) các điều khoản khó mà chỉ làm tăng độ trễ tìm kiếm thêm vài mili-giây.

### 5.2. Tham số `Top-K`
* **Khái niệm**: Số lượng đoạn trích có điểm tương đồng cao nhất được trích xuất từ cơ sở dữ liệu.
* **Ý nghĩa đánh giá**: Hệ thống truy xuất cố định `Top-10` chunks cho mỗi truy vấn, sau đó tính toán và đối chiếu các chỉ số tại 3 mốc:
  * **`Top-3`**: Đo lường khả năng cô đọng thông tin trong phạm vi ngắn (cực kỳ quan trọng để tiết kiệm context window của LLM).
  * **`Top-5`**: Ngưỡng context tiêu chuẩn thường được đưa vào prompt của RAG Chatbot.
  * **`Top-10`**: Ngưỡng bao phủ tối đa để đánh giá khả năng không bỏ sót thông tin của mô hình embedding.

### 5.3. Chiều vector (Embedding Dimension)
* **Khái niệm**: Số lượng phần tử số thực (floating-point numbers) trong mảng vector biểu diễn ngữ nghĩa của văn bản.
* **Ý nghĩa thực tế**: Tất cả **5 mô hình** được cấu hình chuẩn hóa ở **1024 chiều** (trong $\mathbb{R}^{1024}$). Cần phân biệt rõ:
  * **Số chiều vector (1024)**: Độ chi tiết của biểu diễn ngữ nghĩa (1 vector chiếm $\approx 4\text{ KB}$ lưu trữ).
  * **Số lượng chunks (618)**: Tổng số bản ghi văn bản pháp lý lưu trong bảng `legal_chunks`.

### 5.4. Độ phủ nhúng (Coverage)
* **Độ phủ: `618 / 618` (100.0%)**:
  Toàn bộ 618 đoạn văn bản pháp luật hiện có trong cơ sở dữ liệu đều đã được sinh vector embedding đầy đủ và không có bất kỳ chunk nào mang giá trị NULL ở cột vector của 6 mô hình.

### 5.5. Thời gian chạy benchmark (Runtime)
* **Ý nghĩa con số thời gian**: Thời gian đo lường trong log benchmark Hybrid+Rerank 2026-10-06 (từ ~154s đến ~200s) là tổng thời gian thực hiện toàn bộ quy trình: nạp mô hình vào GPU, mã hóa 100 câu hỏi liên tiếp, gửi truy vấn qua mạng tới NeonDB, thực hiện HNSW + BM25 + RRF + CrossEncoder reranking và tính toán metrics đối sánh.
* **Cảnh báo thống kê**: Con số thời gian ghi nhận từ một lần chạy duy nhất, phụ thuộc vào độ trễ mạng internet tới máy chủ NeonDB và tải phần cứng GPU tại thời điểm đó, không nên được coi là chuẩn đo lường tốc độ tuyệt đối giữa các kiến trúc.

---

## 6. Cấu hình Kỹ thuật 5 Mô hình Embedding

Toàn bộ **5 mô hình** đều sinh vector **1024 chiều**, sử dụng độ đo **cosine similarity** và được lưu trữ cách ly ở các cột riêng trong bảng `legal_chunks`:

| Model Identifier | Model Alias | Chiều vector | Backend | Cột Embedding | Database |
|---|---|---:|---|---|---|
| `BAAI/bge-m3` | `bge-m3` | 1024 | `bge` | `embedding` | `neondb` |
| `darklethelong/vnlegal-lal` | `vnlegal-lal` | 1024 | `sentence_transformer` | `embedding_vnlegal_lal` | `neondb` |
| `mainguyen9/vietlegal-harrier-0.6b` | `vietlegal-harrier` | 1024 | `sentence_transformer` | `embedding_vietlegal_harrier` | `neondb` |
| `mainguyen9/vietlegal-e5` | `vietlegal-e5` | 1024 | `sentence_transformer` | `embedding_vietlegal_e5` | `neondb` |
| `jinaai/jina-embeddings-v3-hf` | `jina-v3` | 1024 | `jina` | `embedding_jina_v3` | `neondb` |

> [!NOTE]
> Top-K và ef_search phụ thuộc vào cấu hình benchmark, không phải cố định theo model. Benchmark mới nhất (2026-10-06) dùng `candidate-k=40`, `top-k=20`, `ef_search=80`, `rerank-top-k=10`. Xem Section 7.1 để biết cấu hình đầy đủ.

---

## 7. Cập nhật Benchmark Mới nhất (Sau Dataset Update — 2026-10-06)

### 7.1. Benchmark Configuration

| Tham số | Giá trị |
|---|---|
| Dataset | `EvalQueryV2` (sau dataset update) |
| Corpus size | 618 chunks (100% embedded) |
| Total queries | 100 (95 answerable, 5 OOS/Invalid) |
| Retrieval method | HYBRID: HNSW + BM25 → RRF → CrossEncoder Rerank |
| HNSW `candidate-k` | 40 |
| BM25 `candidate-k` | 40 |
| RRF `k` | 60 |
| `ef_search` | 80 |
| `top-k` (trả về) | 20 |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| `rerank-top-k` | 10 |
| Evaluation cut-offs | @3, @5, @10 |
| Relevance rule | Hit/Recall/Precision/MRR: grade ≥ 2; nDCG: grade 1–3 |
| Embedding dimension | 1024 (tất cả 5 models) |
| Database | NeonDB (PostgreSQL + pgvector) |

> [!NOTE]
> Dataset `EvalQueryV2` đã được cập nhật để bổ sung các QA-reference chunks (`hoi-dap-nghi-dinh-116-2020-nd-cp-qa-*`) vào `relevant_documents` với `grade 2` hoặc `3` nếu thực sự cung cấp bằng chứng hỗ trợ trực tiếp. Trước khi cập nhật, các chunk QA này bị gán `grade 0` (không liên quan). Đây là thay đổi **evaluation semantics** quan trọng cần lưu ý khi đối chiếu với kết quả benchmark cũ.

---

### 7.2. Overall Results — Bảng tổng hợp 5 models

Tất cả 5 benchmark được chạy vào ngày **2026-10-06** trên cùng corpus, cùng dataset và cùng cấu hình Hybrid+Rerank:

| Model | Alias | Runtime | Hit@3 | Hit@5 | Hit@10 | Recall@3 | Recall@5 | Recall@10 | Precision@3 | Precision@5 | Precision@10 | MRR | nDCG@3 | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `BAAI/bge-m3` | `bge-m3` | 197.27s | 0.8421 | 0.8737 | 0.9053 | 0.4263 | 0.5244 | **0.6169** | **0.4702** | **0.3453** | **0.2126** | **0.7568** | **0.5317** | **0.5409** | **0.5713** |
| `mainguyen9/vietlegal-harrier-0.6b` | `vietlegal-harrier` | 190.04s | 0.8316 | **0.9158** | **0.9368** | **0.4344** | **0.5355** | 0.6124 | 0.4632 | 0.3389 | 0.2042 | 0.7299 | 0.5231 | 0.5359 | 0.5626 |
| `mainguyen9/vietlegal-e5` | `vietlegal-e5` | 193.47s | **0.8421** | 0.8947 | 0.9158 | 0.4273 | 0.5091 | 0.5944 | 0.4596 | 0.3347 | 0.2042 | 0.7362 | 0.5105 | 0.5156 | 0.5468 |
| `jinaai/jina-embeddings-v3-hf` | `jina-v3` | 177.13s | 0.7895 | 0.8526 | 0.8632 | 0.4064 | 0.4770 | 0.5385 | 0.4211 | 0.3011 | 0.1811 | 0.7028 | 0.5083 | 0.5117 | 0.5318 |
| `darklethelong/vnlegal-lal` | `vnlegal-lal` | 183.77s | 0.7789 | 0.8211 | 0.8526 | 0.3985 | 0.4683 | 0.5176 | 0.4281 | 0.3011 | 0.1726 | 0.6875 | 0.4890 | 0.4887 | 0.5023 |

> **In đậm** = giá trị cao nhất trong cột đó.

---

### 7.3. Category-level Results

#### 7.3.1. Category: `semantic` (n=20)

Queries về khái niệm, chính sách chung, không trích dẫn số điều khoản cụ thể.

| Model | Hit@3 | Hit@10 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| `bge-m3` | **1.0000** | **1.0000** | **0.7375** | **0.9500** | **0.7534** |
| `vietlegal-harrier` | **1.0000** | **1.0000** | 0.7150 | **0.9750** | 0.7459 |
| `vietlegal-e5` | **1.0000** | **1.0000** | 0.7275 | **0.9500** | 0.7488 |
| `jina-v3` | 0.9000 | 0.9000 | 0.5425 | 0.8500 | 0.6448 |
| `vnlegal-lal` | 0.9500 | 0.9500 | 0.6508 | 0.9000 | 0.6892 |

**Nhận xét**: `bge-m3`, `vietlegal-harrier` và `vietlegal-e5` đều đạt Hit@10 = 1.0 (tìm thấy ít nhất 1 chunk liên quan ở tất cả 20 semantic queries). `jina-v3` yếu nhất ở category này với Recall@10 chỉ 0.5425, thấp hơn đáng kể.

#### 7.3.2. Category: `exact` (n=15)

Queries trích dẫn chính xác số Điều, Khoản, Điểm cụ thể.

| Model | Hit@3 | Hit@10 | Recall@3 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| `bge-m3` | **0.8000** | **0.8000** | **0.5556** | **0.6667** | **0.7222** | **0.6009** |
| `vietlegal-harrier` | 0.6000 | 0.7333 | 0.4889 | 0.5444 | 0.6000 | 0.5232 |
| `vietlegal-e5` | 0.6667 | 0.7333 | 0.5000 | 0.5444 | 0.5722 | 0.4997 |
| `jina-v3` | 0.7333 | 0.7333 | **0.5556** | 0.6111 | 0.6111 | 0.5445 |
| `vnlegal-lal` | 0.6000 | 0.6667 | 0.4833 | 0.5833 | 0.5833 | 0.5165 |

**Nhận xét**: `bge-m3` dẫn đầu rõ ràng ở exact lookup với Hit@3 = 0.80 và Recall@10 = 0.6667. `vietlegal-harrier` và `vnlegal-lal` yếu hơn đáng kể ở category này (Hit@10 ≤ 0.73, Hit@3 ≤ 0.60). `jina-v3` có Recall@3 ngang `bge-m3` nhưng Hit@3 thấp hơn — cho thấy jina-v3 tìm được chunk nhưng không đưa lên vị trí đầu tốt bằng.

#### 7.3.3. Category: `contextual` (n=25)

Queries hỏi về ngữ cảnh áp dụng, điều kiện, trường hợp cụ thể mà không trích dẫn trực tiếp.

| Model | Hit@3 | Hit@10 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| `bge-m3` | 0.8000 | 0.8800 | 0.5701 | 0.6700 | 0.5537 |
| `vietlegal-harrier` | 0.8000 | **0.9200** | **0.6154** | 0.6633 | 0.5441 |
| `vietlegal-e5` | 0.7600 | **0.9200** | 0.5864 | 0.6067 | 0.5040 |
| `jina-v3` | **0.8000** | 0.8400 | 0.5507 | 0.6467 | **0.5421** |
| `vnlegal-lal` | 0.6800 | 0.7200 | 0.4117 | 0.5367 | 0.4009 |

**Nhận xét**: `vietlegal-harrier` và `vietlegal-e5` đồng dẫn đầu về Hit@10 (0.92) và `vietlegal-harrier` có Recall@10 cao nhất nhóm (0.6154). `vnlegal-lal` yếu rõ rệt ở contextual queries (Hit@10 chỉ 0.72, Recall@10 chỉ 0.4117).

#### 7.3.4. Category: `complex_qa` (n=10)

Queries phức tạp, yêu cầu tổng hợp nhiều điều khoản hoặc suy luận đa bước.

| Model | Hit@3 | Hit@10 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| `bge-m3` | **1.0000** | **1.0000** | **0.5467** | **0.9000** | 0.5503 |
| `vietlegal-harrier` | 0.9000 | **1.0000** | 0.5234 | 0.8700 | **0.5659** |
| `vietlegal-e5` | **1.0000** | **1.0000** | 0.4851 | **0.9000** | 0.5189 |
| `jina-v3` | 0.9000 | 0.9000 | 0.3691 | 0.8000 | 0.4530 |
| `vnlegal-lal` | 0.9000 | 0.9000 | 0.3527 | 0.8500 | 0.4187 |

**Nhận xét**: `bge-m3`, `vietlegal-harrier`, `vietlegal-e5` đều đạt Hit@10 = 1.0 trên complex_qa. Nhưng Recall@10 chỉ đạt 0.49–0.55, phản ánh việc chưa thu hồi đủ *tất cả* các chunks cần thiết cho các câu hỏi đòi hỏi nhiều căn cứ. `jina-v3` và `vnlegal-lal` có Recall@10 thấp đáng kể (0.37 và 0.35).

#### 7.3.5. Category: `multi_chunk` (n=15)

Queries cần nhiều chunks từ cùng một văn bản (ví dụ: cần đọc cả Khoản 1 lẫn Khoản 2, Khoản 3 của cùng một Điều).

| Model | Hit@3 | Hit@10 | Recall@3 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| `bge-m3` | 0.6667 | 0.8000 | 0.2364 | 0.4196 | 0.5784 | 0.3686 |
| `vietlegal-harrier` | 0.6667 | **0.8667** | 0.2402 | 0.4100 | 0.5433 | 0.3658 |
| `vietlegal-e5` | **0.7333** | 0.7333 | 0.2364 | 0.3841 | 0.5000 | 0.3343 |
| `jina-v3` | 0.6667 | 0.7333 | 0.2231 | 0.4313 | 0.5333 | 0.3585 |
| `vnlegal-lal` | 0.5333 | 0.6667 | 0.1723 | 0.3426 | 0.4911 | 0.3018 |

**Nhận xét**: Đây là category **yếu nhất của tất cả models**. Recall@10 chỉ đạt 0.34–0.43, nghĩa là hệ thống chỉ thu hồi được khoảng hơn 1/3 số chunks cần thiết. Cần lưu ý Recall thấp một phần do ground truth đòi hỏi nhiều chunks.

#### 7.3.6. Category: `multi_document` (n=10)

Queries yêu cầu tổng hợp thông tin từ nhiều văn bản khác nhau (ví dụ: ND116 + ND60 + Luật Giáo dục 2019).

| Model | Hit@3 | Hit@10 | Recall@3 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| `bge-m3` | 0.7000 | 0.7000 | 0.2486 | 0.4062 | 0.6500 | 0.3633 |
| `vietlegal-harrier` | **0.8000** | **0.8000** | **0.2827** | 0.3812 | **0.7500** | 0.3764 |
| `vietlegal-e5` | **0.8000** | **0.8000** | **0.2827** | 0.3920 | 0.7000 | 0.3441 |
| `jina-v3` | **0.8000** | **0.8000** | 0.2577 | 0.3768 | 0.7000 | 0.3545 |
| `vnlegal-lal` | 0.7000 | 0.7000 | 0.2486 | 0.4019 | 0.6500 | 0.3231 |

**Nhận xét**: Multi-document cũng là một bottleneck quan trọng — Recall@10 dao động quanh 0.37–0.40. Tương tự `multi_chunk`, Recall thấp một phần do ground truth có nhiều relevant chunks.

---

### 7.4. Phân tích tổng hợp

#### Overall ranking theo metric

| Metric | Dẫn đầu | Ghi chú |
|---|---|---|
| **Hit@10** | `vietlegal-harrier` (0.9368) | Nhỉnh hơn `vietlegal-e5` (0.9158) và `bge-m3` (0.9053) |
| **Recall@10** | `bge-m3` (0.6169) | Cao hơn `vietlegal-harrier` (0.6124) |
| **Precision@10** | `bge-m3` (0.2126) | Dẫn đầu |
| **MRR** | `bge-m3` (0.7568) | Đưa chunk đúng lên đầu tốt nhất |
| **nDCG@10** | `bge-m3` (0.5713) | Ranking quality tốt nhất |
| **Runtime** | `jina-v3` (177s) | Nhanh nhất; `bge-m3` chậm nhất (197s) |

**Nhận xét tổng thể:**

- **`BAAI/bge-m3`** mạnh nhất tổng thể về Recall@10, Precision@10, MRR và nDCG@10.
- **`mainguyen9/vietlegal-harrier-0.6b`** có Hit@10 cao nhất (0.9368), phù hợp khi ưu tiên coverage/hit probability.
- **`mainguyen9/vietlegal-e5`** thuộc nhóm trên thứ ba, kết quả khá cân bằng.
- **`jinaai/jina-embeddings-v3-hf`** thấp hơn nhóm dẫn đầu về coverage.
- **`darklethelong/vnlegal-lal`** là model thấp nhất ở các metric tổng hợp chính.

#### Trade-offs đáng chú ý

- `bge-m3` và `vietlegal-harrier` có trade-off giữa Hit@10 (harrier cao hơn) và Recall/MRR (bge-m3 cao hơn).
- `semantic` category là điểm mạnh của tất cả models; `multi_chunk` và `multi_document` là bottleneck chung.

---

### 7.5. So sánh chênh lệch: 2026-10-02 vs 2026-10-06

Sự thay đổi về cấu hình (thêm Reranker Cross-Encoder với `rerank-top-k=10`) và cập nhật dataset (đưa QA chunks vào ground truth với grade 2/3) đã tạo ra sự thay đổi lớn về chất lượng ranking. So sánh trên cùng mô hình `vietlegal-harrier`:

| Giai đoạn | Cấu hình | Hit@10 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| **2026-10-02** | Hybrid (không rerank), top-k=10 | 0.8947 | 0.5069 | **0.6255** | 0.6144 | 0.5376 |
| **2026-10-06** | Hybrid + Cross-Encoder Rerank, top-k=10 | **0.9368** | **0.5355** | 0.6124 | **0.7299** | **0.5626** |
| **Thay đổi** | | **+0.0421** | **+0.0286** | -0.0131 | **+0.1155** | **+0.0250** |

**Phân tích độ chênh lệch:**
- **MRR tăng vọt (+0.1155)**: Tác dụng rõ rệt nhất của Cross-Encoder Reranker là khả năng đẩy chunk liên quan lên vị trí số 1 (Rank 1).
- **Hit@10 tăng (+0.0421)**: Do candidate pool được mở rộng (`candidate-k=40` thay vì 20) trước khi rerank.
- **Recall@10 giảm nhẹ (-0.0131)**: Nguyên nhân chính là do ground truth được bổ sung thêm QA chunks, làm tăng tổng số lượng relevant chunks cần tìm (mẫu số của Recall tăng lên), khiến Recall bị giảm tương đối mặc dù hệ thống thực tế truy xuất chính xác hơn.

---

### 7.6. Important Evaluation Caveats

> [!WARNING]
> **Thay đổi Evaluation Semantics**: Dataset `EvalQueryV2` đã được cập nhật để đưa QA-reference chunks (`hoi-dap-nghi-dinh-116-2020-nd-cp-qa-*`) vào ground truth với `grade 2/3`. Trước đây, các chunk này bị gán `grade 0` (irrelevant). Việc metric tăng so với benchmark cũ **có thể một phần hoặc đáng kể đến từ việc mở rộng ground truth**, không nhất thiết phản ánh retrieval algorithm thực sự tốt hơn. Không nên so sánh trực tiếp raw scores giữa benchmark trước và sau dataset update như một phép đo thuần túy về model improvement.

> [!CAUTION]
> **Retrieval Quality ≠ Generation Quality**: Các chỉ số `Recall@K`, `Precision@K`, `nDCG@K`, `MRR` chỉ đánh giá khả năng mô hình thu hồi đúng đoạn văn bản làm evidence. Một model có Recall@10 cao không đảm bảo LLM sẽ sinh ra câu trả lời pháp lý chính xác — Generation accuracy còn phụ thuộc vào khả năng lập luận và faithfulness của LLM với ngữ cảnh đã truy xuất.

**Các lưu ý bổ sung:**

1. **Similarity / RRF Score chỉ là diagnostic**: `Average Top-1 RRF score` trong log (`bge-m3`: 0.849, `harrier`: 0.830, `e5`: 0.842, `jina-v3`: 0.818, `vnlegal-lal`: 0.803) **không được dùng để xếp hạng model**. Thang đo khoảng cách của các embedding space khác nhau không tương đương nhau.

2. **QA chunk vs. Legal chunk**: Các chunk QA (`hoi-dap-nghi-dinh-116-2020-nd-cp-qa-*`) được đưa vào ground truth với grade 2/3 dựa trên **evidence value** (giá trị bằng chứng), không phải **source authority** (thẩm quyền pháp lý). Hai khái niệm này khác nhau: chunk QA có thể hỗ trợ trả lời câu hỏi nhưng không thể được trích dẫn như văn bản quy phạm pháp luật chính thức trong thực tiễn pháp lý.

3. **OOS/Invalid behavior**: Tất cả 5 models đều trả về kết quả cho 5 OOS/Invalid queries thay vì từ chối — đây là hành vi kỳ vọng của retriever (không lọc OOS ở tầng retrieval), nhưng cần LLM hoặc tầng guardrail xử lý phân loại ngoài phạm vi.

---

### 7.7. Điểm Cần Lưu Ý (Issues — Verified & Pending)

> [!IMPORTANT]
> **Không có thay đổi code nào được thực hiện.** Các vấn đề được phân loại theo mức độ xác minh.

#### [V-01] VERIFIED — Alias mismatch `vietlegal-lal` vs `vnlegal-lal`

**Evidence:** Model registry trong `scripts/embeddings/model_registry.py` định nghĩa alias là `vnlegal-lal` (không phải `vietlegal-lal`). Command đầu tiên dùng sai alias → exit code 1. Benchmark chỉ hoàn chỉnh sau khi dùng alias đúng.

**Required action:** Chuẩn hóa alias trong tất cả documentation và hướng dẫn tái lập. Lệnh tái lập đúng đã được cập nhật tại Section 7.7.

---

### 7.8. Lệnh tái lập

```powershell
conda activate chatbot
# Chạy lần lượt, mỗi lần một model:
$env:EMBEDDING_MODEL="bge-m3";         python scripts/test_retrieval.py --method hybrid --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80 --rerank --rerank-model BAAI/bge-reranker-v2-m3 --rerank-top-k 10
$env:EMBEDDING_MODEL="vietlegal-harrier"; python scripts/test_retrieval.py --method hybrid --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80 --rerank --rerank-model BAAI/bge-reranker-v2-m3 --rerank-top-k 10
$env:EMBEDDING_MODEL="vietlegal-e5";    python scripts/test_retrieval.py --method hybrid --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80 --rerank --rerank-model BAAI/bge-reranker-v2-m3 --rerank-top-k 10
$env:EMBEDDING_MODEL="jina-v3";         python scripts/test_retrieval.py --method hybrid --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80 --rerank --rerank-model BAAI/bge-reranker-v2-m3 --rerank-top-k 10
$env:EMBEDDING_MODEL="vnlegal-lal";     python scripts/test_retrieval.py --method hybrid --candidate-k 40 --top-k 20 --rrf-k 60 --ef-search 80 --rerank --rerank-model BAAI/bge-reranker-v2-m3 --rerank-top-k 10
```

> [!NOTE]
> Alias `vnlegal-lal` (không phải `vietlegal-lal`) là alias hợp lệ cho model `darklethelong/vnlegal-lal`. Xem `scripts/embeddings/model_registry.py` để tra cứu alias chính xác của từng model.

