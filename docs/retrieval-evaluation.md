# Đánh giá Retrieval (Retrieval Evaluation)

> **LƯU Ý:** Kết quả benchmark hiện tại (mới nhất) được đánh giá trên corpus **618 chunks**. Tuy nhiên, các bảng phân tích chi tiết bên dưới (theo từng nhóm query, từng model) có thể vẫn đang lưu giữ dữ liệu lịch sử từ bản benchmark trước đó trên corpus 618 chunks.



Tài liệu này là **nguồn sự thật duy nhất (single canonical source of truth)** về phương pháp luận đánh giá, định nghĩa và ý nghĩa thực tiễn của các chỉ số, giải thích tham số kỹ thuật, cấu trúc ground truth phân cấp chia mức độ liên quan (**`EvalQueryV2`**), điều kiện thực nghiệm chuẩn hóa, kết quả đo lường toàn diện của **6 mô hình embedding**, phân tích chuyên sâu đa chiều và hướng dẫn thực thi tái lập trên Windows PowerShell.

---

## 1. Mục đích và Phạm vi đánh giá

### 1.1. Mục đích đánh giá
Hệ thống RAG (Retrieval-Augmented Generation) phục vụ tra cứu văn bản quy phạm pháp luật giáo dục (trọng tâm là **Nghị định 116/2020/NĐ-CP**, văn bản sửa đổi bổ sung **Nghị định 60/2025/NĐ-CP**, và **Luật Giáo dục 2019**). Tầng Retrieval chịu trách nhiệm nhận câu hỏi ngôn ngữ tự nhiên tiếng Việt từ người dùng, tìm kiếm trong cơ sở dữ liệu vector và trích xuất ra các đoạn văn bản (chunks) pháp lý liên quan nhất làm ngữ cảnh (context) cho mô hình ngôn ngữ lớn (LLM) sinh câu trả lời.

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
> Các chỉ số trong tài liệu này đo lường chất lượng của tầng **Dense Retrieval (tìm kiếm vector)** đối chiếu với ground truth phân cấp của các chuyên viên pháp lý. Các chỉ số này **KHÔNG** đồng nghĩa với độ chính xác câu trả lời cuối cùng của Chatbot (Generation Accuracy).

Cần phân định rành mạch 4 khái niệm kỹ thuật:
1. **Chất lượng truy xuất (Retrieval Quality)**: Tỷ lệ các đoạn pháp lý liên quan được tìm thấy trong tập kết quả trả về Top-K (đo bằng `Hit@K`, `Recall@K`, `Precision@K`).
2. **Chất lượng xếp hạng (Ranking Quality)**: Mức độ ưu tiên đưa đoạn pháp lý quan trọng nhất lên đầu danh sách (đo bằng `MRR`, `nDCG@K`).
3. **Điểm tương đồng (Similarity Score)**: Giá trị cosin hình học giữa vector câu hỏi và vector đoạn văn trong không gian nhiều chiều. Đây là **tín hiệu chẩn đoán nội bộ** của thuật toán tìm kiếm, không phải thước đo trực tiếp độ đúng của câu trả lời pháp lý và không thể so sánh giữa các mô hình khác nhau.
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
| **`3`** | **Rất liên quan (Core / Direct Answer)** | Đoạn trích chứa thông tin trả lời trực tiếp, đầy đủ cho câu hỏi. Đây là căn cứ pháp lý quan trọng nhất mà chatbot bắt buộc phải tìm thấy để trả lời chính xác. | Câu hỏi: *"Hồ sơ đề nghị hỗ trợ tiền đóng học phí gồm những gì?"* → Chunk `116-2020-ND-CP-dieu-7-khoan-1` quy định Đơn đề nghị theo Mẫu số 01 đạt **grade 3**. |
| **`2`** | **Liên quan (Supporting Context)** | Đoạn trích cung cấp thông tin bổ trợ có giá trị cao, ví dụ: thẩm quyền phê duyệt, thời hạn nộp hồ sơ, công thức tính toán hoặc trường hợp miễn trừ. | Cùng câu hỏi trên → Chunk `116-2020-ND-CP-dieu-7-khoan-2` quy định về thời hạn nộp hồ sơ và hình thức nộp đạt **grade 2**. |
| **`1`** | **Liên quan yếu (Weak / Contextual)** | Đoạn trích mang tính chất ngữ cảnh rộng hoặc định nghĩa khái niệm nền tảng trong cùng văn bản, có liên quan gián tiếp nhưng không trực tiếp giải quyết vấn đề. | Chunk `116-2020-ND-CP-dieu-1` về phạm vi điều chỉnh đối với câu hỏi về hồ sơ đạt **grade 1**. |
| **`0`** | **Không liên quan (Irrelevant)** | Đoạn trích về chủ đề khác, điều khoản không liên quan, hoặc các chunk dạng Hỏi–Đáp (QA) không mang giá trị văn bản quy phạm chính thức. | Các điều khoản về trách nhiệm của UBND tỉnh đối với câu hỏi về thủ tục sinh viên nộp hồ sơ đạt **grade 0**. |

### 3.3. Quy tắc đối sánh và Lọc nguồn pháp lý

1. **Lọc nguồn pháp lý chính thức (`_is_legal_source`)**: Chỉ các chunk có thuộc tính `content_type == 'legal_text'` mới được tham gia đối chiếu ground truth cấu trúc. Toàn bộ các chunk câu hỏi - đáp tham khảo (`hoi-dap-nghi-dinh-116...`) đều bị gán `grade = 0` trong đánh giá pháp lý phân cấp.
2. **Quy tắc nhị phân hóa (`min_grade = 2`)**:
   * Khi tính toán các chỉ số nhị phân (`Hit@K`, `Recall@K`, `Precision@K`, `MRR`), hệ thống áp dụng ngưỡng:
     $$\text{Relevant} \iff \text{grade} \ge 2$$
   * Các chunk có `grade = 1` được coi là thông tin phụ trợ, không được tính vào số lượng chunk thỏa mãn để đảm bảo đánh giá khắt khe chất lượng truy xuất pháp lý.
3. **Quy tắc tính toán đa mức (`nDCG@K`)**:
   * Toàn bộ các mức điểm **`1`, `2`, `3`** được đưa vào công thức nDCG có trọng số, giúp ghi nhận công bằng việc mô hình tìm thấy cả thông tin trực tiếp lẫn thông tin ngữ cảnh hỗ trợ.

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
* **Ý nghĩa thực tế**: Tất cả 6 mô hình được cấu hình chuẩn hóa ở **1024 chiều** (trong $\mathbb{R}^{1024}$). Cần phân biệt rõ:
  * **Số chiều vector (1024)**: Độ chi tiết của biểu diễn ngữ nghĩa (1 vector chiếm $\approx 4\text{ KB}$ lưu trữ).
  * **Số lượng chunks (618)**: Tổng số bản ghi văn bản pháp lý lưu trong bảng `legal_chunks`.

### 5.4. Độ phủ nhúng (Coverage)
* **Độ phủ: `618 / 618` (100.0%)**:
  Toàn bộ 618 đoạn văn bản pháp luật hiện có trong cơ sở dữ liệu đều đã được sinh vector embedding đầy đủ và không có bất kỳ chunk nào mang giá trị NULL ở cột vector của 6 mô hình.

### 5.5. Thời gian chạy benchmark (Runtime)
* **Ý nghĩa con số thời gian**: Thời gian đo lường trong log (từ ~87s đến ~195s) là tổng thời gian thực hiện toàn bộ quy trình: nạp mô hình vào GPU, mã hóa 100 câu hỏi liên tiếp, gửi truy vấn qua mạng tới NeonDB, thực hiện HNSW search và tính toán metrics đối sánh.
* **Cảnh báo thống kê**: Con số thời gian ghi nhận từ một lần chạy duy nhất, phụ thuộc vào độ trễ mạng internet tới máy chủ NeonDB và tải phần cứng GPU tại thời điểm đó, không nên được coi là chuẩn đo lường tốc độ tuyệt đối giữa các kiến trúc.

---

## 6. Cấu hình Kỹ thuật 6 Mô hình Embedding

Toàn bộ 6 mô hình đều sinh vector **1024 chiều**, sử dụng độ đo **cosine similarity** và được lưu trữ cách ly ở các cột riêng trong bảng `legal_chunks`:

| Model Identifier | Model Alias | Chiều vector | Backend | Cột Embedding | Retriever | Database | Top-K | ef_search |
|---|---|---:|---|---|---|---|---:|---:|
| `BAAI/bge-m3` | `bge-m3` | 1024 | `bge` | `embedding` | `HNSW (pgvector)` | `neondb` | 10 | 80 |
| `darklethelong/vnlegal-lal` | `vnlegal-lal` | 1024 | `sentence_transformer` | `embedding_vnlegal_lal` | `HNSW (pgvector)` | `neondb` | 10 | 80 |
| `mainguyen9/vietlegal-harrier-0.6b` | `vietlegal-harrier` | 1024 | `sentence_transformer` | `embedding_vietlegal_harrier` | `HNSW (pgvector)` | `neondb` | 10 | 80 |
| `mainguyen9/vietlegal-e5` | `vietlegal-e5` | 1024 | `sentence_transformer` | `embedding_vietlegal_e5` | `HNSW (pgvector)` | `neondb` | 10 | 80 |
| `jinaai/jina-embeddings-v3-hf` | `jina-v3` | 1024 | `jina` | `embedding_jina_v3` | `HNSW (pgvector)` | `neondb` | 10 | 80 |

---

## 7. Kết quả Thực nghiệm Tổng thể (Global Results)

Nguồn sự thật khách quan: Trích xuất trực tiếp từ 5 tệp log chính thức trong thư mục `logs/` chạy trên tập 95 câu hỏi hợp lệ của `EvalQueryV2`:

| Chỉ số | `bge-m3` | `vnlegal-lal` | `vietlegal-harrier` | `vietlegal-e5` | `jina-v3` | `bm25` |
|---|---:|---:|---:|---:|---:|---:|
| **Hit@3** | 0.6947 | 0.5684 | 0.7895 | 0.6421 | 0.6000 | 0.4842 |
| **Hit@5** | 0.7895 | 0.6316 | 0.8421 | 0.7263 | 0.6737 | 0.5684 |
| **Hit@10** | 0.8842 | 0.7158 | 0.9053 | 0.8316 | 0.7684 | 0.7368 |
| **Recall@3** | 0.4265 | 0.3309 | 0.4707 | 0.3377 | 0.3665 | 0.3024 |
| **Recall@5** | 0.5213 | 0.3646 | 0.5427 | 0.4328 | 0.4539 | 0.3668 |
| **Recall@10** | 0.6282 | 0.4647 | 0.6591 | 0.5502 | 0.5527 | 0.4696 |
| **Precision@3** | 0.3088 | 0.2281 | 0.3474 | 0.2702 | 0.2561 | 0.1895 |
| **Precision@5** | 0.2421 | 0.1621 | 0.2526 | 0.2126 | 0.1958 | 0.1432 |
| **Precision@10** | 0.1558 | 0.1116 | 0.1663 | 0.1379 | 0.1379 | 0.1011 |
| **MRR** | 0.6229 | 0.4759 | 0.6528 | 0.5432 | 0.4983 | 0.4101 |
| **nDCG@3** | 0.4728 | 0.3476 | 0.5051 | 0.3843 | 0.3787 | 0.3112 |
| **nDCG@5** | 0.5045 | 0.3512 | 0.5203 | 0.4150 | 0.4057 | 0.3390 |
| **nDCG@10** | 0.5457 | 0.3903 | 0.5653 | 0.4549 | 0.4484 | 0.3798 |
| **Average Top-1 Similarity** | 0.698086 | 0.933826 | 0.601330 | 0.678649 | 0.732983 | 32.577733 |
| **Thời gian thực thi benchmark** | 135.58s | 152.00s | 136.10s | 143.65s | 128.16s | 1.83s | 152.00s | 136.10s | 143.65s | 128.16s | 152.00s | 136.10s | 143.65s | 128.16s | 87.52s | 94.67s | 92.51s | 111.35s |

---

## 8. Phân tích Kết quả theo Danh mục (Per-Category Results)

95 câu hỏi hợp lệ được phân bổ vào 6 danh mục phản ánh các loại yêu cầu tra cứu khác nhau:

### 8.1. Bảng số liệu chi tiết theo từng danh mục

#### 1. Category: `exact` (n = 15 câu - Trích dẫn trực tiếp Điều/Khoản)
| Model | Hit@3 | Hit@5 | Hit@10 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bge-m3` | 0.8000 | 0.8000 | 0.9333 | **0.7333** | 0.7667 | 0.8667 | 0.7503 | **0.7781** |
| `vnlegal-lal` | 0.6667 | 0.6667 | 0.8000 | 0.5556 | 0.5889 | 0.7444 | 0.5622 | 0.5547 |
| `vietlegal-harrier` | **0.8667** | **0.9333** | **1.0000** | 0.7111 | **0.8333** | **0.9333** | **0.7789** | 0.7731 |
| `vietlegal-e5` | 0.6000 | 0.6667 | 0.8000 | 0.4222 | 0.5667 | 0.7000 | 0.4790 | 0.5190 |
| `jina-v3` | 0.8000 | 0.8667 | 0.8667 | **0.7333** | 0.8000 | 0.8333 | 0.7689 | 0.7696 |
| `bm25` | 0.4667 | 0.5333 | 0.5333 | 0.3556 | 0.4222 | 0.4222 | 0.3133 | 0.3287 |

#### 2. Category: `semantic` (n = 20 câu - Ngôn ngữ tự nhiên / Diễn giải tương đương)
| Model | Hit@3 | Hit@5 | Hit@10 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bge-m3` | **0.8500** | **0.8500** | **0.8500** | **0.6208** | **0.7000** | 0.7125 | 0.6750 | 0.6465 |
| `vnlegal-lal` | 0.6500 | 0.6500 | 0.7000 | 0.5000 | 0.5375 | 0.6000 | 0.4889 | 0.4998 |
| `vietlegal-harrier` | 0.8000 | 0.8000 | **0.8500** | 0.6167 | 0.6792 | **0.7667** | **0.6905** | **0.6666** |
| `vietlegal-e5` | 0.6500 | 0.6500 | 0.8000 | 0.5125 | 0.5625 | 0.7083 | 0.5933 | 0.5843 |
| `jina-v3` | 0.5500 | 0.6500 | 0.6500 | 0.4500 | 0.5875 | 0.6000 | 0.4417 | 0.4666 |
| `bm25` | 0.6500 | 0.6500 | 0.8000 | 0.5417 | 0.5417 | 0.6667 | 0.5501 | 0.5666 |

#### 3. Category: `contextual` (n = 25 câu - Ngữ cảnh chính sách / Điều kiện thi hành)
| Model | Hit@3 | Hit@5 | Hit@10 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bge-m3` | 0.4800 | 0.6800 | **0.8800** | 0.3200 | 0.4433 | 0.6333 | 0.4607 | 0.4672 |
| `vnlegal-lal` | 0.4400 | 0.4800 | 0.6000 | 0.2667 | 0.2800 | 0.4000 | 0.3283 | 0.3048 |
| `vietlegal-harrier` | **0.6000** | **0.7200** | **0.8800** | **0.4433** | **0.5033** | **0.6567** | **0.4863** | **0.4696** |
| `vietlegal-e5` | 0.5200 | 0.6800 | 0.8000 | 0.3100 | 0.4300 | 0.5867 | 0.4361 | 0.4234 |
| `jina-v3` | 0.4400 | 0.4800 | 0.7200 | 0.2867 | 0.3667 | 0.5233 | 0.3407 | 0.3480 |
| `bm25` | 0.4800 | 0.6400 | 0.8400 | 0.3133 | 0.4467 | 0.6000 | 0.4564 | 0.4399 |

#### 4. Category: `multi_chunk` (n = 15 câu - Yêu cầu nhiều đoạn trong cùng văn bản)
| Model | Hit@3 | Hit@5 | Hit@10 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bge-m3` | 0.6667 | 0.9333 | 0.9333 | 0.1940 | 0.2917 | 0.4346 | 0.6856 | 0.4578 |
| `vnlegal-lal` | 0.6000 | 0.8000 | 0.8667 | 0.1451 | 0.2006 | 0.2790 | 0.5167 | 0.3028 |
| `vietlegal-harrier` | **0.9333** | **1.0000** | **1.0000** | **0.2346** | **0.3302** | **0.4441** | **0.7911** | **0.5023** |
| `vietlegal-e5` | 0.6000 | 0.8000 | **1.0000** | 0.1273 | 0.2006 | 0.3857 | 0.4973 | 0.3497 |
| `jina-v3` | 0.6667 | 0.7333 | 0.8667 | 0.2213 | 0.3146 | 0.4124 | 0.5003 | 0.4088 |

#### 5. Category: `multi_document` (n = 10 câu - Liên kết xuyên văn bản)
| Model | Hit@3 | Hit@5 | Hit@10 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bge-m3` | 0.7000 | 0.8000 | **0.9000** | 0.2678 | 0.3317 | **0.4880** | 0.6893 | **0.4174** |
| `vnlegal-lal` | 0.7000 | 0.7000 | 0.8000 | 0.2339 | 0.2589 | 0.2958 | 0.5944 | 0.3315 |
| `vietlegal-harrier` | 0.7000 | 0.8000 | 0.8000 | **0.2798** | **0.3430** | 0.4153 | 0.5533 | 0.4070 |
| `vietlegal-e5` | **0.9000** | **0.9000** | **0.9000** | 0.2271 | 0.2882 | 0.3516 | 0.6833 | 0.3433 |
| `jina-v3` | 0.8000 | **0.9000** | **0.9000** | 0.2589 | 0.2930 | 0.3554 | **0.7200** | 0.3971 |
| `bm25` | 0.3000 | 0.4000 | 0.5000 | 0.1250 | 0.1861 | 0.2872 | 0.3361 | 0.2534 |

#### 6. Category: `complex_qa` (n = 10 câu - Câu hỏi tình huống tổng hợp)
| Model | Hit@3 | Hit@5 | Hit@10 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bge-m3` | 0.7000 | 0.7000 | 0.8000 | 0.2662 | 0.3912 | 0.4246 | 0.4976 | 0.3867 |
| `vnlegal-lal` | 0.4000 | 0.6000 | 0.7000 | 0.1625 | 0.2208 | 0.3554 | 0.3593 | 0.3006 |
| `vietlegal-harrier` | **1.0000** | **1.0000** | **1.0000** | **0.3829** | **0.4612** | **0.5075** | **0.7167** | **0.5322** |
| `vietlegal-e5` | 0.6000 | 0.8000 | 0.8000 | 0.2396 | 0.3246 | 0.3446 | 0.4617 | 0.3619 |
| `jina-v3` | 0.5000 | 0.6000 | 0.7000 | 0.2013 | 0.2846 | 0.3808 | 0.2917 | 0.2772 |
| `bm25` | 0.5000 | 0.5000 | 0.8000 | 0.1562 | 0.1562 | 0.2546 | 0.3042 | 0.2348 |

### 8.2. Nhận xét thực nghiệm theo danh mục

1. **Nhóm truy vấn chính xác (`exact`)**: Đạt tỷ lệ thành công cao nhất toàn bộ đợt đánh giá (`vietlegal-harrier` đạt Hit@10 tuyệt đối 100% và Recall@10 = 0.8500, MRR = 0.8122; `bge-m3` đạt Hit@10 = 93.3% và Recall@10 = 0.8333). Điều này chứng minh cấu trúc metadata-aware (`[Document]`, `[Article]`, `[Clause]`, `[Point]`) phát huy tác dụng tối ưu khi người dùng nêu rõ số hiệu văn bản pháp lý.
2. **Nhóm truy vấn ngữ nghĩa tự nhiên (`semantic`)**: `BAAI/bge-m3` đạt kết quả vượt trội ở Hit@10 (**0.8500**) và Recall@10 (**0.6333**), cho thấy khả năng ánh xạ từ vựng tự nhiên sang thuật ngữ hành chính của BGE-M3 rất mạnh mẽ nhờ tập ngữ liệu tiền huấn luyện đa ngữ quy mô lớn.
3. **Nhóm đa văn bản và đa chunk (`multi_chunk`, `multi_document`)**: `vietlegal-harrier` đạt Hit@10 là 1.0000 ở cả hai nhóm, chứng minh khả năng tìm thấy ít nhất 1 căn cứ pháp lý là tuyệt đối. Tuy nhiên, chỉ số Recall@10 chỉ đạt quanh mức 0.44 - 0.51, phản ánh rằng việc gom đủ *toàn bộ* các đoạn pháp lý liên quan rải rác ở nhiều điều khoản khác nhau vào Top-10 là thách thức lớn đối với retrieval một giai đoạn.

---

## 9. Phân tích Kết quả theo Độ khó (Per-Difficulty Results)

95 câu hỏi hợp lệ được phân loại theo 3 mức độ phức tạp:

### 9.1. Bảng số liệu chi tiết theo độ khó

#### 1. Mức độ: `easy` (n = 10 câu)
| Model | Hit@3 | Hit@5 | Hit@10 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bge-m3` | 0.8000 | 0.8000 | 0.9000 | **0.7500** | **0.8000** | 0.8500 | 0.7611 | 0.7800 |
| `vnlegal-lal` | 0.7000 | 0.7000 | 0.8000 | 0.6000 | 0.6500 | 0.8000 | 0.6000 | 0.6129 |
| `vietlegal-harrier` | **0.9000** | **0.9000** | **1.0000** | **0.7500** | **0.8000** | **0.9000** | **0.8100** | **0.7957** |
| `vietlegal-e5` | 0.6000 | 0.7000 | 0.7000 | 0.4500 | 0.6000 | 0.6000 | 0.5083 | 0.5093 |
| `jina-v3` | 0.7000 | 0.8000 | 0.8000 | 0.6500 | 0.7500 | 0.8000 | 0.7200 | 0.7343 |
| `bm25` | 0.5000 | 0.6000 | 0.6000 | 0.4000 | 0.5000 | 0.5000 | 0.3200 | 0.3642 |

#### 2. Mức độ: `medium` (n = 41 câu)
| Model | Hit@3 | Hit@5 | Hit@10 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bge-m3` | 0.7073 | 0.7805 | 0.8780 | 0.4442 | 0.5300 | 0.6280 | 0.6103 | 0.5421 |
| `vnlegal-lal` | 0.6098 | 0.6585 | 0.7317 | 0.3774 | 0.4107 | 0.4813 | 0.4770 | 0.4049 |
| `vietlegal-harrier` | **0.7561** | **0.8293** | **0.9024** | **0.4680** | **0.5508** | **0.6915** | **0.6535** | **0.5793** |
| `vietlegal-e5` | 0.6098 | 0.6829 | 0.8780 | 0.3434 | 0.4485 | 0.6506 | 0.5278 | 0.4863 |
| `jina-v3` | 0.5610 | 0.6341 | 0.7317 | 0.3719 | 0.4677 | 0.5726 | 0.4571 | 0.4422 |
| `bm25` | 0.5122 | 0.6098 | 0.7317 | 0.3452 | 0.4143 | 0.5031 | 0.4454 | 0.4110 |

#### 3. Mức độ: `hard` (n = 44 câu)
| Model | Hit@3 | Hit@5 | Hit@10 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bge-m3` | 0.6591 | 0.7955 | **0.8864** | 0.3365 | 0.4499 | **0.5779** | 0.6031 | 0.4959 |
| `vnlegal-lal` | 0.5000 | 0.5909 | 0.6818 | 0.2265 | 0.2568 | 0.3730 | 0.4466 | 0.3261 |
| `vietlegal-harrier` | **0.7955** | **0.8409** | **0.8864** | **0.4097** | **0.4767** | 0.5741 | **0.6165** | **0.5000** |
| `vietlegal-e5` | 0.6818 | 0.7727 | 0.8182 | 0.3068 | 0.3802 | 0.4453 | 0.5656 | 0.4134 |
| `jina-v3` | 0.6136 | 0.6818 | 0.7955 | 0.2970 | 0.3737 | 0.4779 | 0.4863 | 0.3892 |
| `bm25` | 0.4545 | 0.5227 | 0.7727 | 0.2404 | 0.2922 | 0.4315 | 0.3977 | 0.3543 |

### 9.2. Nhận xét thực nghiệm theo độ khó

* **Quy luật suy giảm hiệu năng theo độ phức tạp**: Ở tất cả các mô hình, các chỉ số đều giảm rõ rệt theo thứ tự `easy` $\rightarrow$ `medium` $\rightarrow$ `hard`. Ví dụ với `BAAI/bge-m3`, Recall@10 giảm từ **0.8500** (easy) xuống **0.5451** (medium) và **0.4709** (hard).
* **Năng lực vượt trội ở câu hỏi khó**: `vietlegal-harrier` thể hiện khả năng thích ứng xuất sắc ở nhóm câu hỏi khó (`hard`, n=44), đạt Hit@10 là **0.8636**, Recall@10 là **0.5152** và MRR là **0.5613**, vượt trên BGE-M3 (Hit@10 = 0.7727, MRR = 0.5223).

---

## 10. Phân tích Truy vấn Ngoài phạm vi (OOS) và Không hợp lệ (Invalid)

5 câu hỏi kiểm thử đặc biệt trong `EvalQueryV2` gồm:
1. `[invalid/easy]` *asdfghjkl qwerty 12345 nghi dinh su pham ???*
2. `[invalid/easy]` *Giá vàng hôm nay là bao nhiêu? Tỷ giá đô la Mỹ?*
3. `[out_of_scope/easy]` *Luật Lao động 2019 quy định về sa thải người lao động trái pháp luật như thế nào*
4. `[out_of_scope/medium]` *Nghị định 71/2020/NĐ-CP quy định lộ trình nâng chuẩn trình độ giáo viên mầm non*
5. `[out_of_scope/easy]` *Theo Bộ luật Hình sự, hành vi gian lận trong thi cử đại học bị xử lý như thế nào*

### Kết quả đo lường từ log benchmark

* **Số lượng kết quả trả về**: Cả 5 mô hình đều trả về đủ 10 chunks cho mỗi truy vấn (tổng cộng 50 kết quả cho 5 câu).
* **Số lỗi runtime / ngoại lệ**: 0 lỗi trên toàn bộ các mô hình.
* **Top-1 Chunk trả về**: Hệ thống luôn trả về các đoạn trích có vector gần nhất trong 618 chunks hiện có (ví dụ với BGE-M3, câu hỏi giá vàng trả về `60-2025-ND-CP-dieu-3-khoan-2`, câu hỏi luật lao động trả về `LUAT-GIAO-DUC-2019-dieu-65-khoan-3`).
* **Hàm ý kiến trúc quan trọng (Abstention Requirement)**:
  * Thuật toán tìm kiếm vector láng giềng gần nhất (k-NN) hoạt động thuần túy trên phép chiếu khoảng cách hình học, **không có khả năng tự nhận biết câu hỏi có thuộc phạm vi hay không**.
  * Điều này khẳng định sự cần thiết phải xây dựng cơ chế phát hiện từ chối trả lời (Abstention / Guardrails) ở tầng ứng dụng (Application Layer), ví dụ: bộ phân loại intent trước khi tìm kiếm, hoặc thiết lập ngưỡng tương đồng tối thiểu kết hợp prompt hướng dẫn LLM từ chối khi ngữ cảnh không khớp.

---

## 11. Diễn giải Kết quả, Cảnh báo Điểm tương đồng và Giới hạn

### 11.1. So sánh đa chiều giữa 5 mô hình

1. **`mainguyen9/vietlegal-harrier-0.6b` (Mô hình tối ưu cho xếp hạng đỉnh)**:
   * Đạt hiệu năng tổng thể cao nhất: Hit@10 = **0.8632**, Recall@10 = **0.5539**, MRR = **0.6027**, nDCG@10 = **0.4954**.
   * Rất mạnh trong việc đưa đúng điều khoản sửa đổi của Nghị định 60/2025 lên ngay vị trí Rank 1.
2. **`BAAI/bge-m3` (Mô hình bao quát ngữ nghĩa tự nhiên tốt nhất)**:
   * Bám sát ở vị trí hàng đầu: Hit@10 = **0.8316**, Recall@10 = **0.5428**, MRR = **0.5733**, nDCG@10 = **0.4823**.
   * Dẫn đầu tuyệt đối ở nhóm câu hỏi diễn đạt tự nhiên (`semantic`, Hit@10 = 0.8500) và câu hỏi ngữ cảnh (`contextual`, Hit@10 = 0.8400).
   * Tốc độ suy luận nhanh nhất nhóm (87.72s).
3. **`mainguyen9/vietlegal-e5` (Mô hình ứng viên tiềm năng cho Reranker)**:
   * Đạt độ phủ Top-10 tốt (Hit@10 = **0.7789**, Recall@10 = **0.4594**).
   * Tuy nhiên, MRR chỉ đạt **0.4064** do hiện tượng "Ranking Latency" (chunk đúng thường bị đẩy xuống khoảng Rank 5 - 10). Mô hình này rất phù hợp làm bộ lọc ứng viên trước khi chuyển qua tầng Reranker.
4. **`jinaai/jina-embeddings-v3-hf`**:
   * Hiệu năng ở mức trung bình đồng đều (Hit@10 = **0.6526**, Recall@10 = **0.4052**, MRR = **0.4015**).
5. **`darklethelong/vnlegal-lal`**:
   * Hit@10 đạt **0.6632**, Recall@10 đạt **0.3588**, MRR đạt **0.3749**. Điểm tương đồng cao bất thường phản ánh không gian vector co cụm.

### 11.2. Cảnh báo nghiêm ngặt về Điểm tương đồng (Similarity Score)

> [!WARNING]
> **Điểm tương đồng cosin cao KHÔNG phản ánh chất lượng truy xuất vượt trội.**

* **Minh chứng dữ liệu thực nghiệm**:
  * Mô hình `darklethelong/vnlegal-lal` ghi nhận điểm tương đồng trung bình Top-1 cao nhất nhóm (**0.930506**). Tuy nhiên, các chỉ số thực tế lại ở mức trung bình thấp: Recall@10 chỉ đạt 0.3588 và MRR chỉ đạt 0.3749.
  * Ngược lại, mô hình `mainguyen9/vietlegal-harrier-0.6b` có điểm tương đồng Top-1 trung bình thấp nhất nhóm (**0.583023**), nhưng lại đứng đầu toàn bảng về Recall@10 (0.5539), Hit@10 (0.8632) và MRR (0.6027).
* **Kết luận**: Tuyệt đối không sử dụng điểm tương đồng cosin để so sánh chất lượng giữa các mô hình embedding. Việc đánh giá bắt buộc phải căn cứ vào các chỉ số IR chuẩn (`Recall@K`, `Hit@K`, `MRR`, `nDCG@K`).

### 11.3. Giới hạn của phép đánh giá

* **Đánh giá một giai đoạn (Single-stage Dense Retrieval)**: Phép đo này chưa áp dụng tầng Reranker (Cross-Encoder) hay tìm kiếm kết hợp Hybrid (Vector + Trigram BM25).
* **Quy mô tập kiểm thử**: Tập dữ liệu gồm 100 câu hỏi được thiết kế chuẩn xác nhưng chưa bao quát toàn bộ mọi biến thể câu hỏi trong thực tế đời sống.
* **Thời gian thực thi**: Thời gian chạy được đo trên môi trường máy trạm cụ thể kết nối dịch vụ cơ sở dữ liệu đám mây NeonDB, mang tính tham khảo kỹ thuật.

---

## 12. Hướng dẫn Thực thi Tái lập (Reproducibility Instructions)

Toàn bộ kết quả benchmark có thể được tái lập 100% bằng Windows PowerShell theo quy trình chuẩn sau:

> [!NOTE]
> **Phân biệt Indexing và Evaluation:**
> * **Indexing (`scripts/index_embeddings.py`)**: Sinh vector và ghi vào database. Bước này chỉ cần chạy **một lần duy nhất** cho mỗi mô hình nếu cột vector chưa tồn tại.
> * **Evaluation (`scripts/test_retrieval.py`)**: Đọc dữ liệu vector đã có từ database và chạy benchmark đối sánh với `EvalQueryV2`. Lệnh này **không** tự động sinh lại vector.

### 12.1. Lệnh tái lập Benchmark cho từng mô hình

```powershell
# Bước 1: Kích hoạt môi trường Conda
conda activate chatbot

# --- 1. Đánh giá BAAI/bge-m3 (cột 'embedding') ---
$env:EMBEDDING_MODEL = "bge-m3"
python scripts/test_retrieval.py --top-k 10 --ef-search 80

# --- 2. Đánh giá darklethelong/vnlegal-lal ---
# (Tiền đề: python scripts/index_embeddings.py --model vnlegal-lal)
$env:EMBEDDING_MODEL = "vnlegal-lal"
python scripts/test_retrieval.py --top-k 10 --ef-search 80

# --- 3. Đánh giá mainguyen9/vietlegal-harrier-0.6b ---
# (Tiền đề: python scripts/index_embeddings.py --model vietlegal-harrier)
$env:EMBEDDING_MODEL = "vietlegal-harrier"
python scripts/test_retrieval.py --top-k 10 --ef-search 80

# --- 4. Đánh giá mainguyen9/vietlegal-e5 ---
# (Tiền đề: python scripts/index_embeddings.py --model vietlegal-e5)
$env:EMBEDDING_MODEL = "vietlegal-e5"
python scripts/test_retrieval.py --top-k 10 --ef-search 80

# --- 5. Đánh giá jinaai/jina-embeddings-v3-hf ---
# (Tiền đề: python scripts/index_embeddings.py --model jina-v3)
$env:EMBEDDING_MODEL = "jina-v3"
python scripts/test_retrieval.py --top-k 10 --ef-search 80
```

Báo cáo kết quả chi tiết từng câu hỏi và tổng hợp sẽ được ghi tự động vào thư mục `logs/` theo định dạng `<model_prefix>_<timestamp>_<random_suffix>.txt`.

### 12.2. Chế độ truy vấn đơn lẻ (Ad-hoc Single Query Mode)

Khi cần kiểm tra nhanh kết quả trả về của một câu hỏi riêng biệt mà không tính toán metrics hay tạo file log:

```powershell
conda activate chatbot
python scripts/test_retrieval.py --query "Điều kiện để được miễn bồi hoàn kinh phí hỗ trợ là gì?" --top-k 5 --ef-search 80
```

---

## 13. Tài liệu liên quan

* [Kiến trúc tầng Retrieval](retrieval.md)
* [Lịch sử phát triển tầng Retrieval](retrieval-development-history.md)
* [Hướng dẫn chuyển đổi mô hình embedding](embedding-model-switching.md)
* [Quay lại README](../README.md)
