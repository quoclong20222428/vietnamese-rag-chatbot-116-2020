"""Comprehensive retrieval evaluation dataset -- graded chunk-level ground truth.

Corpus: 618 chunks from 4 sources
  - Nghi dinh 116/2020/ND-CP          (72 chunks, legal_text)
  - Nghi dinh 60/2025/ND-CP           (54 chunks, legal_text)
  - Luat Giao duc 2019                 (461 chunks, legal_text)
  - Hoi dap Nghi dinh 116/2020/ND-CP  (31 chunks, qa)

Ground-truth relevance scale
  3 = Highly relevant -- directly provides essential evidence
  2 = Relevant -- provides important supporting evidence
  1 = Related -- useful context but insufficient to answer alone
  0 = Not relevant (omitted from relevant_chunks)

Category distribution (100 queries)
  exact           15  (~15%)
  semantic        20  (~20%)
  contextual      25  (~25%)
  multi_chunk     15  (~15%)
  multi_document  10  (~10%)
  complex_qa      10  (~10%)
  out_of_scope     3  ( ~3%)
  invalid          2  ( ~2%)

nDCG@K
  Unitless in [0,1]. DCG@K = sum_i (2^rel_i - 1) / log2(i+1).
  nDCG@K = DCG@K / IDCG@K.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Document title shorthands (exact strings stored in the vector database)
# ---------------------------------------------------------------------------
_ND116 = "Nghị định 116/2020/NĐ-CP"
_ND60  = "Nghị định 60/2025/NĐ-CP"
_LGD   = "Luật Giáo dục 2019"
_QA116 = "Hỏi đáp Nghị định 116/2020/NĐ-CP"

CATEGORIES = frozenset([
    "exact", "semantic", "contextual", "multi_chunk",
    "multi_document", "complex_qa", "out_of_scope", "invalid",
])
DIFFICULTIES = frozenset(["easy", "medium", "hard"])

GRADE_HIGHLY_RELEVANT = 3
GRADE_RELEVANT        = 2
GRADE_RELATED         = 1
GRADE_NOT_RELEVANT    = 0


@dataclass
class EvalQueryV2:
    """One evaluation query with graded chunk-level ground truth.

    Attributes
    ----------
    query               : Vietnamese natural-language query string.
    description         : Brief English description of the information need.
    category            : exact | semantic | contextual | multi_chunk |
                          multi_document | complex_qa | out_of_scope | invalid.
    difficulty          : easy | medium | hard.
    is_answerable       : False for out_of_scope and invalid queries.
    relevant_chunks     : chunk_id -> relevance grade (1/2/3).
    relevant_documents  : document titles containing relevant supporting evidence.
                           Includes _QA116 when one or more QA-reference chunks
                           are present in relevant_chunks.
    requires_multi_chunk: True when >=2 chunks are needed to fully answer.
    requires_multi_document: True when evidence spans >=2 official legal
                              documents; QA-reference chunks alone do not make
                              this flag True.
    requires_verification: Legacy compatibility flag. Always False here.
    notes               : Hard negatives, boundary conditions, etc.
    """
    query:                   str
    description:             str
    category:                str
    difficulty:              str
    is_answerable:           bool
    relevant_chunks:         dict[str, int] = field(default_factory=dict)
    relevant_documents:      list[str]      = field(default_factory=list)
    requires_multi_chunk:    bool           = False
    requires_multi_document: bool           = False
    requires_verification:   bool           = False
    notes:                   str            = ""

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"Invalid category: {self.category!r}")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"Invalid difficulty: {self.difficulty!r}")
        for cid, grade in self.relevant_chunks.items():
            if grade not in (1, 2, 3):
                raise ValueError(f"Invalid grade {grade} for chunk {cid!r}")
        if not self.is_answerable:
            if self.relevant_chunks:
                raise ValueError("Unanswerable query has relevant_chunks")
            if self.relevant_documents:
                raise ValueError("Unanswerable query has relevant_documents")


# ===========================================================================
# EVALUATION DATASET  (100 queries)
# ===========================================================================
EVAL_QUERIES_V2: list[EvalQueryV2] = [

    # -----------------------------------------------------------------------
    # EXACT  (E-01 .. E-15)
    # -----------------------------------------------------------------------
    EvalQueryV2(  # E-01 | easy
        query="Khoản 1 điều 2 Nghị định 116/2020/NĐ-CP quy định thế nào về cách xác định số tháng làm tròn khi tính thời gian làm việc?",
        description="Exact: ND116 Dieu 2 Khoan 1 -- rounding rule for working months",
        category="exact", difficulty="easy", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-2-khoan-1": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-018": 3,
        },
        relevant_documents=[_ND116, _QA116],
        notes="Hard negative: 116-2020-ND-CP-dieu-2-khoan-3 (authority to certify, same article).",
    ),
    EvalQueryV2(  # E-02 | easy
        query="Khoản 2 điều 2 Nghị định 116/2020/NĐ-CP xác định người được xem là công tác trong ngành giáo dục gồm những nhóm nào?",
        description="Exact: ND116 Dieu 2 Khoan 2 -- definition of education-sector employment",
        category="exact", difficulty="easy", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-2-khoan-2-diem-a": 3,
            "116-2020-ND-CP-dieu-2-khoan-2-diem-b": 3,
        },
        relevant_documents=[_ND116],
    ),
    EvalQueryV2(  # E-03 | easy
        query="điều 4 Nghị định 116/2020/NĐ-CP quy định mức hỗ trợ chi phí sinh hoạt cho sinh viên sư phạm là bao nhiêu tiền mỗi tháng?",
        description="Exact: ND116 Dieu 4 Khoan 1 -- monthly living allowance amount",
        category="exact", difficulty="easy", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-4-khoan-1": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-026": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-001": 2,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # E-04 | easy
        query="Khoản 4 điều 9 Nghị định 116/2020/NĐ-CP quy định gì đối với sinh viên thuộc đối tượng chính sách khó khăn khi bồi hoàn?",
        description="Exact: ND116 Dieu 9 Khoan 4 -- policy-target students and reimbursement hardship",
        category="exact", difficulty="easy", is_answerable=True,
        relevant_chunks={"116-2020-ND-CP-dieu-9-khoan-4": 3},
        relevant_documents=[_ND116],
    ),
    EvalQueryV2(  # E-05 | easy
        query="Khoản 6 điều 9 Nghị định 116/2020/NĐ-CP quy định hậu quả khi sinh viên hoặc gia đình không thực hiện nghĩa vụ bồi hoàn là gì?",
        description="Exact: ND116 Dieu 9 Khoan 6 -- consequences of non-compliance with reimbursement",
        category="exact", difficulty="easy", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-9-khoan-6": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-021": 2,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # E-06 | easy
        query="Điểm a Khoản 1 điều 6 Nghị định 116 áp dụng cho sinh viên sư phạm nào sau khi tốt nghiệp?",
        description="Exact: ND116 Dieu 6 Khoan 1 Diem a -- no education employment within 2 years",
        category="exact", difficulty="easy", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-1-diem-a": 3,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-a": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-031": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-009": 1,
        },
        relevant_documents=[_ND116, _QA116],
        notes="Diem a Khoan 2 is the exemption counterpart (grade 2).",
    ),
    EvalQueryV2(  # E-07 | easy
        query="Điểm b Khoản 1 điều 6 Nghị định 116 liên quan đến điều kiện thời gian công tác không đủ như thế nào?",
        description="Exact: ND116 Dieu 6 Khoan 1 Diem b -- insufficient service period",
        category="exact", difficulty="easy", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-1-diem-b": 3,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-a": 2,
        },
        relevant_documents=[_ND116],
    ),
    EvalQueryV2(  # E-08 | easy
        query="Điểm c Khoản 1 điều 6 Nghị định 116 quy định những trường hợp nào đang trong thời gian đào tạo phải bồi hoàn?",
        description="Exact: ND116 Dieu 6 Khoan 1 Diem c -- training interruption triggers reimbursement",
        category="exact", difficulty="easy", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-1-diem-c": 3,
            "116-2020-ND-CP-dieu-6-khoan-4": 1,
        },
        relevant_documents=[_ND116],
        notes="Khoan 4 (illness permitted break) is a direct contrast boundary.",
    ),
    EvalQueryV2(  # E-09 | easy
        query="Nghị định 60/2025/NĐ-CP sửa đổi Khoản 1 điều 1 Nghị định 116 xác định đối tượng hỗ trợ như thế nào?",
        description="Exact: ND60 Dieu 1 Khoan 1 -- amended scope of support recipients",
        category="exact", difficulty="easy", is_answerable=True,
        relevant_chunks={
            "60-2025-ND-CP-dieu-1-khoan-1": 3,
            "116-2020-ND-CP-dieu-1-khoan-1": 2,
        },
        relevant_documents=[_ND60, _ND116],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # E-10 | easy
        query="điều 66 Luật Giáo dục 2019 phân biệt giáo viên và giảng viên ở điểm nào?",
        description="Exact: LGD Dieu 66 -- teacher vs lecturer definition",
        category="exact", difficulty="easy", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-66-khoan-1": 3,
        },
        relevant_documents=[_LGD],
    ),
    EvalQueryV2(  # E-11 | medium
        query="điều 8 Nghị định 116 quy định công thức tính số tiền phải hoàn trả khi sinh viên chưa đủ thời gian công tác như thế nào?",
        description="Exact: ND116 Dieu 8 Khoan 3 -- partial reimbursement calculation formula",
        category="exact", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-8-khoan-3": 3,
            "116-2020-ND-CP-dieu-8-khoan-1": 2,
            "116-2020-ND-CP-dieu-8-khoan-2": 1,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-016": 2,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # E-12 | medium
        query="Khoản 3 điều 3 Nghị định 116/2020/NĐ-CP liệt kê các hình thức giao nhiệm vụ, đặt hàng hoặc đấu thầu đào tạo giáo viên gồm những hình thức nào?",
        description="Exact: ND116 Dieu 3 Khoan 3 -- training assignment modalities",
        category="exact", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-3-khoan-3-diem-a": 3,
            "116-2020-ND-CP-dieu-3-khoan-3-diem-b": 3,
            "116-2020-ND-CP-dieu-3-khoan-3-diem-c": 3,
            "116-2020-ND-CP-dieu-3-khoan-1": 1,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-010": 2,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # E-13 | medium
        query="Cơ sở đào tạo giáo viên phải công khai những thông tin gì theo quy định tại Khoản 4 điều 12 Nghị định 116?",
        description="Exact: ND116 Dieu 12 Khoan 4 -- institution disclosure obligations",
        category="exact", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-12-khoan-4": 3,
        },
        relevant_documents=[_ND116],
    ),
    EvalQueryV2(  # E-14 | medium
        query="điều 67 Luật Giáo dục 2019 quy định tiêu chuẩn của nhà giáo về phẩm chất đạo đức như thế nào?",
        description="Exact: LGD Dieu 67 Khoan 1 -- teacher ethics standard",
        category="exact", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-67-khoan-1": 3,
        },
        relevant_documents=[_LGD],
    ),
    EvalQueryV2(  # E-15 | medium
        query="Khoản 2 điều 85 Luật Giáo dục 2019 quy định điều gì về học phí của người học?",
        description="Exact: LGD Dieu 85 Khoan 2 -- student tuition fee provisions",
        category="exact", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-2": 3,
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-1": 1,
        },
        relevant_documents=[_LGD],
    ),

    # -----------------------------------------------------------------------
    # SEMANTIC  (S-01 .. S-20)
    # -----------------------------------------------------------------------
    EvalQueryV2(  # S-01 | medium
        query="Nhà nước có hỗ trợ tiền cho sinh viên học ngành sư phạm không?",
        description="Semantic: state support for pedagogy students",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-1-khoan-1": 3,
            "116-2020-ND-CP-dieu-4-khoan-1": 3,
            "116-2020-ND-CP-dieu-1-khoan-2-diem-a": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-026": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-007": 3,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # S-02 | medium
        query="Sau khi ra trường, sinh viên sư phạm cần làm gì để không phải trả lại tiền đã nhận?",
        description="Semantic: exemption from reimbursement -- post-graduation teaching commitment",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-2-diem-a": 3,
            "116-2020-ND-CP-dieu-6-khoan-1-diem-a": 2,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-b": 1,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-c": 1,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-029": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-019": 2,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # S-03 | medium
        query="Một người học sư phạm ra trường làm việc ở đâu thì được tính là công tác trong ngành giáo dục?",
        description="Semantic: which workplaces qualify as education-sector employment?",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-2-khoan-2-diem-a": 3,
            "116-2020-ND-CP-dieu-2-khoan-2-diem-b": 3,
            "116-2020-ND-CP-dieu-2-khoan-3": 1,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-009": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-031": 2,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # S-04 | medium
        query="Học sư phạm mà giữa chừng bỏ học thì phải trả lại bao nhiêu tiền?",
        description="Semantic: voluntary withdrawal -- full reimbursement",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-1-diem-c": 3,
            "116-2020-ND-CP-dieu-8-khoan-2": 3,
            "116-2020-ND-CP-dieu-8-khoan-1": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-027": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-028": 2,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # S-05 | medium
        query="Đang học bị đình chỉ tạm thời thì Khoản hỗ trợ có bị dừng trong thời gian đó không?",
        description="Semantic: temporary suspension and support cessation (ND116 Dieu 6 Khoan 3)",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-3": 3,
            "116-2020-ND-CP-dieu-6-khoan-4": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-012": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-011": 2,
        },
        relevant_documents=[_ND116, _QA116],
        notes="Khoan 4 (illness permitted break) is a hard negative at grade 2.",
    ),
    EvalQueryV2(  # S-06 | medium
        query="Nếu sinh viên sư phạm ốm đau phải tạm dừng việc học, sau đó học lại thì có được tiếp tục nhận tiền hỗ trợ không?",
        description="Semantic: illness interruption and continued support (ND116 Dieu 6 Khoan 4)",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-4": 3,
            "116-2020-ND-CP-dieu-6-khoan-3": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-014": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-012": 3,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # S-07 | medium
        query="Tiền hỗ trợ đã nhận gồm những Khoản nào khi tính số tiền phải hoàn trả?",
        description="Semantic: components of reimbursement cost (ND116 Dieu 8 Khoan 1)",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-8-khoan-1": 3,
            "116-2020-ND-CP-dieu-8-khoan-2": 2,
            "116-2020-ND-CP-dieu-8-khoan-3": 2,
        },
        relevant_documents=[_ND116],
    ),
    EvalQueryV2(  # S-08 | medium
        query="Sau khi tốt nghiệp người nhận hỗ trợ cần định kỳ cung cấp thông tin gì về thời gian công tác?",
        description="Semantic: post-graduation reporting obligation (ND116 Dieu 13 Khoan 2)",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-13-khoan-2": 3,
            "116-2020-ND-CP-dieu-13-khoan-1": 2,
        },
        relevant_documents=[_ND116],
    ),
    EvalQueryV2(  # S-09 | medium
        query="Tiền được thu hồi từ bồi hoàn học phí được đưa vào đâu?",
        description="Semantic: destination of recovered reimbursement funds (ND116 Dieu 9 Khoan 5)",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-9-khoan-5": 3,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-022": 2,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # S-10 | medium
        query="Địa phương phải gửi thông tin nhu cầu tuyển dụng giáo viên cho Bộ trước ngày nào?",
        description="Semantic: local demand submission deadline to MOET (ND116 Dieu 3 Khoan 1)",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-3-khoan-1": 3,
            "60-2025-ND-CP-dieu-1-khoan-3-sua-doi-bo-sung-dieu-3-khoan-1-diem-a": 2,
        },
        relevant_documents=[_ND116, _ND60],
        requires_multi_document=True,
        notes="ND60 Dieu 3 Khoan 1 Diem a amends the deadline.",
    ),
    EvalQueryV2(  # S-11 | medium
        query="Nhà giáo cần đáp ứng những yêu cầu gì về trình độ được đào tạo?",
        description="Semantic: standard training qualifications for teachers (LGD Dieu 72)",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-72-khoan-1-diem-a": 3,
            "LUAT-GIAO-DUC-2019-dieu-72-khoan-1-diem-b": 3,
            "LUAT-GIAO-DUC-2019-dieu-72-khoan-1-diem-c": 3,
            "LUAT-GIAO-DUC-2019-dieu-72-khoan-1-diem-d": 3,
            "LUAT-GIAO-DUC-2019-dieu-72-khoan-2": 1,
        },
        relevant_documents=[_LGD],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # S-12 | medium
        query="Người học theo chế độ cử tuyển có trách nhiệm gì sau khi tốt nghiệp và việc bố trí việc làm được thực hiện thế nào?",
        description="Semantic: cử tuyển post-graduation obligation and job placement (LGD Dieu 87)",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-87-khoan-2": 3,
            "LUAT-GIAO-DUC-2019-dieu-87-khoan-3": 3,
            "LUAT-GIAO-DUC-2019-dieu-87-khoan-4": 1,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-009": 2,
        },
        relevant_documents=[_LGD, _QA116],
    ),
    EvalQueryV2(  # S-13 | medium
        query="Người học thuộc hộ nghèo và hộ cận nghèo có thể được hưởng chính sách gì về học phí?",
        description="Semantic: tuition support for poor and near-poor households (LGD Dieu 85 Khoan 2)",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-2": 3,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-007": 2,
        },
        relevant_documents=[_LGD, _QA116],
    ),
    EvalQueryV2(  # S-14 | medium
        query="Theo Nghị định 60/2025, Bộ Giáo dục và Đào tạo phải thông báo chỉ tiêu tuyển sinh cho các cơ sở đào tạo giáo viên trước thời điểm nào?",
        description="Semantic: current MOET quota notification deadline (ND60 Dieu 3 Khoan 1 Diem b)",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "60-2025-ND-CP-dieu-1-khoan-3-sua-doi-bo-sung-dieu-3-khoan-1-diem-b": 3,
        },
        relevant_documents=[_ND60],
    ),
    EvalQueryV2(  # S-15 | medium
        query="Cơ sở đào tạo chi trả tiền sinh hoạt cho sinh viên sư phạm qua phương tiện gì?",
        description="Semantic: payment channel for living allowance (ND116 Dieu 5 Khoan 2)",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-5-khoan-1-diem-c": 3,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-004": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-013": 2,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # S-16 | medium
        query="Trường sư phạm có được phép giảm bớt số tháng hỗ trợ nếu đào tạo theo tín chỉ không?",
        description="Semantic: credit-based programme and support month cap (ND116 Dieu 4 Khoan 2)",
        category="semantic", difficulty="medium", is_answerable=True,
        relevant_chunks={"116-2020-ND-CP-dieu-4-khoan-2": 3, "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-003": 2, "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-007": 2},
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # S-17 | hard
        query="Quy định năm 2025 có thay đổi gì về thời hạn sinh viên phải nộp tiền bồi hoàn kể từ khi có thông báo?",
        description="Semantic (temporal): ND60 amendment to reimbursement payment deadline",
        category="semantic", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-4": 3,
            "116-2020-ND-CP-dieu-9-khoan-3": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-020": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-027": 2,
        },
        relevant_documents=[_ND60, _ND116, _QA116],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # S-18 | hard
        query="Sinh viên tốt nghiệp loại giỏi văn bằng thứ nhất học sư phạm có được hưởng hỗ trợ không?",
        description="Semantic: second-degree excellent students eligibility (ND116 Dieu 1 Khoan 2 Diem a)",
        category="semantic", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-1-khoan-2-diem-a": 3,
            "116-2020-ND-CP-dieu-1-khoan-1": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-031": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-004": 2,
        },
        relevant_documents=[_ND116, _QA116],
        notes="Hard negative: 116-2020-ND-CP-dieu-1-khoan-3 (exclusion for in-service upgrade training).",
    ),
    EvalQueryV2(  # S-19 | hard
        query="Giáo viên trường tư thục dạy học theo chương trình nhà nước có phải trả lại tiền hỗ trợ sư phạm không?",
        description="Semantic: private-school employment and reimbursement obligation",
        category="semantic", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-2-khoan-2-diem-a": 3,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-a": 2,
            "116-2020-ND-CP-dieu-6-khoan-1-diem-a": 1,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-019": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-029": 2,
        },
        relevant_documents=[_ND116, _QA116],
        notes="Dieu 2 Khoan 2 Diem a covers co so giao duc which includes private schools.",
    ),
    EvalQueryV2(  # S-20 | hard
        query="Nghị định 116 có hiệu lực từ ngày nào và áp dụng từ khóa tuyển sinh nào?",
        description="Semantic: effective date and first applicable admission intake (ND116 Dieu 15 Khoan 1)",
        category="semantic", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-15-khoan-1": 3,
        },
        relevant_documents=[_ND116],
    ),
    EvalQueryV2(  # C-01 | medium
        query="Sinh viên ngành sư phạm có bị ràng buộc gì sau khi ra trường không?",
        description="Contextual: implicit -- post-graduation obligations/commitments",
        category="contextual", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-1-diem-a": 3,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-a": 3,
            "116-2020-ND-CP-dieu-8-khoan-2": 2,
            "116-2020-ND-CP-dieu-13-khoan-1": 2,
            "116-2020-ND-CP-dieu-13-khoan-2": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-029": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-019": 3,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # C-02 | medium
        query="Nếu tôi học sư phạm mà muốn chuyển sang ngành khác thì sao?",
        description="Contextual: implicit -- switching majors triggers reimbursement",
        category="contextual", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-1-diem-c": 3,
            "116-2020-ND-CP-dieu-8-khoan-2": 3,
            "116-2020-ND-CP-dieu-8-khoan-1": 2,
        },
        relevant_documents=[_ND116],
    ),
    EvalQueryV2(  # C-03 | medium
        query="Trách nhiệm của Bộ Giáo dục và Đào tạo trong việc xác định và thông báo chỉ tiêu đào tạo giáo viên là gì?",
        description="Contextual: MOET responsibilities for determining and notifying teacher-training quotas",
        category="contextual", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-10-khoan-1-diem-b": 3,
            "116-2020-ND-CP-dieu-10-khoan-1-diem-a": 2,
            "116-2020-ND-CP-dieu-10-khoan-1-diem-c": 1,
        },
        relevant_documents=[_ND116],
    ),
    EvalQueryV2(  # C-04 | medium
        query="Trường đại học sư phạm có phải công khai chỉ tiêu tuyển sinh trên cổng thông tin điện tử không?",
        description="Contextual: public disclosure of teacher-training admission quotas on e-portals",
        category="contextual", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "60-2025-ND-CP-dieu-1-khoan-3-sua-doi-bo-sung-dieu-3-khoan-1-diem-c": 3,
            "116-2020-ND-CP-dieu-3-khoan-2": 2,
        },
        relevant_documents=[_ND60, _ND116],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # C-05 | medium
        query="Khi nào một sinh viên sư phạm bị xem là vi phạm cam kết và bị thu hồi tiền?",
        description="Contextual: implicit -- conditions triggering reimbursement demand",
        category="contextual", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-1-diem-a": 3,
            "116-2020-ND-CP-dieu-6-khoan-1-diem-b": 3,
            "116-2020-ND-CP-dieu-6-khoan-1-diem-c": 3,
            "116-2020-ND-CP-dieu-9-khoan-1": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-022": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-021": 1,
        },
        relevant_documents=[_ND116, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # C-06 | medium
        query="Theo Nghị định 60/2025, cơ quan nào ra thông báo thu hồi tiền bồi hoàn tùy theo hình thức hỗ trợ?",
        description="Contextual: recovery-notice authority under allocated-budget versus commissioned support",
        category="contextual", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-2": 3,
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-3": 3,
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-1": 1,
        },
        relevant_documents=[_ND60],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # C-07 | medium
        query="Có quy định nào về việc giáo viên tiếp tục được cử đi học lên cao hơn sau khi tốt nghiệp không?",
        description="Contextual: higher teacher training continuation as exemption from reimbursement",
        category="contextual", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-2-diem-c": 3,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-a": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-009": 2,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # C-08 | medium
        query="Nghị định 116 có áp dụng đối với các tổ chức, cá nhân có nhu cầu đào tạo giáo viên không?",
        description="Contextual: applicability to organisations and individuals with teacher-training needs",
        category="contextual", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-1-khoan-2-diem-b": 3,
        },
        relevant_documents=[_ND116],
    ),
    EvalQueryV2(  # C-09 | medium
        query="Danh sách sinh viên sư phạm hưởng hỗ trợ có phải công khai không và được công khai ở đâu?",
        description="Contextual: transparency of beneficiary lists and disclosure locations",
        category="contextual", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-7-khoan-6": 3,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-008": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-004": 2,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # C-10 | medium
        query="điều kiện để đăng ký hưởng hỗ trợ tiền học phí và sinh hoạt phí là gì?",
        description="Contextual: registration conditions for support (process/eligibility)",
        category="contextual", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-7-khoan-1": 3,
            "116-2020-ND-CP-dieu-7-khoan-2": 3,
            "116-2020-ND-CP-dieu-7-khoan-3": 2,
            "116-2020-ND-CP-dieu-1-khoan-2-diem-a": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-005": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-007": 2,
        },
        relevant_documents=[_ND116, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # C-11 | medium
        query="Có thể kiện trường hoặc địa phương nếu không được nhận tiền hỗ trợ hợp lệ không?",
        description="Contextual: dispute/remedy mechanisms -- implied by enforcement provisions",
        category="contextual", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-14": 3,
            "116-2020-ND-CP-dieu-9-khoan-6": 1,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-021": 1,
        },
        relevant_documents=[_ND116, _QA116],
        notes="No explicit dispute-resolution clause; Dieu 14 is the closest provision.",
    ),
    EvalQueryV2(  # C-12 | medium
        query="Khi chuyển công tác ra ngoài ngành giáo dục theo quyết định nhà nước thì có bị phạt không?",
        description="Contextual: state-ordered reassignment outside education -- exemption from reimbursement",
        category="contextual", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-2-diem-b": 3,
            "116-2020-ND-CP-dieu-6-khoan-1-diem-b": 2,
        },
        relevant_documents=[_ND116],
    ),
    EvalQueryV2(  # C-13 | hard
        query="Chính sách hỗ trợ học phí và sinh hoạt phí cho sinh viên sư phạm được hình thành từ cơ sở pháp lý nào?",
        description="Contextual: legal authority basis for the support policy (LGD -> ND116)",
        category="contextual", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-4": 3,
            "116-2020-ND-CP-dieu-1-khoan-1": 3,
            "60-2025-ND-CP-dieu-1-khoan-1": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-004": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-013": 2,
        },
        relevant_documents=[_LGD, _ND116, _ND60, _QA116],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # C-14 | hard
        query="Sinh viên sư phạm dừng học do ốm đau hoặc tai nạn thì có tiếp tục được hưởng chính sách hỗ trợ không?",
        description="Contextual: temporary study interruption for illness or accident and continued support (ND116 Dieu 6 Khoan 4)",
        category="contextual", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-4": 3,
            "116-2020-ND-CP-dieu-6-khoan-3": 1,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-014": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-012": 2,
        },
        relevant_documents=[_ND116, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # C-15 | hard
        query="UBND cấp tỉnh có thể đặt hàng đào tạo giáo viên với cơ sở đào tạo không?",
        description="Contextual: provincial ordering authority for teacher training",
        category="contextual", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-1-khoan-2-diem-b": 3,
            "116-2020-ND-CP-dieu-3-khoan-3-diem-b": 3,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-010": 2,
        },
        relevant_documents=[_ND116, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # C-16 | hard
        query="Sinh viên học chương trình liên thông chính quy có được hưởng chính sách hỗ trợ giống sinh viên chính quy thông thường không?",
        description="Contextual: articulation programme student eligibility",
        category="contextual", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-1-khoan-2-diem-a": 3,
            "116-2020-ND-CP-dieu-1-khoan-1": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-004": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-007": 2,
        },
        relevant_documents=[_ND116, _QA116],
        notes="Dieu 1 Khoan 2 Diem a explicitly mentions 'lien thong chinh quy' as eligible.",
    ),
    EvalQueryV2(  # C-17 | hard
        query="Quy định sửa đổi năm 2025 có xóa bỏ hình thức đấu thầu đào tạo giáo viên không?",
        description="Contextual: ND60 removes the old tender reference while preserving existing signed contracts",
        category="contextual", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "60-2025-ND-CP-dieu-2-khoan-3": 3,
            "60-2025-ND-CP-dieu-2-khoan-2": 2,
            "116-2020-ND-CP-dieu-3-khoan-3-diem-c": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-010": 2,
        },
        relevant_documents=[_ND60, _ND116, _QA116],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # C-18 | hard
        query="Có trường hợp nào mà sinh viên sư phạm bị buộc thôi học mà vẫn không phải hoàn trả tiền không?",
        description="Contextual: forced withdrawal (discipline) and whether any exemption applies",
        category="contextual", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-1-diem-c": 3,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-a": 1,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-b": 1,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-c": 1,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-028": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-025": 3,
        },
        relevant_documents=[_ND116, _QA116],
        notes="Khoan 2 exemptions do NOT cover disciplinary expulsion -- they are hard negatives (grade 1).",
    ),
    EvalQueryV2(  # C-19 | hard
        query="Theo Luật Giáo dục 2019, Hội đồng nhân dân cấp tỉnh có thẩm quyền gì trong việc quyết định mức học phí của cơ sở giáo dục công lập?",
        description="Contextual: provincial People's Council authority over tuition for public institutions",
        category="contextual", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-99-khoan-6-diem-b": 3,
            "LUAT-GIAO-DUC-2019-dieu-99-khoan-6-diem-a": 2,
        },
        relevant_documents=[_LGD],
    ),
    EvalQueryV2(  # C-20 | hard
        query="Nếu sinh viên sư phạm bị kỷ luật buộc thôi học thì chính sách hỗ trợ có bị ảnh hưởng không?",
        description="Contextual: disciplinary expulsion and reimbursement obligation",
        category="contextual", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-1-diem-c": 3,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-012": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-025": 3,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # C-21 | hard
        query="Trường hợp suy giảm khả năng lao động từ 61% trở lên, sinh viên sư phạm có phải hoàn trả tiền không?",
        description="Contextual: disability threshold >=61% and reimbursement waiver (ND60)",
        category="contextual", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-5": 3,
            "116-2020-ND-CP-dieu-9-khoan-4": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-028": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-023": 2,
        },
        relevant_documents=[_ND60, _ND116, _QA116],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # C-22 | hard
        query="Giáo viên được phép dạy thêm ngoài giờ không và quy định nào điều chỉnh việc này?",
        description="Contextual: extra tutoring -- LGD teacher rights vs prohibitions",
        category="contextual", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-69-khoan-2": 3,
            "LUAT-GIAO-DUC-2019-dieu-70-khoan-1": 2,
        },
        relevant_documents=[_LGD],
    ),
    EvalQueryV2(  # C-23 | hard
        query="Ai có thẩm quyền xác nhận rằng một sinh viên sư phạm đã làm đủ thời gian công tác trong ngành giáo dục?",
        description="Contextual: authority to certify qualifying service period",
        category="contextual", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-2-khoan-3": 3,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-009": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-018": 2,
        },
        relevant_documents=[_ND116, _QA116],
    ),
    EvalQueryV2(  # C-24 | hard
        query="Khi kinh phí hỗ trợ được cấp theo dạng giao dự toán, trường sư phạm phải chuyển tiền sinh hoạt cho sinh viên muộn nhất vào ngày nào?",
        description="Contextual: payment deadline after budget support is received (ND60 Dieu 5 Khoan 2 Diem b)",
        category="contextual", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "60-2025-ND-CP-dieu-1-khoan-4-sua-doi-bo-sung-dieu-5-khoan-2-diem-b": 3,
            "60-2025-ND-CP-dieu-1-khoan-4-sua-doi-bo-sung-dieu-5-khoan-2-diem-a": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-020": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-002": 2,
        },
        relevant_documents=[_ND60, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # C-25 | hard
        query="Nhà giáo có những quyền gì liên quan đến giảng dạy theo chuyên môn và bồi dưỡng nâng cao trình độ?",
        description="Contextual: teacher rights to teach in trained specialism and receive professional development",
        category="contextual", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-70-khoan-1": 3,
            "LUAT-GIAO-DUC-2019-dieu-70-khoan-2": 3,
        },
        relevant_documents=[_LGD],
    ),
    EvalQueryV2(  # MC-01 | medium
        query="Tôi có phải hoàn trả tiền hỗ trợ không, và những trường hợp nào được miễn?",
        description="Multi-chunk: obligation (Khoan 1) + exemption (Khoan 2) -- ND116 Dieu 6",
        category="multi_chunk", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-1-diem-a": 3,
            "116-2020-ND-CP-dieu-6-khoan-1-diem-b": 3,
            "116-2020-ND-CP-dieu-6-khoan-1-diem-c": 3,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-a": 3,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-b": 2,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-c": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-024": 3,
        },
        relevant_documents=[_ND116, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # MC-02 | medium
        query="Khoản nào quy định chi phí phải bồi hoàn và cách tính số tiền phải trả lại, toàn bộ hoặc một phần?",
        description="Multi-chunk: cost base (Khoan 1) + full amount (Khoan 2) + partial formula (Khoan 3) -- ND116 Dieu 8",
        category="multi_chunk", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-8-khoan-1": 3,
            "116-2020-ND-CP-dieu-8-khoan-2": 3,
            "116-2020-ND-CP-dieu-8-khoan-3": 3,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-016": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-022": 2,
        },
        relevant_documents=[_ND116, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # MC-03 | medium
        query="Nếu nhận thông báo bồi hoàn theo điều 9 Nghị định 116, sinh viên phải nộp tiền theo quy trình nào?",
        description="Multi-chunk: full reimbursement workflow across ND116 Dieu 9 clauses",
        category="multi_chunk", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-9-khoan-1": 3,
            "116-2020-ND-CP-dieu-9-khoan-2": 3,
            "116-2020-ND-CP-dieu-9-khoan-3": 3,
            "116-2020-ND-CP-dieu-9-khoan-4": 2,
            "116-2020-ND-CP-dieu-9-khoan-5": 2,
            "116-2020-ND-CP-dieu-9-khoan-6": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-022": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-016": 1,
        },
        relevant_documents=[_ND116, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # MC-04 | medium
        query="Địa phương xác định nhu cầu tuyển giáo viên rồi chuyển thành chỉ tiêu tuyển sinh theo quy trình nào tại điều 3 Nghị định 116?",
        description="Multi-chunk: demand-to-quota workflow across Dieu 3 clauses",
        category="multi_chunk", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-3-khoan-1": 3,
            "116-2020-ND-CP-dieu-3-khoan-2": 3,
            "116-2020-ND-CP-dieu-3-khoan-3-diem-a": 2,
            "116-2020-ND-CP-dieu-3-khoan-3-diem-b": 2,
            "116-2020-ND-CP-dieu-3-khoan-3-diem-c": 2,
            "116-2020-ND-CP-dieu-3-khoan-4": 1,
        },
        relevant_documents=[_ND116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # MC-05 | medium
        query="UBND cấp tỉnh có những trách nhiệm gì khi triển khai chính sách hỗ trợ sinh viên sư phạm?",
        description="Multi-chunk: full provincial responsibilities across ND116 Dieu 11",
        category="multi_chunk", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-11-khoan-1": 3,
            "116-2020-ND-CP-dieu-11-khoan-2": 3,
            "116-2020-ND-CP-dieu-11-khoan-3": 3,
            "116-2020-ND-CP-dieu-11-khoan-4": 3,
            "116-2020-ND-CP-dieu-11-khoan-5": 3,
            "116-2020-ND-CP-dieu-11-khoan-6": 3,
            "116-2020-ND-CP-dieu-11-khoan-7": 3,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-013": 2,
        },
        relevant_documents=[_ND116, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # MC-06 | medium
        query="điều 12 Nghị định 116 giao cho cơ sở đào tạo giáo viên bao nhiêu trách nhiệm, gồm những việc gì?",
        description="Multi-chunk: all institutional responsibilities across ND116 Dieu 12",
        category="multi_chunk", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-12-khoan-1": 3,
            "116-2020-ND-CP-dieu-12-khoan-2": 3,
            "116-2020-ND-CP-dieu-12-khoan-3": 3,
            "116-2020-ND-CP-dieu-12-khoan-4": 3,
            "116-2020-ND-CP-dieu-12-khoan-5": 3,
            "116-2020-ND-CP-dieu-12-khoan-6": 3,
            "116-2020-ND-CP-dieu-12-khoan-7": 3,
        },
        relevant_documents=[_ND116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # MC-07 | medium
        query="Theo Nghị định 60/2025, những điểm nào của khoản 1 Điều 3 tạo thành quy trình từ xác định nhu cầu đến công khai chỉ tiêu tuyển sinh?",
        description="Multi-chunk: ND60 Dieu 3 Khoan 1 workflow from local demand to public quota disclosure",
        category="multi_chunk", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "60-2025-ND-CP-dieu-1-khoan-3-sua-doi-bo-sung-dieu-3-khoan-1-diem-a": 3,
            "60-2025-ND-CP-dieu-1-khoan-3-sua-doi-bo-sung-dieu-3-khoan-1-diem-b": 3,
            "60-2025-ND-CP-dieu-1-khoan-3-sua-doi-bo-sung-dieu-3-khoan-1-diem-c": 3,
        },
        relevant_documents=[_ND60],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # MC-08 | medium
        query="Sau sửa đổi năm 2025, Điều 5 quy định việc lập dự toán, cấp kinh phí và chi trả hỗ trợ cho sinh viên sư phạm như thế nào?",
        description="Multi-chunk: ND60 amended Dieu 5 budget planning, funding source, payment and settlement",
        category="multi_chunk", difficulty="medium", is_answerable=True,
        relevant_chunks={
            "60-2025-ND-CP-dieu-1-khoan-4-sua-doi-bo-sung-dieu-5-khoan-1-diem-a": 2,
            "60-2025-ND-CP-dieu-1-khoan-4-sua-doi-bo-sung-dieu-5-khoan-1-diem-b": 2,
            "60-2025-ND-CP-dieu-1-khoan-4-sua-doi-bo-sung-dieu-5-khoan-1-diem-c": 2,
            "60-2025-ND-CP-dieu-1-khoan-4-sua-doi-bo-sung-dieu-5-khoan-2-diem-a": 3,
            "60-2025-ND-CP-dieu-1-khoan-4-sua-doi-bo-sung-dieu-5-khoan-2-diem-b": 3,
            "60-2025-ND-CP-dieu-1-khoan-4-sua-doi-bo-sung-dieu-5-khoan-3": 2,
            "60-2025-ND-CP-dieu-1-khoan-4-sua-doi-bo-sung-dieu-5-khoan-4": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-002": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-013": 2,
        },
        relevant_documents=[_ND60, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # MC-09 | hard
        query="Muốn biết mình đã làm đủ thời gian để được miễn hoàn trả chưa, cần căn cứ vào cách tính thời gian, điều kiện miễn và cơ quan xác nhận nào?",
        description="Multi-chunk: full exemption verification chain -- Dieu 6 Khoan 2 + Dieu 2 + Khoan 3",
        category="multi_chunk", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-2-diem-a": 3,
            "116-2020-ND-CP-dieu-2-khoan-1": 3,
            "116-2020-ND-CP-dieu-2-khoan-2-diem-a": 3,
            "116-2020-ND-CP-dieu-2-khoan-2-diem-b": 2,
            "116-2020-ND-CP-dieu-2-khoan-3": 2,
        },
        relevant_documents=[_ND116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # MC-10 | hard
        query="Theo Luật Giáo dục 2019, nhà giáo có những quyền gì về chuyên môn, bồi dưỡng và đời sống?",
        description="Multi-chunk: full teacher rights across LGD Dieu 70 clauses",
        category="multi_chunk", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-70-khoan-1": 3,
            "LUAT-GIAO-DUC-2019-dieu-70-khoan-2": 3,
            "LUAT-GIAO-DUC-2019-dieu-70-khoan-3": 3,
            "LUAT-GIAO-DUC-2019-dieu-70-khoan-4": 2,
            "LUAT-GIAO-DUC-2019-dieu-70-khoan-5": 2,
        },
        relevant_documents=[_LGD],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # MC-11 | hard
        query="Trước khi được sửa đổi, điều 5 Nghị định 116 quy định việc lập dự toán và chi trả kinh phí hỗ trợ ra sao?",
        description="Multi-chunk: ND116 Dieu 5 original budget planning and payment provisions",
        category="multi_chunk", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-5-khoan-1-diem-a": 3,
            "116-2020-ND-CP-dieu-5-khoan-1-diem-b": 3,
            "116-2020-ND-CP-dieu-5-khoan-1-diem-a-2": 3,
            "116-2020-ND-CP-dieu-5-khoan-1-diem-b-2": 3,
            "116-2020-ND-CP-dieu-5-khoan-1-diem-c": 3,
            "116-2020-ND-CP-dieu-5-khoan-3": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-002": 2,
        },
        relevant_documents=[_ND116, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # MC-12 | hard
        query="Theo Nghị định 60/2025, quy trình thu hồi tiền bồi hoàn khác nhau thế nào giữa sinh viên được giao nhiệm vụ và sinh viên được giao dự toán?",
        description="Multi-chunk: ND60 Dieu 9 recovery workflow for commissioned vs allocated-budget students",
        category="multi_chunk", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-2": 3,
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-3": 3,
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-4": 3,
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-5": 2,
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-6": 2,
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-1": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-022": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-016": 1,
        },
        relevant_documents=[_ND60, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # MC-13 | hard
        query="Theo Luật Giáo dục 2019, người học có thể được hưởng những hình thức học bổng, trợ cấp, miễn giảm học phí và hỗ trợ nào?",
        description="Multi-chunk: LGD Dieu 85 provisions on scholarships, social assistance, tuition relief and pedagogy support",
        category="multi_chunk", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-1": 3,
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-2": 3,
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-3": 2,
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-4": 3,
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-5": 2,
        },
        relevant_documents=[_LGD],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # MC-14 | hard
        query="Sau khi Nghị định 60/2025 sửa đổi, trách nhiệm của cơ sở đào tạo giáo viên tại Điều 12 thay đổi hoặc được bổ sung những gì so với Nghị định 116?",
        description="Multi-chunk: ND116 Dieu 12 versus ND60 amendments to Dieu 12 Khoan 3 and new Khoan 8",
        category="multi_chunk", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-12-khoan-3": 3,
            "60-2025-ND-CP-dieu-1-khoan-10-sua-doi-bo-sung-dieu-12-khoan-3": 3,
            "60-2025-ND-CP-dieu-1-khoan-10-bo-sung-dieu-12-khoan-8": 3,
        },
        relevant_documents=[_ND60, _ND116],
        requires_multi_chunk=True,
        requires_multi_document=True,
    ),
    EvalQueryV2(  # MC-15 | hard
        query="Nghị định 60/2025 thay điều 13 cũ bằng quy định nào về nghĩa vụ của sinh viên sau khi tốt nghiệp?",
        description="Multi-chunk: ND60 Dieu 13 full beneficiary post-graduation obligations",
        category="multi_chunk", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "60-2025-ND-CP-dieu-1-khoan-11-sua-doi-bo-sung-dieu-13-khoan-1": 3,
            "60-2025-ND-CP-dieu-1-khoan-11-sua-doi-bo-sung-dieu-13-khoan-2": 3,
            "60-2025-ND-CP-dieu-1-khoan-11-sua-doi-bo-sung-dieu-13-khoan-3": 3,
            "60-2025-ND-CP-dieu-1-khoan-11-sua-doi-bo-sung-dieu-13-khoan-4": 2,
            "60-2025-ND-CP-dieu-1-khoan-11-sua-doi-bo-sung-dieu-13-khoan-5": 2,
            "116-2020-ND-CP-dieu-13-khoan-1": 1,
            "116-2020-ND-CP-dieu-13-khoan-2": 1,
            "116-2020-ND-CP-dieu-13-khoan-3": 1,
        },
        relevant_documents=[_ND60, _ND116],
        requires_multi_chunk=True,
        requires_multi_document=True,
    ),

    # -----------------------------------------------------------------------
    # MULTI_DOCUMENT  (MD-01 .. MD-10)
    # -----------------------------------------------------------------------
    EvalQueryV2(  # MD-01 | hard
        query="Luật Giáo dục 2019 ghi nhận chính sách hỗ trợ học sinh, sinh viên sư phạm như thế nào; Nghị định 116 triển khai chính sách đó ra sao và Nghị định 60/2025 sửa phạm vi đối tượng thế nào?",
        description="Multi-doc: LGD policy basis -> ND116 implementation -> ND60 scope amendment",
        category="multi_document", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-4": 3,
            "116-2020-ND-CP-dieu-1-khoan-1": 3,
            "60-2025-ND-CP-dieu-1-khoan-1": 3,
            "60-2025-ND-CP-dieu-1-khoan-2": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-001": 1,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-030": 1,
        },
        relevant_documents=[_LGD, _ND116, _ND60, _QA116],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # MD-02 | hard
        query="Luật Giáo dục 2019 có quy định nào liên quan đến việc sinh viên sư phạm phải hoàn trả tiền hỗ trợ sau khi tốt nghiệp?",
        description="Multi-doc: LGD authority for support + ND116 repayment obligation",
        category="multi_document", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-4": 3,
            "116-2020-ND-CP-dieu-6-khoan-1-diem-a": 3,
            "116-2020-ND-CP-dieu-6-khoan-1-diem-b": 3,
            "116-2020-ND-CP-dieu-6-khoan-1-diem-c": 3,
            "116-2020-ND-CP-dieu-8-khoan-1": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-019": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-023": 2,
        },
        relevant_documents=[_LGD, _ND116, _QA116],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # MD-03 | hard
        query="Nghị định 60/2025 đã thay đổi trách nhiệm của cơ sở đào tạo giáo viên tại Điều 12 như thế nào so với Nghị định 116?",
        description="Multi-doc: ND116 Dieu 12 versus ND60 amendments to Dieu 12 Khoan 3 and new Khoan 8",
        category="multi_document", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-12-khoan-3": 3,
            "60-2025-ND-CP-dieu-1-khoan-10-sua-doi-bo-sung-dieu-12-khoan-3": 3,
            "60-2025-ND-CP-dieu-1-khoan-10-bo-sung-dieu-12-khoan-8": 3,
        },
        relevant_documents=[_ND116, _ND60],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # MD-04 | hard
        query="Theo Luật Giáo dục 2019, nhà giáo được xác định về vị trí, tiêu chuẩn như thế nào, và các quy định đó liên quan thế nào đến phạm vi đào tạo giáo viên của Nghị định 116?",
        description="Multi-doc: LGD teacher role and standards mapped to ND116 teacher-training policy scope",
        category="multi_document", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-66-khoan-1": 3,
            "LUAT-GIAO-DUC-2019-dieu-67-khoan-1": 3,
            "LUAT-GIAO-DUC-2019-dieu-67-khoan-2": 3,
            "LUAT-GIAO-DUC-2019-dieu-67-khoan-3": 3,
            "LUAT-GIAO-DUC-2019-dieu-67-khoan-4": 3,
            "116-2020-ND-CP-dieu-1-khoan-1": 2,
        },
        relevant_documents=[_LGD, _ND116],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # MD-05 | hard
        query="Cần đối chiếu những quy định nào trong Luật Giáo dục 2019, Nghị định 116 và Nghị định 60/2025 để xác định chính sách hỗ trợ, trường hợp phải bồi hoàn và trách nhiệm của người nhận hỗ trợ?",
        description="Multi-doc: cross-corpus synthesis -- support, repayment and beneficiary duties",
        category="multi_document", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-4": 3,
            "116-2020-ND-CP-dieu-6-khoan-1-diem-a": 3,
            "116-2020-ND-CP-dieu-6-khoan-1-diem-b": 3,
            "116-2020-ND-CP-dieu-6-khoan-1-diem-c": 3,
            "116-2020-ND-CP-dieu-8-khoan-1": 2,
            "116-2020-ND-CP-dieu-8-khoan-2": 3,
            "60-2025-ND-CP-dieu-1-khoan-11-sua-doi-bo-sung-dieu-13-khoan-1": 3,
            "60-2025-ND-CP-dieu-1-khoan-11-sua-doi-bo-sung-dieu-13-khoan-2": 3,
            "60-2025-ND-CP-dieu-1-khoan-11-sua-doi-bo-sung-dieu-13-khoan-3": 3,
            "60-2025-ND-CP-dieu-1-khoan-11-sua-doi-bo-sung-dieu-13-khoan-4": 2,
            "60-2025-ND-CP-dieu-1-khoan-11-sua-doi-bo-sung-dieu-13-khoan-5": 2,
        },
        relevant_documents=[_LGD, _ND116, _ND60],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # MD-06 | hard
        query="Luật Giáo dục 2019 quy định nhà giáo phải thực hiện những nhiệm vụ gì, và Nghị định 116 yêu cầu cơ sở đào tạo giáo viên bảo đảm chất lượng, đào tạo ra sao?",
        description="Multi-doc: teacher duties under LGD mapped to training and quality responsibilities under ND116",
        category="multi_document", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-69-khoan-1": 3,
            "LUAT-GIAO-DUC-2019-dieu-69-khoan-4": 3,
            "116-2020-ND-CP-dieu-12-khoan-1": 3,
            "116-2020-ND-CP-dieu-12-khoan-6": 3,
        },
        relevant_documents=[_LGD, _ND116],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # MD-07 | hard
        query="Nghị định 60/2025 có thay đổi điều kiện hưởng hỗ trợ hoặc được miễn bồi hoàn so với Nghị định 116 ban đầu không? Cụ thể là gì?",
        description="Multi-doc: ND116 eligibility/exemption vs ND60 amendments",
        category="multi_document", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-2-diem-a": 3,
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-5": 3,
            "116-2020-ND-CP-dieu-9-khoan-4": 2,
            "60-2025-ND-CP-dieu-1-khoan-1": 2,
            "116-2020-ND-CP-dieu-1-khoan-2-diem-a": 1,
        },
        relevant_documents=[_ND116, _ND60],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # MD-08 | hard
        query="Luật Giáo dục 2019 quy định chính sách miễn, giảm học phí và hỗ trợ đối với sinh viên sư phạm ra sao, và Nghị định 116 cụ thể hóa hỗ trợ này như thế nào?",
        description="Multi-doc: LGD fee relief and pedagogy support versus ND116 implementation",
        category="multi_document", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-2": 3,
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-4": 3,
            "116-2020-ND-CP-dieu-4-khoan-1": 3,
            "116-2020-ND-CP-dieu-1-khoan-1": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-001": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-004": 2,
        },
        relevant_documents=[_LGD, _ND116, _QA116],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # MD-09 | hard
        query="Nghị định 116 quy định các hình thức giao nhiệm vụ, đặt hàng và đấu thầu đào tạo giáo viên như thế nào, và Nghị định 60/2025 thay đổi cơ chế này ra sao?",
        description="Multi-doc: teacher-training procurement/commissioning forms before and after ND60",
        category="multi_document", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-3-khoan-3-diem-a": 3,
            "116-2020-ND-CP-dieu-3-khoan-3-diem-b": 3,
            "116-2020-ND-CP-dieu-3-khoan-3-diem-c": 3,
            "60-2025-ND-CP-dieu-1-khoan-3-sua-doi-bo-sung-dieu-3-khoan-2-diem-a": 3,
            "60-2025-ND-CP-dieu-1-khoan-3-sua-doi-bo-sung-dieu-3-khoan-2-diem-b": 3,
            "60-2025-ND-CP-dieu-1-khoan-3-sua-doi-bo-sung-dieu-3-khoan-2-diem-c": 2,
            "60-2025-ND-CP-dieu-1-khoan-3-sua-doi-bo-sung-dieu-3-khoan-2-diem-d": 2,
            "60-2025-ND-CP-dieu-2-khoan-2": 2,
            "60-2025-ND-CP-dieu-2-khoan-3": 3,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-010": 2,
        },
        relevant_documents=[_ND116, _ND60, _QA116],
        requires_multi_document=True,
    ),
    EvalQueryV2(  # MD-10 | hard
        query="Quy định về chức danh giáo sư, phó giáo sư trong Luật Giáo dục 2019 liên quan gì đến chính sách đào tạo giáo viên trong các Nghị định?",
        description="Multi-doc: LGD Dieu 68 professor titles vs ND116 pedagogy training policy",
        category="multi_document", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "LUAT-GIAO-DUC-2019-dieu-68-khoan-1": 3,
            "LUAT-GIAO-DUC-2019-dieu-68-khoan-2": 2,
            "116-2020-ND-CP-dieu-1-khoan-1": 1,
        },
        relevant_documents=[_LGD, _ND116],
        requires_multi_document=True,
        notes="The connection is indirect; tests cross-document reasoning with a weaker link.",
    ),

    # -----------------------------------------------------------------------
    # COMPLEX_QA  (CQ-01 .. CQ-10)
    # -----------------------------------------------------------------------
    EvalQueryV2(  # CQ-01 | hard
        query="Một sinh viên sư phạm đã nhận hỗ trợ nhưng sau khi tốt nghiệp quá 2 năm vẫn không làm trong ngành giáo dục; quy định nào áp dụng và số tiền phải trả là bao nhiêu?",
        description="Complex Q&A: no education employment 2 years post-graduation -> full reimbursement",
        category="complex_qa", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-1-diem-a": 3,
            "116-2020-ND-CP-dieu-8-khoan-2": 3,
            "116-2020-ND-CP-dieu-8-khoan-1": 3,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-a": 2,
            "116-2020-ND-CP-dieu-9-khoan-2": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-017": 3,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-019": 2,
        },
        relevant_documents=[_ND116, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # CQ-02 | hard
        query="Sinh viên vẫn đang học, tự thôi học giữa chừng và đã nhận học phí cùng sinh hoạt phí; phải tra quy định nào để tính số tiền phải trả lại?",
        description="Complex Q&A: voluntary withdrawal during study -- obligation and calculation",
        category="complex_qa", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-1-diem-c": 3,
            "116-2020-ND-CP-dieu-8-khoan-2": 3,
            "116-2020-ND-CP-dieu-8-khoan-1": 3,
            "116-2020-ND-CP-dieu-9-khoan-2": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-028": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-027": 2,
        },
        relevant_documents=[_ND116, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # CQ-03 | hard
        query="Sinh viên đã làm trong ngành giáo dục nhưng chưa đủ thời gian bắt buộc; muốn biết có phải trả toàn bộ hay một phần và theo công thức nào thì cần xem những điều Khoản nào?",
        description="Complex Q&A: insufficient service + partial vs full reimbursement formula",
        category="complex_qa", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-6-khoan-1-diem-b": 3,
            "116-2020-ND-CP-dieu-8-khoan-3": 3,
            "116-2020-ND-CP-dieu-8-khoan-1": 3,
            "116-2020-ND-CP-dieu-8-khoan-2": 2,
            "116-2020-ND-CP-dieu-2-khoan-1": 2,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-018": 1,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-016": 1,
        },
        relevant_documents=[_ND116, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # CQ-04 | hard
        query="Một sinh viên thuộc diện giao nhiệm vụ đã bị xác định phải bồi hoàn; cơ quan nào thu hồi và sinh viên phải nộp trả theo quy trình nào sau sửa đổi năm 2025?",
        description="Complex Q&A: commissioned student, recovery authority and process under ND60",
        category="complex_qa", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-3": 3,
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-4": 3,
            "60-2025-ND-CP-dieu-1-khoan-11-sua-doi-bo-sung-dieu-13-khoan-3": 3,
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-5": 2,
            "116-2020-ND-CP-dieu-9-khoan-3": 1,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-022": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-016": 1,
        },
        relevant_documents=[_ND60, _ND116, _QA116],
        requires_multi_chunk=True,
        requires_multi_document=True,
    ),
    EvalQueryV2(  # CQ-05 | hard
        query="Một sinh viên được hưởng hỗ trợ nhưng thuộc trường hợp suy giảm khả năng lao động từ 61% trở lên; theo quy định sửa đổi, có được miễn bồi hoàn không và điều kiện là gì?",
        description="Complex Q&A: disability >=61% and reimbursement waiver",
        category="complex_qa", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "60-2025-ND-CP-dieu-1-khoan-7-sua-doi-bo-sung-dieu-9-khoan-5": 3,
            "116-2020-ND-CP-dieu-9-khoan-4": 2,
        },
        relevant_documents=[_ND60, _ND116],
        requires_multi_chunk=True,
        requires_multi_document=True,
    ),
    EvalQueryV2(  # CQ-06 | hard
        query="Muốn xác định rõ thời gian công tác tối thiểu để không phải hoàn trả tiền, cần kết hợp quy định về cách tính tháng và quy định về thời gian đủ điều kiện miễn trừ như thế nào?",
        description="Complex Q&A: minimum service calculation -- rounding rule + exemption duration",
        category="complex_qa", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-2-khoan-1": 3,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-a": 3,
            "116-2020-ND-CP-dieu-4-khoan-2": 2,
            "116-2020-ND-CP-dieu-2-khoan-2-diem-a": 2,
        },
        relevant_documents=[_ND116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # CQ-07 | hard
        query="Một sinh viên thuộc hộ nghèo học sư phạm; ngoài hỗ trợ theo Nghị định 116, họ còn được hưởng thêm chính sách gì từ Nhà nước?",
        description="Complex Q&A: poor-household student -- ND116 support plus LGD tuition-relief policy",
        category="complex_qa", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-4-khoan-1": 2,
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-2": 3,
            "LUAT-GIAO-DUC-2019-dieu-85-khoan-4": 3,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-007": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-001": 2,
        },
        relevant_documents=[_ND116, _LGD, _QA116],
        requires_multi_document=True,
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # CQ-08 | hard
        query="Khi so sánh trách nhiệm của UBND cấp tỉnh theo Nghị định 116 và theo Nghị định 60/2025, các điểm mới bổ sung là gì?",
        description="Complex Q&A: provincial responsibilities comparison ND116 Dieu 11 vs ND60 amended Dieu 11",
        category="complex_qa", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-11-khoan-1": 2,
            "116-2020-ND-CP-dieu-11-khoan-2": 2,
            "116-2020-ND-CP-dieu-11-khoan-3": 2,
            "116-2020-ND-CP-dieu-11-khoan-4": 2,
            "116-2020-ND-CP-dieu-11-khoan-5": 2,
            "116-2020-ND-CP-dieu-11-khoan-6": 2,
            "116-2020-ND-CP-dieu-11-khoan-7": 2,
            "60-2025-ND-CP-dieu-1-khoan-9-sua-doi-bo-sung-dieu-11-khoan-1": 3,
            "60-2025-ND-CP-dieu-1-khoan-9-sua-doi-bo-sung-dieu-11-khoan-2": 3,
            "60-2025-ND-CP-dieu-1-khoan-9-sua-doi-bo-sung-dieu-11-khoan-3": 3,
            "60-2025-ND-CP-dieu-1-khoan-9-sua-doi-bo-sung-dieu-11-khoan-4": 3,
            "60-2025-ND-CP-dieu-1-khoan-9-sua-doi-bo-sung-dieu-11-khoan-5": 3,
            "60-2025-ND-CP-dieu-1-khoan-9-sua-doi-bo-sung-dieu-11-khoan-6": 3,
            "60-2025-ND-CP-dieu-1-khoan-9-sua-doi-bo-sung-dieu-11-khoan-7": 3,
            "60-2025-ND-CP-dieu-1-khoan-9-sua-doi-bo-sung-dieu-11-khoan-8": 3,
            "60-2025-ND-CP-dieu-1-khoan-9-sua-doi-bo-sung-dieu-11-khoan-9": 3,
        },
        relevant_documents=[_ND116, _ND60],
        requires_multi_chunk=True,
        requires_multi_document=True,
    ),
    EvalQueryV2(  # CQ-09 | hard
        query="Giả sử một sinh viên sư phạm tốt nghiệp, được điều động sang làm việc tại cơ quan quản lý nhà nước về giáo dục cấp huyện; điều đó có được tính là công tác trong ngành giáo dục không và có phải hoàn trả tiền không?",
        description="Complex Q&A: civil servant at district education management body -- education sector definition + exemption",
        category="complex_qa", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-2-khoan-2-diem-b": 3,
            "116-2020-ND-CP-dieu-6-khoan-2-diem-a": 3,
            "116-2020-ND-CP-dieu-2-khoan-3": 2,
            "116-2020-ND-CP-dieu-6-khoan-1-diem-a": 1,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-019": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-009": 2,
        },
        relevant_documents=[_ND116, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # CQ-10 | hard
        query="Một trường đại học sư phạm đào tạo theo tín chỉ muốn tính tổng kinh phí hỗ trợ cả khóa; mức hỗ trợ học phí và giới hạn tối đa theo Nghị định 116 được xác định như thế nào?",
        description="Complex Q&A: credit-based tuition support and whole-course support cap (ND116 Dieu 4)",
        category="complex_qa", difficulty="hard", is_answerable=True,
        relevant_chunks={
            "116-2020-ND-CP-dieu-4-khoan-1": 3,
            "116-2020-ND-CP-dieu-4-khoan-2": 3,
                    "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-001": 2,
            "hoi-dap-nghi-dinh-116-2020-nd-cp-qa-003": 2,
        },
        relevant_documents=[_ND116, _QA116],
        requires_multi_chunk=True,
    ),
    EvalQueryV2(  # OOS-01 | easy
        query="Luật Lao động 2019 quy định về sa thải người lao động trái pháp luật như thế nào?",
        description="OOS: Labour Law 2019 -- unfair dismissal (not in corpus)",
        category="out_of_scope", difficulty="easy", is_answerable=False,
        relevant_chunks={}, relevant_documents=[],
        notes="Corpus contains no Labour Law provisions.",
    ),
    EvalQueryV2(  # OOS-02 | medium
        query="Nghị định 71/2020/NĐ-CP quy định lộ trình nâng chuẩn trình độ giáo viên mầm non như thế nào?",
        description="OOS: ND71/2020 teacher upskilling roadmap -- referenced but not in corpus",
        category="out_of_scope", difficulty="medium", is_answerable=False,
        relevant_chunks={}, relevant_documents=[],
        notes="ND116 Dieu 1 Khoan 3 excludes this group and references ND71/2020, but its content is absent.",
    ),
    EvalQueryV2(  # OOS-03 | easy
        query="Theo Bộ luật Hình sự, hành vi gian lận trong thi cử đại học bị xử lý như thế nào?",
        description="OOS: Criminal Code on examination fraud (entirely outside corpus)",
        category="out_of_scope", difficulty="easy", is_answerable=False,
        relevant_chunks={}, relevant_documents=[],
    ),

    # -----------------------------------------------------------------------
    # INVALID  (INV-01 .. INV-02)
    # -----------------------------------------------------------------------
    EvalQueryV2(  # INV-01 | easy
        query="asdfghjkl qwerty 12345 nghi dinh su pham ???",
        description="Invalid: keyboard noise + random legal keywords -- nonsensical query",
        category="invalid", difficulty="easy", is_answerable=False,
        relevant_chunks={}, relevant_documents=[],
        notes="Intentional noise query with no coherent information need.",
    ),
    EvalQueryV2(  # INV-02 | easy
        query="Giá vàng hôm nay là bao nhiêu? Tỷ giá đô la Mỹ?",
        description="Invalid: commodity/forex price query -- completely unrelated to legal domain",
        category="invalid", difficulty="easy", is_answerable=False,
        relevant_chunks={}, relevant_documents=[],
        notes="Real-world consumer question with zero overlap with the legal corpus.",
    ),
]


# ===========================================================================
# Dataset validation
# ===========================================================================

def validate_dataset(queries: list[EvalQueryV2] | None = None) -> None:
    """Validate all queries for internal consistency.

    Raises ValueError on the first error found. Prints a summary on success.
    """
    from collections import Counter

    if queries is None:
        queries = EVAL_QUERIES_V2

    cats  = Counter(q.category   for q in queries)
    diffs = Counter(q.difficulty for q in queries)

    answerable   = [q for q in queries if q.is_answerable]
    unanswerable = [q for q in queries if not q.is_answerable]

    for q in unanswerable:
        if q.relevant_chunks:
            raise ValueError(f"Unanswerable query has relevant_chunks: {q.query[:60]!r}")
    for q in answerable:
        if not q.relevant_chunks:
            raise ValueError(f"Answerable query has empty relevant_chunks: {q.query[:60]!r}")
        for grade in q.relevant_chunks.values():
            if grade not in (1, 2, 3):
                raise ValueError(f"Invalid grade {grade} in: {q.query[:60]!r}")

    print(f"Dataset validation PASSED -- {len(queries)} queries.")
    print(f"  Category  : {dict(cats)}")
    print(f"  Difficulty: {dict(diffs)}")
    print(f"  Answerable: {len(answerable)} | Unanswerable: {len(unanswerable)}")


if __name__ == "__main__":
    validate_dataset()
