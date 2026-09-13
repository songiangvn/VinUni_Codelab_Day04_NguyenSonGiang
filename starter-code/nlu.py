"""
nlu.py — Lớp hiểu ngôn ngữ tự nhiên (rule-based) cho VinAssistant.

Tách riêng khỏi template.py để Agent Loop chỉ tập trung vào điều phối tool,
còn việc "đọc hiểu" câu tiếng Việt nằm gọn ở đây.

Gồm 4 nhóm hàm:
  1. parse_vnd_amount()       — "600 triệu" -> 600_000_000
  2. extract_customer_name()  — "Tôi tên Lê Minh Khoa, ..." -> "Lê Minh Khoa"
  3. extract_issue_description() / detect_priority()
  4. detect_intents()         — xác định cần gọi tool nào
"""

import re
from typing import Dict, Any, Optional

# ═══════════════════════════════════════════════════════════════════════════
# 1. PARSE TIỀN TỆ TIẾNG VIỆT
# ═══════════════════════════════════════════════════════════════════════════

# Hệ số nhân cho từng đơn vị.
_MONEY_UNITS = {
    "tỷ": 1_000_000_000,
    "tỉ": 1_000_000_000,
    "ty": 1_000_000_000,
    "triệu": 1_000_000,
    "trieu": 1_000_000,
    "tr": 1_000_000,
    "nghìn": 1_000,
    "nghin": 1_000,
    "ngàn": 1_000,
    "ngan": 1_000,
    "k": 1_000,
}

# Bắt: số (có thể có , hoặc . phân cách, có thể thập phân) + đơn vị.
# Cụm dài đặt trước cụm ngắn để "triệu" không bị "tr" nuốt mất.
_MONEY_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)*)\s*(tỷ|tỉ|triệu|trieu|nghìn|nghin|ngàn|ngan|tr|ty|k)\b",
    re.IGNORECASE,
)

# Bắt số trần kèm "đồng"/"vnd"/"vnđ": "500000000 đồng"
_PLAIN_VND_PATTERN = re.compile(
    r"(\d[\d.,]{5,})\s*(?:đồng|dong|vnđ|vnd|đ)\b",
    re.IGNORECASE,
)


def _to_number(raw: str) -> float:
    """Chuyển '1.5' / '1,5' / '600' / '1.129' thành số thực.

    Quy ước: dấu phân cách cuối cùng nếu theo sau đúng 1-2 chữ số thì coi là
    dấu thập phân, ngược lại coi là phân cách hàng nghìn.
    """
    raw = raw.strip()
    match = re.search(r"[.,](\d{1,2})$", raw)
    if match:
        integer_part = re.sub(r"[.,]", "", raw[: match.start()])
        return float(integer_part + "." + match.group(1))
    return float(re.sub(r"[.,]", "", raw))


def parse_vnd_amount(text: str) -> Optional[int]:
    """
    Trích xuất số tiền (VNĐ) đầu tiên xuất hiện trong câu.

    Ví dụ:
        "xe dưới 600 triệu"   -> 600000000
        "resort dưới 6 triệu" -> 6000000
        "tầm 1.5 tỷ"          -> 1500000000
        "khoảng 900k"         -> 900000
        "dưới 500000000 đồng" -> 500000000

    Returns:
        Số tiền dạng int, hoặc None nếu không tìm thấy.
    """
    if not text:
        return None

    match = _MONEY_PATTERN.search(text)
    if match:
        number = _to_number(match.group(1))
        multiplier = _MONEY_UNITS.get(match.group(2).lower(), 1)
        return int(number * multiplier)

    match = _PLAIN_VND_PATTERN.search(text)
    if match:
        return int(_to_number(match.group(1)))

    return None


# ═══════════════════════════════════════════════════════════════════════════
# 2. TRÍCH XUẤT TÊN KHÁCH HÀNG
# ═══════════════════════════════════════════════════════════════════════════

# Ký tự hợp lệ trong tên tiếng Việt có dấu.
_VN_UPPER = "A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-ỹ"
_VN_LOWER = "a-zàáâãèéêìíòóôõùúăđĩũơưạ-ỹ"

# Một "từ tên": chữ cái đầu viết hoa + phần còn lại viết thường.
_NAME_WORD = "[" + _VN_UPPER + "][" + _VN_LOWER + "]+"

# Các mẫu giới thiệu tên thường gặp, sắp theo độ cụ thể giảm dần.
_NAME_PATTERNS = [
    r"(?:tôi|tui|mình|em|anh|chị)\s+tên\s+(?:là\s+)?(" + _NAME_WORD + r"(?:\s+" + _NAME_WORD + r"){0,3})",
    r"tên\s+(?:tôi|tui|mình|em|anh|chị)\s+là\s+(" + _NAME_WORD + r"(?:\s+" + _NAME_WORD + r"){0,3})",
    r"(?:tôi|tui|mình|em)\s+là\s+(" + _NAME_WORD + r"(?:\s+" + _NAME_WORD + r"){0,3})",
    r"(?:họ\s+(?:và\s+)?tên|khách\s+hàng)\s*:?\s*(" + _NAME_WORD + r"(?:\s+" + _NAME_WORD + r"){0,3})",
    r"tên\s+(?:là\s+)?(" + _NAME_WORD + r"(?:\s+" + _NAME_WORD + r"){1,3})",
]

# Những từ viết hoa KHÔNG phải tên người — tránh bắt nhầm thương hiệu/địa danh.
_NAME_STOPWORDS = {
    "vinfast", "vinpearl", "vingroup", "vinhomes", "vinwonders", "vinmec",
    "nha", "trang", "phú", "quốc", "hà", "nội", "sài", "gòn", "đà", "nẵng",
    "landmark", "safari", "adas", "resort", "spa", "discovery", "luxury",
    "xe", "phòng", "pin",
}


def extract_customer_name(text: str) -> Optional[str]:
    """
    Trích xuất tên khách hàng từ câu giới thiệu tự do.

    Ví dụ:
        "Tôi tên Lê Minh Khoa, xe VF 8 bị lỗi"     -> "Lê Minh Khoa"
        "tên tôi là Phạm Thị Dung, phòng bị mốc"   -> "Phạm Thị Dung"
        "Mình là Nguyễn An"                        -> "Nguyễn An"

    Returns:
        Tên đã chuẩn hoá khoảng trắng, hoặc None nếu không nhận ra.
    """
    if not text:
        return None

    for pattern in _NAME_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue

        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" ,.")

        # Loại bỏ các từ đuôi không thuộc tên (thương hiệu, địa danh).
        words = candidate.split()
        while words and words[-1].lower() in _NAME_STOPWORDS:
            words.pop()

        if words:
            return " ".join(words)

    return None


# ═══════════════════════════════════════════════════════════════════════════
# 3. MÔ TẢ VẤN ĐỀ & MỨC ƯU TIÊN
# ═══════════════════════════════════════════════════════════════════════════

_HIGH_PRIORITY_KEYWORDS = [
    "gấp", "khẩn", "nghiêm trọng", "nguy hiểm", "ngay lập tức",
    "không sử dụng được", "không dùng được", "hỏng nặng", "mất an toàn",
    "urgent", "critical",
]

_MEDIUM_PRIORITY_KEYWORDS = [
    "mức độ trung bình", "trung bình", "bình thường", "medium",
]

_LOW_PRIORITY_KEYWORDS = [
    "góp ý", "gop y", "đề xuất", "khi nào cũng được", "không gấp",
    "không vội", "low",
]


def detect_priority(text: str) -> str:
    """Suy ra mức ưu tiên ticket từ ngôn ngữ khách hàng dùng."""
    lowered = (text or "").lower()

    if any(kw in lowered for kw in _HIGH_PRIORITY_KEYWORDS):
        return "high"
    if any(kw in lowered for kw in _MEDIUM_PRIORITY_KEYWORDS):
        return "medium"
    if any(kw in lowered for kw in _LOW_PRIORITY_KEYWORDS):
        return "low"
    return "medium"


# Các mệnh đề cần cắt bỏ khi tóm tắt vấn đề.
_ISSUE_NOISE_PATTERNS = [
    r"(?:tôi|tui|mình|em|anh|chị)\s+tên\s+(?:là\s+)?" + _NAME_WORD + r"(?:\s+" + _NAME_WORD + r"){0,3}",
    r"tên\s+(?:tôi|tui|mình|em|anh|chị)\s+là\s+" + _NAME_WORD + r"(?:\s+" + _NAME_WORD + r"){0,3}",
    r"đây\s+là\s+vấn\s+đề\s+[^,.]*",
    r"cần\s+xử\s+lý\s+(?:gấp|ngay|sớm)[^,.]*",
    r"mức\s+độ\s+[^,.]*",
    r"(?:tôi|mình|em)\s+(?:cũng\s+)?muốn\s+ghi\s+nhận\s+(?:phản\s+hồi|góp\s+ý)\s*:?",
    r"^(?:và|với|ngoài\s+ra)\s+",
]


def extract_issue_description(text: str) -> str:
    """
    Rút gọn câu người dùng thành mô tả vấn đề sạch, bỏ phần giới thiệu tên
    và phần nêu mức độ ưu tiên.

    Khi câu vừa hỏi sản phẩm vừa báo sự cố (TC03), chỉ giữ mệnh đề chứa
    từ khoá sự cố — tránh lấy nhầm phần hỏi catalog.

    Ví dụ:
        "Tôi tên Lê Minh Khoa, xe VF 8 của tôi bị lỗi hệ thống ADAS.
         Đây là vấn đề nghiêm trọng, cần xử lý gấp."
        -> "xe VF 8 của tôi bị lỗi hệ thống ADAS"
    """
    if not text:
        return ""

    cleaned = text
    for pattern in _ISSUE_NOISE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:")

    # Tách thành các mệnh đề rồi dọn từ nối thừa ở đầu mỗi mệnh đề.
    clauses = []
    for raw_clause in re.split(r"[.;]", cleaned):
        clause = re.sub(
            r"^\s*(?:và|với|ngoài\s+ra|còn|thì)\b\s*",
            "",
            raw_clause.strip(" ,.;:"),
            flags=re.IGNORECASE,
        ).strip(" ,.;:")
        if clause:
            clauses.append(clause)

    if not clauses:
        return text.strip()

    # Ưu tiên mệnh đề thực sự nói về sự cố.
    for clause in clauses:
        if any(kw in clause.lower() for kw in _TICKET_KEYWORDS):
            return clause

    return clauses[0]


# ═══════════════════════════════════════════════════════════════════════════
# 4. INTENT DETECTION
# ═══════════════════════════════════════════════════════════════════════════

# Từ khoá cho thấy khách muốn TRA CỨU sản phẩm.
_CATALOG_KEYWORDS = [
    "xem", "tìm", "tra cứu", "tra cuu", "sản phẩm", "báo giá", "bao gia",
    "giá", "bao nhiêu tiền", "tư vấn", "tu van", "mua", "đặt", "gói",
    "danh sách", "danh mục", "nào", "phù hợp", "ngân sách", "tầm giá", "cho tôi",
]

# Từ khoá phân loại danh mục.
_XE_DIEN_KEYWORDS = [
    "xe điện", "xe dien", "ô tô", "oto", "vinfast", "vf ", "vf3", "vf5",
    "vf8", "vf9", "xe hơi", "bán tải", "xe ",
]
_DU_LICH_KEYWORDS = [
    "du lịch", "du lich", "resort", "vinpearl", "nghỉ dưỡng", "nghi duong",
    "khách sạn", "khach san", "tour", "kỳ nghỉ", "villa", "landmark",
    "vinwonders", "safari", "phòng",
]

# Từ khoá cho thấy khách muốn MỞ TICKET hỗ trợ.
_TICKET_KEYWORDS = [
    "bị lỗi", "bi loi", "lỗi", "hỏng", "hư", "sự cố", "su co", "trục trặc",
    "khiếu nại", "phản ánh", "phản hồi", "than phiền", "không hoạt động",
    "ghi nhận", "hỗ trợ", "sửa chữa", "ẩm mốc", "mốc", "báo lỗi", "ticket",
]

# Từ khoá FAQ — câu hỏi chính sách, KHÔNG cần gọi tool.
_FAQ_KEYWORDS = [
    "chính sách", "chinh sach", "quy định", "điều kiện", "bảo hành",
    "bao hanh", "thủ tục", "kéo dài bao lâu", "trong bao lâu",
    "như thế nào", "làm sao để", "hướng dẫn", "có được không",
]


def detect_intents(user_input: str) -> Dict[str, Any]:
    """
    Phân tích ý định người dùng bằng keyword matching.

    Bẫy #3 trong student_guide: needs_catalog và needs_ticket được kiểm tra
    ĐỘC LẬP (if - if), không dùng if-elif, vì một câu có thể cần CẢ HAI tool.

    Returns:
        dict gồm:
          needs_catalog (bool), catalog_args (dict)
          needs_ticket  (bool), ticket_args  (dict)
          is_faq (bool), faq_topic (str|None)
    """
    text = user_input or ""
    lowered = text.lower()

    # ---- Ý định 1: tra cứu catalog -------------------------------------
    has_xe_dien = any(kw in lowered for kw in _XE_DIEN_KEYWORDS)
    has_du_lich = any(kw in lowered for kw in _DU_LICH_KEYWORDS)

    category = None
    if has_xe_dien and has_du_lich:
        # Câu nhắc cả hai -> ưu tiên danh mục xuất hiện TRƯỚC trong câu.
        first_xe = min(lowered.find(k) for k in _XE_DIEN_KEYWORDS if k in lowered)
        first_dl = min(lowered.find(k) for k in _DU_LICH_KEYWORDS if k in lowered)
        category = "xe_dien" if first_xe < first_dl else "du_lich"
    elif has_xe_dien:
        category = "xe_dien"
    elif has_du_lich:
        category = "du_lich"

    has_catalog_verb = any(kw in lowered for kw in _CATALOG_KEYWORDS)
    max_price = parse_vnd_amount(text)

    needs_catalog = bool(category) and (has_catalog_verb or max_price is not None)

    catalog_args: Dict[str, Any] = {}
    if needs_catalog:
        catalog_args["category"] = category
        if max_price is not None:
            catalog_args["max_price"] = max_price

    # ---- Ý định 2: mở ticket hỗ trợ ------------------------------------
    has_ticket_keyword = any(kw in lowered for kw in _TICKET_KEYWORDS)
    customer_name = extract_customer_name(text)

    # Chỉ mở ticket khi có dấu hiệu sự cố VÀ biết tên khách hàng.
    needs_ticket = has_ticket_keyword and customer_name is not None

    ticket_args: Dict[str, Any] = {}
    if needs_ticket:
        ticket_args = {
            "customer_name": customer_name,
            "issue_description": extract_issue_description(text),
            "priority": detect_priority(text),
        }

    # Một câu vừa hỏi sản phẩm vừa báo lỗi thì catalog không được "nuốt"
    # phần khiếu nại — cả hai cờ cùng True (Bẫy #3).

    # ---- Ý định 3: FAQ (không cần tool) --------------------------------
    is_faq = (
        not needs_catalog
        and not needs_ticket
        and any(kw in lowered for kw in _FAQ_KEYWORDS)
    )

    faq_topic = None
    if is_faq:
        if "bảo hành" in lowered or "bao hanh" in lowered:
            faq_topic = "warranty"
        elif "đổi trả" in lowered or "hoàn tiền" in lowered or "hủy" in lowered:
            faq_topic = "refund"
        elif "sạc" in lowered:
            faq_topic = "charging"
        else:
            faq_topic = "general"

    # Khách báo sự cố nhưng chưa cho biết tên -> phải HỎI LẠI, không tự đặt tên
    # (Core Rule #5 trong SYSTEM_PROMPT).
    needs_name_clarification = has_ticket_keyword and customer_name is None

    return {
        "needs_catalog": needs_catalog,
        "catalog_args": catalog_args,
        "needs_ticket": needs_ticket,
        "ticket_args": ticket_args,
        "is_faq": is_faq,
        "faq_topic": faq_topic,
        "needs_name_clarification": needs_name_clarification,
    }
