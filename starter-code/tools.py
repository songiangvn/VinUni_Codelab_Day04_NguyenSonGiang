import json
import os
from typing import List, Dict, Any
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "raw-data")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# Đọc file product_catalog.json, lọc theo category và max_price.
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.

    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.

    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")

    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    try:
        with open(catalog_file, "r", encoding="utf-8") as f:
            products = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return [{"error": f"Cannot read product catalog: {exc}"}]

    # max_price có thể về dưới dạng None/chuỗi khi LLM trả tham số -> chuẩn hoá.
    if max_price is None:
        max_price = 999999999999
    try:
        max_price = int(max_price)
    except (TypeError, ValueError):
        max_price = 999999999999

    category_norm = (category or "").strip().lower()

    results = [
        p for p in products
        if p.get("category", "").lower() == category_norm
        and p.get("price_vnd", 0) <= max_price
    ]
    return results


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# Tạo ticket mới và lưu (append) vào support_tickets.json.
# ---------------------------------------------------------------------------

VALID_PRIORITIES = {"low", "medium", "high"}


def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.

    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.

    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    # Chuẩn hoá priority — giá trị lạ sẽ rơi về 'medium'.
    priority_norm = (priority or "medium").strip().lower()
    if priority_norm not in VALID_PRIORITIES:
        priority_norm = "medium"

    # Bẫy #2: PHẢI load danh sách cũ trước, nếu không sẽ ghi đè toàn bộ file.
    existing_tickets: List[Dict[str, Any]] = []
    if os.path.exists(tickets_file):
        try:
            with open(tickets_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                existing_tickets = loaded
        except (json.JSONDecodeError, OSError):
            existing_tickets = []

    # Sinh ticket_id dạng TK-YYYYMMDD-NNN
    today = datetime.now().strftime("%Y%m%d")
    seq = len(existing_tickets) + 1
    ticket_id = f"TK-{today}-{seq:03d}"

    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority_norm,
        "status": "open",
        "created_at": datetime.now().isoformat() + "+07:00",
        "category": "general"
    }
    existing_tickets.append(new_ticket)

    try:
        with open(tickets_file, "w", encoding="utf-8") as f:
            json.dump(existing_tickets, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        return {"error": f"Cannot save ticket: {exc}", "status": "failed"}

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority_norm,
        "status": "open",
        "created_at": new_ticket["created_at"],
        "message": f"Ticket {ticket_id} đã được tạo thành công."
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": (
            "Tra cứu sản phẩm và dịch vụ trong hệ sinh thái Vingroup theo danh mục "
            "và mức giá tối đa. Dùng tool này khi khách hàng hỏi về xe điện VinFast "
            "hoặc gói nghỉ dưỡng Vinpearl, hỏi giá, hỏi xe/resort nào phù hợp ngân sách. "
            "KHÔNG được tự bịa tên sản phẩm hay giá — luôn gọi tool này để lấy dữ liệu thật."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Danh mục sản phẩm. 'xe_dien' cho ô tô điện VinFast "
                        "(VF 3, VF 5, VF 8, VF 9, VF Wild); "
                        "'du_lich' cho gói nghỉ dưỡng/resort Vinpearl."
                    ),
                    "enum": ["xe_dien", "du_lich"]
                },
                "max_price": {
                    "type": "integer",
                    "description": (
                        "Giá tối đa tính bằng VNĐ (đơn vị đồng, không phải triệu). "
                        "Ví dụ '600 triệu' -> 600000000. Bỏ trống nếu khách không nêu ngân sách."
                    )
                }
            },
            "required": ["category"]
        }
    },
    {
        "name": "submit_support_ticket",
        "description": (
            "Tạo một ticket hỗ trợ khách hàng trong hệ thống CSKH Vingroup. "
            "Dùng tool này khi khách báo lỗi, khiếu nại, phản ánh sự cố, hoặc yêu cầu "
            "được hỗ trợ kỹ thuật/bảo hành. Chỉ gọi khi đã biết tên khách hàng và mô tả vấn đề."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ tên đầy đủ của khách hàng, ví dụ 'Lê Minh Khoa'."
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả ngắn gọn, rõ ràng vấn đề khách hàng đang gặp phải."
                },
                "priority": {
                    "type": "string",
                    "description": (
                        "Mức độ ưu tiên. 'high' khi khách nói gấp/nghiêm trọng/khẩn cấp; "
                        "'low' khi chỉ là góp ý nhẹ; còn lại là 'medium'."
                    ),
                    "enum": ["low", "medium", "high"],
                    "default": "medium"
                }
            },
            "required": ["customer_name", "issue_description"]
        }
    }
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
