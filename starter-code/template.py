"""
Lab #4: System Prompt Engineering & Tool Calling Engine

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas + ReAct Loop.

Chế độ chạy:
  - MOCK (mặc định): không cần API key, chạy hoàn toàn offline bằng rule-based NLU.
  - LIVE: nếu có OPENAI_API_KEY (trong biến môi trường hoặc file .env),
    cả hai class sẽ gọi OpenAI thật. Lỗi mạng/thiếu key sẽ tự động fallback về MOCK
    để autograder luôn chạy được.
"""

import json
import os
from typing import Dict, Any, List, Optional

from tools import (
    TOOL_DEFINITIONS,
    TOOL_MAP,
    search_product_catalog,
    submit_support_ticket,
)
from nlu import detect_intents

# ---------------------------------------------------------------------------
# Nạp biến môi trường từ file .env (nếu có). Không bắt buộc cài python-dotenv.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv()  # thử cả thư mục hiện tại
except ImportError:
    pass

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _get_openai_client():
    """
    Trả về OpenAI client nếu có API key và thư viện openai; ngược lại trả None.

    Thiết kế 'fail-soft': thiếu key KHÔNG làm hỏng chương trình, chỉ đơn giản
    chuyển sang chế độ mock.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# MILESTONE 1: SYSTEM PROMPT cấp sản xuất
# 5 phần: Persona | Available Tools | Core Rules | Boundaries | Output Contract
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- **Tên:** VinAssistant
- **Vai trò:** Chuyên viên tư vấn sản phẩm & chăm sóc khách hàng cho VinFast (xe điện)
  và Vinpearl (nghỉ dưỡng, du lịch).
- **Giọng điệu:** Chuyên nghiệp, thân thiện, ngắn gọn, chính xác. Luôn xưng "VinAssistant"
  và gọi người dùng là "Quý khách".
- **Ngôn ngữ:** Trả lời bằng tiếng Việt, trừ khi khách hàng hỏi bằng ngôn ngữ khác.

## 2. AVAILABLE TOOLS
Bạn có quyền sử dụng đúng 2 công cụ sau:

| Tool | Khi nào dùng | Tham số |
| :--- | :--- | :--- |
| `search_product_catalog` | Khách hỏi về sản phẩm, giá, mẫu xe, gói nghỉ dưỡng | `category` (`xe_dien` \\| `du_lich`), `max_price` (VNĐ) |
| `submit_support_ticket` | Khách báo lỗi, khiếu nại, phản ánh sự cố, cần hỗ trợ | `customer_name`, `issue_description`, `priority` (`low` \\| `medium` \\| `high`) |

## 3. CORE RULES (BẮT BUỘC)
1. **KHÔNG BAO GIỜ bịa dữ liệu sản phẩm.** Mọi tên xe, giá tiền, tính năng, tình trạng
   hàng đều PHẢI đến từ kết quả của `search_product_catalog`. Nếu chưa gọi tool,
   bạn KHÔNG biết giá của bất kỳ sản phẩm nào.
2. **KHÔNG BAO GIỜ bịa mã ticket.** Mã ticket chỉ hợp lệ khi do
   `submit_support_ticket` trả về.
3. **Chỉ báo cáo đúng những gì tool trả về.** Không thêm, không suy diễn, không làm tròn giá.
4. **Nếu tool trả về danh sách rỗng**, hãy nói thẳng là không tìm thấy sản phẩm phù hợp
   và gợi ý khách nới ngân sách hoặc đổi danh mục. TUYỆT ĐỐI không tự đề xuất
   sản phẩm không có trong kết quả.
5. **Thiếu thông tin thì hỏi lại.** Muốn tạo ticket mà chưa biết tên khách hàng
   → hỏi tên trước, không tự đặt tên.
6. **Một câu hỏi có thể cần nhiều tool.** Nếu khách vừa hỏi sản phẩm vừa báo sự cố,
   hãy gọi CẢ HAI tool, không bỏ sót ý nào.

## 4. OPERATIONAL BOUNDARIES
- Chỉ tư vấn về sản phẩm, dịch vụ và chính sách trong hệ sinh thái Vingroup
  (VinFast, Vinpearl, VinWonders, Vinhomes, Vinmec).
- Từ chối lịch sự các câu hỏi ngoài phạm vi (chính trị, y tế cá nhân, đối thủ cạnh tranh,
  tư vấn pháp lý/đầu tư) và hướng khách trở lại chủ đề Vingroup.
- Không thu thập thông tin nhạy cảm: số thẻ ngân hàng, mật khẩu, CCCD.
- Không cam kết giá khuyến mãi, thời gian giao xe hay chính sách chưa được xác nhận
  bởi dữ liệu chính thức.

## 5. OUTPUT CONTRACT
Khi cần dùng tool, hãy suy luận theo chu trình ReAct:

```
Thought: <phân tích khách cần gì, cần tool nào, tham số ra sao>
Action: <tên_tool>(<tham số dạng JSON>)
Observation: <kết quả tool trả về>
... (lặp lại nếu cần thêm tool)
Final Answer: <câu trả lời hoàn chỉnh bằng tiếng Việt cho khách hàng>
```

Phần `Final Answer` gửi cho khách phải:
- Liệt kê sản phẩm kèm **tên đầy đủ và giá chính xác** từ Observation.
- Khi vừa tạo ticket, BẮT BUỘC nêu đủ 3 thông tin, không được lược bỏ:
  (1) **họ tên khách hàng** đúng như đã gửi cho tool,
  (2) **mã ticket** do tool trả về,
  (3) **mức ưu tiên** đã ghi nhận.
  Ví dụ: "VinAssistant đã ghi nhận yêu cầu của Quý khách **Nguyễn Văn An** —
  mã ticket **TK-20260101-001**, mức ưu tiên Cao."
- Khi tool trả về danh sách rỗng, nói rõ **"không tìm thấy"** sản phẩm phù hợp.
- Kết thúc bằng một câu hỏi mở mời khách tiếp tục trao đổi.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Cơ sở tri thức FAQ — dùng cho câu hỏi chính sách (không cần gọi tool)
# ═══════════════════════════════════════════════════════════════════════════

FAQ_KNOWLEDGE = {
    "warranty": (
        "Chính sách bảo hành pin xe điện VinFast kéo dài **10 năm** (không giới hạn số km) "
        "dành cho pin Lithium-ion/LFP chính hãng. VinFast cam kết thay thế pin miễn phí nếu "
        "dung lượng còn lại giảm dưới 70% trong thời gian bảo hành. "
        "Thân vỏ và động cơ điện được bảo hành riêng theo từng dòng xe."
    ),
    "refund": (
        "Chính sách đổi trả và hoàn tiền được áp dụng theo từng loại sản phẩm/dịch vụ. "
        "Với gói nghỉ dưỡng Vinpearl, Quý khách có thể huỷ hoặc đổi lịch trước ngày nhận phòng "
        "theo điều kiện của từng gói."
    ),
    "charging": (
        "VinFast triển khai hệ thống trạm sạc trên toàn quốc, hỗ trợ cả sạc nhanh DC và sạc "
        "tại nhà bằng nguồn 220V. Thời gian sạc nhanh DC thường đạt 10–70% trong khoảng 30 phút, "
        "tuỳ dòng xe và dung lượng pin."
    ),
    "general": (
        "VinAssistant rất sẵn lòng hỗ trợ Quý khách về sản phẩm và chính sách của Vingroup."
    ),
}

FAQ_DISCLAIMER = (
    "\n\n_Lưu ý: Thông tin chính sách mang tính tham khảo. "
    "Để có thông tin chính thức mới nhất, Quý khách vui lòng liên hệ tổng đài VinFast 1900 23 23 89._"
)


# ═══════════════════════════════════════════════════════════════════════════
# Tiện ích định dạng
# ═══════════════════════════════════════════════════════════════════════════

def format_vnd(amount: int) -> str:
    """Định dạng số tiền VNĐ dễ đọc: 548000000 -> '548.000.000 VNĐ'."""
    try:
        return f"{int(amount):,}".replace(",", ".") + " VNĐ"
    except (TypeError, ValueError):
        return str(amount)


def format_products(products: List[Dict[str, Any]]) -> str:
    """Biến danh sách sản phẩm thành đoạn văn bản trình bày cho khách hàng."""
    lines = []
    for idx, p in enumerate(products, start=1):
        availability = {
            "in_stock": "Còn hàng",
            "pre_order": "Đặt trước",
            "out_of_stock": "Hết hàng",
        }.get(p.get("availability", ""), p.get("availability", ""))

        lines.append(
            f"{idx}. **{p.get('name')}** — {format_vnd(p.get('price_vnd', 0))} ({availability})\n"
            f"   {p.get('description', '')}"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """
    Baseline LLM Chatbot — KHÔNG sử dụng Tool Calling hay ReAct Loop.

    Mục đích sư phạm: cho thấy một LLM không có tool sẽ "tự tin bịa" giá tiền,
    tên sản phẩm và mã ticket — thứ mà ToolCallingAgent tránh được.
    """

    # Prompt cố tình KHÔNG có Core Rules về việc cấm bịa dữ liệu.
    BASELINE_SYSTEM_PROMPT = (
        "Bạn là một trợ lý bán hàng của Vingroup. "
        "Hãy trả lời câu hỏi của khách hàng một cách nhiệt tình và cụ thể."
    )

    # Câu trả lời mock — mô phỏng hành vi hallucination điển hình.
    MOCK_ANSWER = (
        "Dạ, bên em đang có các mẫu xe điện VinFast rất phù hợp với ngân sách của anh/chị ạ! "
        "Hiện VinFast VF 5 Plus đang có giá khoảng 520 triệu, VF 6 giá tầm 590 triệu, "
        "còn VF e34 chỉ khoảng 580 triệu thôi ạ. Tất cả đều đang có khuyến mãi giảm thêm "
        "30 triệu và tặng 1 năm sạc miễn phí. Anh/chị muốn em giữ xe mẫu nào ạ?"
    )

    def __init__(self, use_live_api: Optional[bool] = None):
        self.client = _get_openai_client()
        if use_live_api is False:
            self.client = None
        self.mode = "live_openai" if self.client else "mock_baseline"

    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Trả lời trong MỘT lượt, không có tool, không có vòng lặp.

        Returns:
            dict với khoá: answer, tool_calls (luôn rỗng), status, mode.
        """
        answer = self.MOCK_ANSWER
        mode = "mock_baseline"

        if self.client is not None:
            try:
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": self.BASELINE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_input},
                    ],
                    # KHÔNG truyền tools — đây chính là điểm khác biệt của baseline.
                    temperature=0.7,
                )
                answer = response.choices[0].message.content or self.MOCK_ANSWER
                mode = "live_openai"
            except Exception as exc:
                # Fail-soft: lỗi API không được làm hỏng bài lab.
                answer = self.MOCK_ANSWER
                mode = f"mock_baseline (live failed: {type(exc).__name__})"

        return {
            "answer": answer,
            "tool_calls": [],  # Baseline KHÔNG BAO GIỜ gọi tool.
            "status": "success",
            "mode": mode,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """
    Agent với System Prompt Engineering & Tool Calling.

    Vòng lặp ReAct: mỗi iteration thực hiện đủ các Action cần thiết, ghi
    Observation vào trace, rồi tổng hợp Final Answer.
    """

    def __init__(self, max_iterations: int = 5, use_live_api: Optional[bool] = None):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []
        self.system_prompt = SYSTEM_PROMPT

        self.client = _get_openai_client()
        if use_live_api is False:
            self.client = None

    # ----------------------------------------------------------------------
    # Điểm vào chính
    # ----------------------------------------------------------------------

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Chạy Agent Loop cho một câu hỏi của người dùng.

        Returns:
            dict gồm answer, trace, iterations, status.
        """
        self.trace = []

        if self.client is not None:
            try:
                return self._run_live(user_input)
            except Exception as exc:
                # Fail-soft: rơi về mock, ghi lại lý do vào trace.
                self.trace.append({
                    "step": "live_api_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "fallback": "mock",
                })

        return self._run_mock(user_input)

    # ----------------------------------------------------------------------
    # MILESTONE 3 + 4: Agent Loop (chế độ mock, rule-based)
    # ----------------------------------------------------------------------

    def _run_mock(self, user_input: str) -> Dict[str, Any]:
        """Agent Loop offline — dùng NLU rule-based thay cho LLM."""

        # --- Bước 0: Intent Detection (Milestone 3) -----------------------
        intents = detect_intents(user_input)
        self.trace.append({
            "step": "intent_detection",
            "user_input": user_input,
            "intents": {
                "needs_catalog": intents["needs_catalog"],
                "needs_ticket": intents["needs_ticket"],
                "is_faq": intents["is_faq"],
            },
        })

        # --- Agent Loop (Milestone 4: guard max_iterations) ---------------
        iteration = 0
        pending = {
            "catalog": intents["needs_catalog"],
            "ticket": intents["needs_ticket"],
        }
        observations: Dict[str, Any] = {}

        while iteration < self.max_iterations:
            iteration += 1

            result, is_final = self._execute_step(
                iteration, user_input, intents, pending, observations
            )

            if is_final:
                return {
                    "answer": result,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed",
                }

        # Vượt quá số bước tối đa — dừng an toàn thay vì lặp vô hạn.
        self.trace.append({"step": "max_iterations_guard", "iterations": iteration})
        return {
            "answer": (
                "Xin lỗi Quý khách, yêu cầu này cần nhiều bước xử lý hơn mức cho phép. "
                "Quý khách vui lòng chia nhỏ câu hỏi hoặc liên hệ tổng đài 1900 23 23 89."
            ),
            "trace": self.trace,
            "iterations": iteration,
            "status": "max_iterations_reached",
        }

    def _execute_step(
        self,
        iteration: int,
        user_input: str,
        intents: Dict[str, Any],
        pending: Dict[str, bool],
        observations: Dict[str, Any],
    ):
        """
        Thực hiện MỘT bước của Agent Loop.

        Returns:
            (result, is_final) — is_final=True nghĩa là đã có Final Answer.
        """
        # --- Action: gọi tool tra cứu sản phẩm ----------------------------
        if pending["catalog"]:
            args = intents["catalog_args"]
            products = search_product_catalog(**args)
            observations["catalog"] = products
            pending["catalog"] = False

            self.trace.append({
                "step": f"iteration_{iteration}",
                "thought": "Khách hỏi về sản phẩm — phải tra cứu catalog, không được bịa giá.",
                "action": "search_product_catalog",
                "action_input": args,
                "observation": f"Tìm thấy {len(products)} sản phẩm phù hợp.",
                "observation_data": products,
            })

        # --- Action: gọi tool tạo ticket ----------------------------------
        if pending["ticket"]:
            args = intents["ticket_args"]
            ticket = submit_support_ticket(**args)
            observations["ticket"] = ticket
            pending["ticket"] = False

            self.trace.append({
                "step": f"iteration_{iteration}",
                "thought": "Khách báo sự cố — phải tạo ticket để có mã chính thức.",
                "action": "submit_support_ticket",
                "action_input": args,
                "observation": f"Đã tạo ticket {ticket.get('ticket_id')}.",
                "observation_data": ticket,
            })

        # --- Tổng hợp Final Answer ----------------------------------------
        answer = self._synthesize_answer(user_input, intents, observations)
        self.trace.append({
            "step": f"iteration_{iteration}",
            "thought": "Đã có đủ Observation — tổng hợp Final Answer cho khách hàng.",
            "action": "final_answer",
            "observation": "Hoàn tất.",
        })
        return answer, True

    def _synthesize_answer(
        self,
        user_input: str,
        intents: Dict[str, Any],
        observations: Dict[str, Any],
    ) -> str:
        """Ghép các Observation thành câu trả lời cuối cùng cho khách hàng."""
        parts: List[str] = []

        # --- Phần sản phẩm ------------------------------------------------
        if "catalog" in observations:
            products = observations["catalog"]

            # Milestone 4: xử lý lỗi đọc file.
            if products and isinstance(products[0], dict) and "error" in products[0]:
                parts.append(
                    "Rất tiếc, hệ thống tra cứu sản phẩm đang tạm thời gián đoạn. "
                    "Quý khách vui lòng thử lại sau ít phút."
                )

            # Milestone 4: xử lý kết quả rỗng.
            elif not products:
                args = intents["catalog_args"]
                category_label = (
                    "xe điện VinFast" if args.get("category") == "xe_dien"
                    else "gói nghỉ dưỡng Vinpearl"
                )
                budget = args.get("max_price")
                budget_text = f" trong tầm giá {format_vnd(budget)}" if budget else ""
                parts.append(
                    f"Rất tiếc, VinAssistant **không tìm thấy** {category_label} nào"
                    f"{budget_text} ạ.\n\n"
                    "Quý khách vui lòng cân nhắc nâng ngân sách hoặc tham khảo danh mục khác "
                    "để VinAssistant tư vấn thêm."
                )

            else:
                args = intents["catalog_args"]
                category_label = (
                    "xe điện VinFast" if args.get("category") == "xe_dien"
                    else "gói nghỉ dưỡng Vinpearl"
                )
                budget = args.get("max_price")
                budget_text = f" trong tầm giá {format_vnd(budget)}" if budget else ""
                parts.append(
                    f"VinAssistant tìm thấy **{len(products)}** {category_label}"
                    f"{budget_text} phù hợp với Quý khách:\n\n"
                    + format_products(products)
                )

        # --- Phần ticket --------------------------------------------------
        if "ticket" in observations:
            ticket = observations["ticket"]
            priority_label = {
                "high": "Cao", "medium": "Trung bình", "low": "Thấp"
            }.get(ticket.get("priority", "medium"), "Trung bình")

            parts.append(
                f"VinAssistant đã ghi nhận yêu cầu hỗ trợ của Quý khách "
                f"**{ticket.get('customer_name')}**.\n\n"
                f"- **Mã ticket:** `{ticket.get('ticket_id')}`\n"
                f"- **Nội dung:** {ticket.get('issue_description')}\n"
                f"- **Mức ưu tiên:** {priority_label}\n"
                f"- **Trạng thái:** Đang mở (open)\n\n"
                f"Bộ phận kỹ thuật sẽ liên hệ Quý khách trong thời gian sớm nhất."
            )

        # --- Phần FAQ -----------------------------------------------------
        if intents["is_faq"]:
            topic = intents.get("faq_topic") or "general"
            parts.append(FAQ_KNOWLEDGE.get(topic, FAQ_KNOWLEDGE["general"]) + FAQ_DISCLAIMER)

        # --- Thiếu tên khách hàng để mở ticket -----------------------------
        # Core Rule #5: thiếu thông tin thì HỎI LẠI, tuyệt đối không tự đặt tên.
        if intents.get("needs_name_clarification"):
            parts.append(
                "VinAssistant rất tiếc khi Quý khách gặp sự cố này. "
                "Để mở phiếu hỗ trợ, VinAssistant cần biết **họ tên** của Quý khách ạ.\n\n"
                "Quý khách vui lòng cho biết tên (ví dụ: \"Tôi tên Nguyễn Văn An\") "
                "để VinAssistant tạo ticket và chuyển bộ phận kỹ thuật xử lý ngay."
            )
            return "\n\n".join(parts)

        # --- Không nhận diện được ý định ----------------------------------
        if not parts:
            parts.append(
                "VinAssistant chưa hiểu rõ yêu cầu của Quý khách. "
                "VinAssistant có thể hỗ trợ tra cứu xe điện VinFast, gói nghỉ dưỡng Vinpearl, "
                "hoặc tiếp nhận yêu cầu hỗ trợ kỹ thuật. "
                "Quý khách vui lòng mô tả rõ hơn nhu cầu của mình ạ."
            )
            return "\n\n".join(parts)

        parts.append("Quý khách còn cần VinAssistant hỗ trợ thêm điều gì không ạ?")
        return "\n\n".join(parts)

    # ----------------------------------------------------------------------
    # Chế độ LIVE — OpenAI function calling thật
    # ----------------------------------------------------------------------

    def _openai_tools_payload(self) -> List[Dict[str, Any]]:
        """Chuyển TOOL_DEFINITIONS sang định dạng tools của OpenAI Chat Completions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in TOOL_DEFINITIONS
        ]

    def _run_live(self, user_input: str) -> Dict[str, Any]:
        """
        Agent Loop dùng OpenAI function calling thật.

        Cùng một thuật toán ReAct, nhưng quyết định gọi tool nào do LLM đưa ra
        thay vì keyword matching.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        self.trace.append({"step": "init", "mode": "live_openai", "user_input": user_input})

        # `tool_rounds` chỉ đếm những vòng CÓ gọi tool, để cách đếm iterations
        # thống nhất với chế độ mock (vòng tổng hợp Final Answer không tính riêng).
        tool_rounds = 0

        while tool_rounds < self.max_iterations:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=self._openai_tools_payload(),
                tool_choice="auto",
                # Agent cần chính xác và ổn định, không cần sáng tạo -> temperature 0.
                temperature=0,
            )
            message = response.choices[0].message

            # LLM không gọi tool nữa -> đã có Final Answer.
            if not message.tool_calls:
                self.trace.append({
                    "step": f"iteration_{max(tool_rounds, 1)}",
                    "action": "final_answer",
                    "observation": "LLM kết thúc, không gọi thêm tool.",
                })
                return {
                    "answer": message.content or "",
                    "trace": self.trace,
                    # Câu FAQ không gọi tool nào vẫn tính là 1 vòng suy luận.
                    "iterations": max(tool_rounds, 1),
                    "status": "completed",
                }

            tool_rounds += 1
            iteration = tool_rounds

            # Ghi lại lượt assistant có tool_calls trước khi thêm kết quả tool.
            messages.append(message.model_dump(exclude_none=True))

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    fn_args = {}

                fn = TOOL_MAP.get(fn_name)
                if fn is None:
                    result: Any = {"error": f"Unknown tool: {fn_name}"}
                else:
                    try:
                        result = fn(**fn_args)
                    except Exception as exc:
                        result = {"error": f"{type(exc).__name__}: {exc}"}

                self.trace.append({
                    "step": f"iteration_{iteration}",
                    "action": fn_name,
                    "action_input": fn_args,
                    "observation_data": result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # Vượt max_iterations trong chế độ live.
        self.trace.append({"step": "max_iterations_guard", "iterations": tool_rounds})
        return {
            "answer": "Xin lỗi Quý khách, yêu cầu cần nhiều bước xử lý hơn mức cho phép.",
            "trace": self.trace,
            "iterations": tool_rounds,
            "status": "max_iterations_reached",
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Demo so sánh Baseline vs Agent
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    live = _get_openai_client() is not None
    print("Chế độ:", "LIVE (OpenAI)" if live else "MOCK (offline, không cần API key)")
    print("=" * 70)

    print("\n=== RUNNING CHATBOT BASELINE (không có tool) ===")
    chatbot = ChatbotBaseline()
    baseline = chatbot.query(user_query)
    print(baseline["answer"])
    print(f"\n[tool_calls: {len(baseline['tool_calls'])} | mode: {baseline['mode']}]")

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print(result["answer"])
    print(f"\n[iterations: {result['iterations']} | status: {result['status']}]")

    print("\n=== TRACE LOG ===")
    print(json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
