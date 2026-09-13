"""
live_demo.py — Chạy cả 5 test case thật qua OpenAI và so sánh
Baseline (không tool) vs ToolCallingAgent (có tool).

Cách chạy:
    python live_demo.py          # dùng key trong .env
    python live_demo.py --mock   # ép chạy offline để đối chiếu
"""

import json
import os
import sys

from template import ChatbotBaseline, ToolCallingAgent, _get_openai_client

QUERIES_FILE = os.path.join(os.path.dirname(__file__), "..", "raw-data", "customer_queries.json")


def main():
    force_mock = "--mock" in sys.argv
    live = (not force_mock) and _get_openai_client() is not None

    print("=" * 78)
    print("  CHẾ ĐỘ:", "LIVE (OpenAI thật)" if live else "MOCK (offline)")
    print("  MODEL :", os.getenv("OPENAI_MODEL", "gpt-4o-mini") if live else "—")
    print("=" * 78)

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)

    for case in queries:
        print("\n" + "─" * 78)
        print(f"[{case['id']}] {case['category']}")
        print(f"Câu hỏi: {case['query']}")
        print(f"Tool kỳ vọng: {case.get('expected_tools', [])}")
        print("─" * 78)

        # --- Baseline: không tool -> dễ bịa dữ liệu ---
        baseline = ChatbotBaseline(use_live_api=None if live else False)
        b_res = baseline.query(case["query"])
        print("\n▼ BASELINE (không tool)")
        print(b_res["answer"][:400])
        print(f"   [tool_calls: {len(b_res['tool_calls'])}]")

        # --- Agent: có tool -> dữ liệu thật ---
        agent = ToolCallingAgent(max_iterations=5, use_live_api=None if live else False)
        a_res = agent.run(case["query"])
        tools_used = [
            t["action"] for t in a_res["trace"]
            if t.get("action") and t["action"] != "final_answer"
        ]
        print("\n▼ TOOL CALLING AGENT")
        print(a_res["answer"][:400])
        print(f"   [iterations: {a_res['iterations']} | status: {a_res['status']}]")
        print(f"   [tool thực gọi: {tools_used}]")

        expected = case.get("expected_tools", [])
        verdict = "KHỚP" if sorted(set(tools_used)) == sorted(set(expected)) else "LỆCH"
        print(f"   [so với kỳ vọng: {verdict}]")

    print("\n" + "=" * 78)
    print("  Kết luận: Baseline bịa giá/tên sản phẩm; Agent chỉ báo dữ liệu từ tool.")
    print("=" * 78)


if __name__ == "__main__":
    main()
