"""
Slang MCP Server.
Exposes a tool 'lookup_slang' to decode drug slang terms.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SlangServer")

# Từ điển từ lóng về ma túy thường gặp
SLANG_DICT = {
    "khay": "Ketamine (chất ma túy nhóm hướng thần)",
    "kẹo": "Thuốc lắc / MDMA (chất ma túy cực độc tổng hợp)",
    "nước vui": "Ma túy nước vui (hỗn hợp ma túy tổng hợp dạng lỏng)",
    "đá": "Ma túy đá / Methamphetamine (chất ma túy kích thích)",
    "cỏ": "Cỏ Mỹ / Cần sa tổng hợp",
    "ke": "Ketamine",
    "bay lắc": "Sử dụng trái phép chất ma túy tập thể kèm âm nhạc mạnh",
}

@mcp.tool()
def lookup_slang(query: str) -> str:
    """Tra cứu các từ lóng liên quan đến ma túy trong câu truy vấn và trả về giải nghĩa chuẩn pháp lý.

    Args:
        query: Câu hỏi hoặc từ khóa chứa từ lóng cần dịch.
    """
    query_lower = query.lower()
    found = []
    for slang, definition in SLANG_DICT.items():
        # Kiểm tra sự xuất hiện dưới dạng từ đơn lẻ hoặc cụm từ
        if slang in query_lower:
            found.append(f" - Từ lóng '{slang}' nghĩa là: {definition}")

    if not found:
        return "Không phát hiện từ lóng ma túy quen thuộc trong câu hỏi."

    return "Phát hiện từ lóng ma túy:\n" + "\n".join(found)

if __name__ == "__main__":
    mcp.run()
