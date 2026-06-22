from mcp.server.fastmcp import FastMCP
import fitz 
from docx import Document

mcp = FastMCP("Document Reader MCP Server")

@mcp.tool()
def read_pdf(file_path: str) -> str:
    """Extract text content from a PDF file given its path."""
    doc = fitz.open(file_path)
    return "\n".join(page.get_text() for page in doc)

@mcp.tool()
def read_docx(file_path: str) -> str:
    """Extract text content from a Word (.docx) file given its path."""
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)

if __name__ == "__main__":
    mcp.run(transport="stdio")