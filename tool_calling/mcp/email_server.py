from mcp.server.fastmcp import FastMCP
import resend
import os

mcp = FastMCP("Email MCP Server")
resend.api_key = os.environ["RESEND_API_KEY"]

@mcp.tool()
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email to a given address with a subject and body."""
    try:
        result = resend.Emails.send({
            "from": os.environ["EMAIL_FROM"],
            "to": to,
            "subject": subject,
            "html": body,
        })
        return {"status": "sent", "id": result.get("id")}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    mcp.run(transport="stdio")