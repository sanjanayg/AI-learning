import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()


def send_email(recipient: str, subject: str, body: str):
    smtp_host = os.getenv("SMTP_HOST")
    print("WTEYRTUYI")
    smtp_port = int(os.getenv("SMTP_PORT"))
    print("WQERWETr",smtp_port)
    smtp_email = os.getenv("SMTP_EMAIL")
    smtp_password = os.getenv("SMTP_PASSWORD")
    print(f"SMTP_HOST={smtp_host!r}, SMTP_PORT={smtp_port!r}")
    smtp_user = os.getenv("SMTP_USER")
    print(f"Logging in with user={smtp_user!r}, password={smtp_password[:6]!r}...")

    msg = EmailMessage()
    msg["From"] = smtp_email
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_email, smtp_password)
        server.login(smtp_user, smtp_password) 
        server.send_message(msg)

    return {
        "email_sent": True,
        "recipient": recipient
    }

import re


def extract_email(question: str):

    match = re.search(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        question
    )

    if match:
        return match.group(0)

    return None