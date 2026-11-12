"""Generate deterministic synthetic email fixtures (.eml and .mbox)."""

from __future__ import annotations

import email
from email.message import EmailMessage
import mailbox
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def make_simple_plain() -> None:
    msg = EmailMessage()
    msg["From"] = "Alice <alice@example.com>"
    msg["To"] = "Bob <bob@example.com>"
    msg["Subject"] = "Quarterly Sync"
    msg["Date"] = "Wed, 11 Nov 2026 14:00:00 +0000"
    msg["Message-ID"] = "<msg-001@example.com>"
    msg.set_content("Hi Bob,\nLet's meet tomorrow at 10 AM to discuss the roadmap.\nBest,\nAlice")
    (FIXTURES / "simple_plain.eml").write_bytes(msg.as_bytes())


def make_html_email() -> None:
    msg = EmailMessage()
    msg["From"] = "Support <support@example.com>"
    msg["To"] = "User <user@example.com>"
    msg["Subject"] = "Welcome to Prome"
    msg["Date"] = "Wed, 11 Nov 2026 15:00:00 +0000"
    msg["Message-ID"] = "<msg-002@example.com>"
    msg.add_alternative("<html><body><h1>Welcome!</h1><p>Thanks for joining Prome RAG Plugins.</p></body></html>", subtype="html")
    (FIXTURES / "html_email.eml").write_bytes(msg.as_bytes())


def make_multipart_attachment() -> None:
    msg = EmailMessage()
    msg["From"] = "Finance <finance@example.com>"
    msg["To"] = "Team <team@example.com>"
    msg["Subject"] = "Monthly Report Attached"
    msg["Date"] = "Wed, 11 Nov 2026 16:00:00 +0000"
    msg["Message-ID"] = "<msg-003@example.com>"
    msg.set_content("Please find the monthly statement attached.")
    msg.add_attachment(b"%PDF-1.4 Fake PDF Data", maintype="application", subtype="pdf", filename="statement_nov.pdf")
    (FIXTURES / "multipart_attachment.eml").write_bytes(msg.as_bytes())


def make_thread_reply() -> None:
    msg = EmailMessage()
    msg["From"] = "Bob <bob@example.com>"
    msg["To"] = "Alice <alice@example.com>"
    msg["Subject"] = "Re: Quarterly Sync"
    msg["Date"] = "Wed, 11 Nov 2026 17:00:00 +0000"
    msg["Message-ID"] = "<msg-004@example.com>"
    msg["In-Reply-To"] = "<msg-001@example.com>"
    msg["References"] = "<msg-001@example.com>"
    msg.set_content("Sounds good Alice, 10 AM works for me.")
    (FIXTURES / "thread_reply.eml").write_bytes(msg.as_bytes())


def make_unicode_eml() -> None:
    msg = EmailMessage()
    msg["From"] = "田中 <tanaka@example.jp>"
    msg["To"] = "François <francois@example.fr>"
    msg["Subject"] = "国際交流: Café et résumé"
    msg["Date"] = "Wed, 11 Nov 2026 18:00:00 +0000"
    msg["Message-ID"] = "<msg-005@example.com>"
    msg.set_content("こんにちは！ C'est un plaisir de vous rencontrer.")
    (FIXTURES / "unicode.eml").write_bytes(msg.as_bytes())


def make_empty_eml() -> None:
    (FIXTURES / "empty.eml").write_bytes(b"")


def make_malformed_eml() -> None:
    (FIXTURES / "malformed.eml").write_bytes(b"\x00\x01\x02\x03\x04")


def make_mbox_archive() -> None:
    mbox_path = FIXTURES / "archive.mbox"
    if mbox_path.exists():
        mbox_path.unlink()
    box = mailbox.mbox(str(mbox_path))
    box.lock()
    try:
        for i in range(1, 4):
            msg = EmailMessage()
            msg["From"] = f"sender{i}@example.com"
            msg["To"] = "all@example.com"
            msg["Subject"] = f"Mbox Newsletter #{i}"
            msg["Message-ID"] = f"<mbox-msg-{i}@example.com>"
            msg.set_content(f"Newsletter volume {i} body text.")
            box.add(msg)
        box.flush()
    finally:
        box.unlock()
        box.close()


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    make_simple_plain()
    make_html_email()
    make_multipart_attachment()
    make_thread_reply()
    make_unicode_eml()
    make_empty_eml()
    make_malformed_eml()
    make_mbox_archive()
    print(f"Generated email fixtures in {FIXTURES}")


if __name__ == "__main__":
    main()
