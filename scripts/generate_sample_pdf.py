"""Generate the tiny, dependency-free PDF used as a sample input.

The repository ships the generated PDF as well; this script documents exactly
how it was produced and can regenerate it.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap

LINES = [
    "PDF Ingestion Note",
    "",
    "PDF files often contain the official version of a policy, design note, or",
    "operating procedure. The ingestion service extracts the text layer page by",
    "page, records the page count, and passes the resulting text to the same",
    "chunking pipeline used for Markdown and plain-text files.",
    "",
    "Scanned PDFs need optical character recognition before their text can be",
    "indexed. Tables and figures need specialized extractors when their layout",
    "carries meaning. Those extractors can emit Document records with the same",
    "source and page metadata used by the text pipeline.",
]


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf() -> bytes:
    text_lines: list[str] = []
    for line in LINES:
        text_lines.extend(wrap(line, width=88) if line else [""])

    commands = ["BT", "/F1 18 Tf", "72 760 Td", "22 TL"]
    for index, line in enumerate(text_lines):
        if index == 1:
            commands.extend(["/F1 11 Tf", "16 TL"])
        commands.append(f"({_escape(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    stream = "\n".join(commands).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    output = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


def main() -> int:
    target = Path(__file__).resolve().parents[1] / "data" / "sample_docs" / "pdf_ingestion_note.pdf"
    target.write_bytes(build_pdf())
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
