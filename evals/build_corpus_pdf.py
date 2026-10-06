from pathlib import Path

OUTPUT = Path(__file__).parent / "corpus" / "fernwick-operations-manual.pdf"

PAGES = [
    [
        "Fernwick Relay operations manual",
        "",
        "Page 1. Regions and data residency",
        "",
        "Fernwick Relay runs in three regions: eu-north, us-east, and ap-south.",
        "A workspace chooses its region when it is created, and the region",
        "cannot be changed later. Webhook payloads never leave the region of",
        "the workspace that received them.",
    ],
    [
        "Page 2. Maintenance and retention",
        "",
        "Planned maintenance happens every Sunday between 02:00 and 03:00 UTC.",
        "Inbound addresses keep accepting webhooks during maintenance, and",
        "deliveries resume when the window ends.",
        "",
        "Delivery logs are kept for 14 days on every plan. Payload bodies are",
        "kept for 3 days and then deleted, even when the delivery failed.",
    ],
    [
        "Page 3. Incident response",
        "",
        "Incidents have three severities. A SEV1 incident stops deliveries for",
        "more than one workspace; the on-call engineer responds within 15",
        "minutes. A SEV2 incident delays deliveries; the response target is",
        "1 hour. A SEV3 incident affects a single feature; the response target",
        "is one business day. Status updates are posted at",
        "status.fernwick.example every 30 minutes during a SEV1 incident.",
    ],
]


def escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def content_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 12 Tf", "16 TL", "72 720 Td"]
    commands += [f"({escape(line)}) Tj T*" for line in lines]
    commands.append("ET")
    return "\n".join(commands).encode("latin-1")


def build() -> bytes:
    font_id = 3
    page_ids = [4 + 2 * index for index in range(len(PAGES))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {len(PAGES)} >>".encode(),
        font_id: (
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            b"/Encoding /WinAnsiEncoding >>"
        ),
    }
    for page_id, lines in zip(page_ids, PAGES, strict=True):
        stream = content_stream(lines)
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {page_id + 1} 0 R >>"
        ).encode()
        objects[page_id + 1] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )

    output = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for object_id in sorted(objects):
        offsets[object_id] = len(output)
        output += f"{object_id} 0 obj\n".encode() + objects[object_id] + b"\nendobj\n"
    xref = len(output)
    count = max(objects) + 1
    output += f"xref\n0 {count}\n0000000000 65535 f \n".encode()
    for object_id in range(1, count):
        output += f"{offsets[object_id]:010d} 00000 n \n".encode()
    output += (
        f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(output)


if __name__ == "__main__":
    OUTPUT.write_bytes(build())
