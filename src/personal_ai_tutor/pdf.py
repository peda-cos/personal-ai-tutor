"""Tiny dependency-free PDF 1.4 writer for weekly progress reports."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Page:
    commands: list[str] = field(default_factory=list)
    footer_written: bool = False


class PDFBuilder:
    page_width = 612
    page_height = 792
    margin = 54

    def __init__(self, title: str) -> None:
        self.title = title
        self.pages: list[_Page] = []
        self.y: float = float(self.page_height - self.margin)
        self._page_number = 0
        self._new_page()

    def _new_page(self) -> None:
        self.pages.append(_Page())
        self.y = self.page_height - self.margin
        self._page_number = len(self.pages)

    @property
    def commands(self) -> list[str]:
        return self.pages[-1].commands

    def _ensure(self, height: float) -> None:
        if self.y - height < self.margin + 24:
            self._footer()
            self._new_page()

    def _footer(self) -> None:
        page = self.pages[-1]
        if page.footer_written:
            return
        command = (
            f"BT /F1 8 Tf 1 0 0 1 {self.margin} 28 Tm "
            f"(Page {self._page_number} - simulated/offline report) Tj ET"
        )
        page.commands.append(command)
        page.footer_written = True

    def text(self, value: str, *, bold: bool = False, size: int = 11, gap: float = 16) -> None:
        self._ensure(gap)
        font = "F2" if bold else "F1"
        for line in _wrap(value, 88):
            self._ensure(gap)
            self.commands.append(
                f"BT /{font} {size} Tf 1 0 0 1 {self.margin} {self.y} Tm "
                f"({_escape(line)}) Tj ET"
            )
            self.y -= gap

    def heading(self, value: str) -> None:
        self.text(value, bold=True, size=16, gap=22)

    def gap(self, value: float = 8) -> None:
        self.y -= value

    def rule(self) -> None:
        self._ensure(6)
        self.commands.append(
            f"0.75 0.75 0.75 RG 0.6 w {self.margin} {self.y} m "
            f"{self.page_width - self.margin} {self.y} l S"
        )
        self.y -= 12

    def bar(self, label: str, value: float, width: float = 300) -> None:
        self._ensure(18)
        self.commands.append(
            f"0.90 0.90 0.90 rg {self.margin + 160} {self.y - 3} {width} 9 re f"
        )
        self.commands.append(
            f"0.20 0.45 0.80 rg {self.margin + 160} {self.y - 3} "
            f"{max(0.0, min(1.0, value)) * width:.2f} 9 re f"
        )
        self.commands.append(
            f"BT /F1 10 Tf 1 0 0 1 {self.margin} {self.y - 2} Tm "
            f"({_escape(label[:32])}) Tj ET"
        )
        self.commands.append(
            f"BT /F1 9 Tf 1 0 0 1 {self.margin + 470} {self.y - 2} Tm "
            f"({value * 100:.0f}%) Tj ET"
        )
        self.y -= 18

    def render(self) -> bytes:
        self._footer()
        objects: list[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        page_object_numbers = []
        content_numbers = []
        next_number = 5
        for _page in self.pages:
            page_object_numbers.append(next_number)
            content_numbers.append(next_number + 1)
            next_number += 2
        kids = " ".join(f"{number} 0 R" for number in page_object_numbers)
        objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(self.pages)} >>".encode())
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        for page, content_number in zip(self.pages, content_numbers, strict=True):
            objects.append(
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.page_width} "
                    f"{self.page_height}] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
                    f"/Contents {content_number} 0 R >>"
                ).encode()
            )
            stream = "\n".join(page.commands).encode("latin-1", "replace")
            objects.append(
                f"<< /Length {len(stream)} >>\nstream\n".encode()
                + stream
                + b"\nendstream"
            )
        output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for index, body in enumerate(objects, start=1):
            offsets.append(len(output))
            output.extend(f"{index} 0 obj\n".encode())
            output.extend(body)
            output.extend(b"\nendobj\n")
        xref_offset = len(output)
        output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode())
        output.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info << /Title "
                f"({_escape(self.title)}) /Producer (personal_ai_tutor) >> >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode()
        )
        return bytes(output)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("\n", " ")


def _wrap(value: str, width: int) -> list[str]:
    words = value.split()
    if not words:
        return [""]
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) <= width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines
