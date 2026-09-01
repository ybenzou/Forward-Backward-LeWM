"""Extract embedded SVG drawings from Draw.io files and convert them to PDF."""

from __future__ import annotations

import base64
import html
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

from selenium import webdriver
from selenium.webdriver.common.print_page_options import PrintOptions
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service


HERE = Path(__file__).resolve().parent
EXPORTS = {
    HERE / "forward_training.drawio": HERE / "fig_has_training.svg",
    HERE / "forward_eval.drawio": HERE / "fig_has_evaluation.svg",
}

LABEL_REPLACEMENTS = {
    "Auto-Forward LeWM Training Pipeline": "HAS Training Pipeline",
    "Auto-Forward LeWM Eval Pipeline": "HAS Evaluation Pipeline",
}

DATA_URI = re.compile(
    r"data:image/svg\+xml(?:;base64)?,([A-Za-z0-9+/=%]+)",
    flags=re.IGNORECASE,
)


def decode_payload(payload: str) -> bytes | None:
    decoded_uri = unquote(payload)
    if decoded_uri.lstrip().startswith(("<svg", "<?xml")):
        return decoded_uri.encode("utf-8")

    try:
        decoded = base64.b64decode(decoded_uri, validate=False)
    except (ValueError, base64.binascii.Error):
        return None
    if decoded.lstrip().startswith((b"<svg", b"<?xml")):
        return decoded
    return None


def embedded_svgs(drawio_path: Path) -> list[bytes]:
    root = ET.parse(drawio_path).getroot()
    candidates: list[bytes] = []
    for element in root.iter():
        values = list(element.attrib.values())
        if element.text:
            values.append(element.text)
        for value in values:
            expanded = html.unescape(value)
            for match in DATA_URI.finditer(expanded):
                decoded = decode_payload(match.group(1))
                if decoded is not None:
                    candidates.append(decoded)
    return candidates


def sanitize_for_pdf(svg: bytes) -> bytes:
    """Resolve Draw.io adaptive-color CSS unsupported by CairoSVG."""
    text = svg.decode("utf-8")
    for old_label, new_label in LABEL_REPLACEMENTS.items():
        text = text.replace(old_label, new_label)
    text = re.sub(
        r"<style[^>]*>@supports \(color: light-dark.*?</style>",
        "",
        text,
        flags=re.DOTALL,
    )
    text = text.replace(
        "light-dark(#ffffff, var(--ge-dark-color, #121212))",
        "#ffffff",
    )
    text = re.sub(
        r"light-dark\((rgb\([^)]+\)),\s*rgb\([^)]+\)\)",
        r"\1",
        text,
    )
    text = re.sub(
        r"light-dark\((#[0-9a-fA-F]{3,8}),\s*#[0-9a-fA-F]{3,8}\)",
        r"\1",
        text,
    )
    text = re.sub(
        r"var\(--[^,]+,\s*([^)]+)\)",
        r"\1",
        text,
    )
    return text.encode("utf-8")


def svg_size(svg: bytes) -> tuple[int, int]:
    root = ET.fromstring(svg)
    width = int(float(root.attrib["width"].removesuffix("px")))
    height = int(float(root.attrib["height"].removesuffix("px")))
    return width, height


def export(drawio_path: Path, svg_path: Path) -> tuple[Path, tuple[int, int]]:
    candidates = embedded_svgs(drawio_path)
    if not candidates:
        raise RuntimeError(f"No embedded SVG found in {drawio_path}")

    # Copy-as-SVG stores the complete drawing as the largest embedded payload.
    svg = sanitize_for_pdf(max(candidates, key=len))
    svg_path.write_bytes(svg)
    print(
        f"{drawio_path.name}: {len(candidates)} SVG candidate(s), "
        f"exported {svg_path.name}"
    )
    return svg_path, svg_size(svg)


def render_with_firefox(exports: list[tuple[Path, tuple[int, int]]]) -> None:
    options = Options()
    options.add_argument("-headless")
    service = Service(executable_path="/snap/bin/geckodriver")

    driver = webdriver.Firefox(options=options, service=service)
    try:
        for svg_path, (width, height) in exports:
            driver.set_window_size(width, height)
            driver.get(svg_path.as_uri())

            print_options = PrintOptions()
            print_options.background = True
            print_options.margin_top = 0
            print_options.margin_bottom = 0
            print_options.margin_left = 0
            print_options.margin_right = 0
            print_options.page_width = 18.5
            print_options.page_height = 18.5 * height / width
            print_options.shrink_to_fit = True
            encoded_pdf = driver.print_page(print_options)
            pdf_path = svg_path.with_suffix(".pdf")
            pdf_path.write_bytes(base64.b64decode(encoded_pdf))
            subprocess.run(
                [
                    "pdftoppm",
                    "-png",
                    "-singlefile",
                    "-r",
                    "120",
                    str(pdf_path),
                    str(svg_path.with_suffix("")),
                ],
                check=True,
            )
    finally:
        driver.quit()


def main() -> None:
    exports = [export(source, target) for source, target in EXPORTS.items()]
    render_with_firefox(exports)


if __name__ == "__main__":
    main()
