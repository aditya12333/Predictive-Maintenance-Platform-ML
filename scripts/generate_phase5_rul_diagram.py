"""Generate the Phase 5 RUL training and safe-inference social visual."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIRECTORY = Path("docs/media")
CANVAS = (1200, 675)
GIF_SIZE = (900, 506)

BG = "#F5F3ED"
CARD = "#FFFFFF"
INK = "#1D2935"
MUTED = "#66727D"
BORDER = "#CCD3D1"
TEAL = "#24766F"
TEAL_LIGHT = "#E6F0EE"
BLUE = "#365F91"
BLUE_LIGHT = "#E8EEF6"
AMBER = "#A66A1F"
AMBER_LIGHT = "#F6EBDD"
RED = "#A8483D"
RED_LIGHT = "#F6E8E6"

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = MONO if mono else BOLD if bold else FONT
    return ImageFont.truetype(path, size=size)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
) -> None:
    draw.line((*start, *end), fill=color, width=3)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    left = (end[0] - 10 * math.cos(angle - 0.45), end[1] - 10 * math.sin(angle - 0.45))
    right = (end[0] - 10 * math.cos(angle + 0.45), end[1] - 10 * math.sin(angle + 0.45))
    draw.line((*end, *left), fill=color, width=3)
    draw.line((*end, *right), fill=color, width=3)


def moving_dot(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    phase: float,
    color: str,
) -> None:
    x = int(start[0] + (end[0] - start[0]) * phase)
    y = int(start[1] + (end[1] - start[1]) * phase)
    draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    detail: str,
    *,
    fill: str = CARD,
    title_color: str = INK,
) -> None:
    draw.rounded_rectangle(box, radius=12, fill=fill, outline=BORDER, width=2)
    x, y, _, _ = box
    draw.text((x + 15, y + 15), title, font=font(18, bold=True), fill=title_color)
    draw.multiline_text((x + 15, y + 47), detail, font=font(14), fill=MUTED, spacing=4)


def badge(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    detail: str,
    fill: str,
    color: str,
) -> None:
    draw.rounded_rectangle(box, radius=10, fill=fill, outline=color, width=2)
    x, y, _, _ = box
    draw.text((x + 13, y + 10), label, font=font(14, bold=True), fill=color)
    draw.text((x + 13, y + 34), detail, font=font(12), fill=MUTED)


def draw_lane(
    draw: ImageDraw.ImageDraw,
    boxes: list[tuple[int, int, int, int]],
    phase: float,
    color: str,
) -> None:
    for current, following in zip(boxes, boxes[1:], strict=False):
        start = (current[2] + 7, (current[1] + current[3]) // 2)
        end = (following[0] - 7, (following[1] + following[3]) // 2)
        arrow(draw, start, end, color)
        moving_dot(draw, start, end, phase, color)


def render(phase: float) -> Image.Image:
    image = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(image)

    draw.text((48, 36), "PHASE 5 · RUL MODEL + SAFE INFERENCE", font=font(14, mono=True), fill=TEAL)
    draw.text(
        (48, 68),
        "From trusted telemetry to traceable predictions",
        font=font(31, bold=True),
        fill=INK,
    )
    draw.text(
        (48, 111),
        "One feature contract across training and serving",
        font=font(16),
        fill=MUTED,
    )

    draw.text((48, 153), "OFFLINE MODEL EVIDENCE", font=font(12, mono=True), fill=BLUE)
    offline = [
        (48, 177, 242, 277),
        (278, 177, 472, 277),
        (508, 177, 702, 277),
        (738, 177, 932, 277),
        (968, 177, 1152, 277),
    ]
    card(draw, offline[0], "Trusted FD001", "manifest verified\n100 train engines")
    card(draw, offline[1], "Engine split", "80 train · 20 val\nno row leakage")
    card(draw, offline[2], "4 regressors", "Linear · Lasso\nXGBoost · LightGBM")
    card(draw, offline[3], "LightGBM", "best validation MAE\n23.419 cycles", fill=BLUE_LIGHT)
    card(draw, offline[4], "Official test", "MAE 19.447\nRMSE 26.820", fill=BLUE_LIGHT)
    draw_lane(draw, offline, phase, BLUE)

    draw.text((48, 322), "ONLINE QUALITY-AWARE INFERENCE", font=font(12, mono=True), fill=TEAL)
    online = [
        (48, 346, 242, 446),
        (278, 346, 472, 446),
        (508, 346, 702, 446),
        (738, 346, 932, 446),
        (968, 346, 1152, 446),
    ]
    card(draw, online[0], "Ordered event", "durable pending row\ntelemetry-v1")
    card(draw, online[1], "Quality gate", "schema + sensors\nvalidity + flags")
    card(draw, online[2], "features-v1", "cycle + 21 sensors\nfixed ordering")
    card(
        draw,
        online[3],
        "Approved model",
        "explicit lifecycle gate\nno auto-promotion",
        fill=TEAL_LIGHT,
    )
    card(draw, online[4], "PostgreSQL", "telemetry + prediction\none transaction", fill=TEAL_LIGHT)
    draw_lane(draw, online, (phase + 0.25) % 1.0, TEAL)

    badge(draw, (278, 500, 522, 557), "AVAILABLE", "valid input + RUL", TEAL_LIGHT, TEAL)
    badge(draw, (534, 500, 778, 557), "DEGRADED", "known non-fatal issue", AMBER_LIGHT, AMBER)
    badge(draw, (790, 500, 1034, 557), "WITHHELD", "unsafe input · no RUL", RED_LIGHT, RED)

    draw.line((48, 591, 1152, 591), fill=BORDER, width=1)
    draw.text(
        (600, 619),
        "VERSIONED · LEAKAGE-SAFE · APPROVAL-GATED · IDEMPOTENT · RETRYABLE",
        font=font(13, bold=True),
        fill=MUTED,
        anchor="mm",
    )
    draw.text(
        (600, 647),
        "141 TESTS PASSED · REAL POSTGRESQL TRANSACTION VERIFICATION",
        font=font(11, mono=True),
        fill=TEAL,
        anchor="mm",
    )
    return image


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    static_path = OUTPUT_DIRECTORY / "phase5-rul-training-inference.png"
    gif_path = OUTPUT_DIRECTORY / "phase5-rul-training-inference.gif"
    render(0.42).save(static_path, format="PNG", optimize=True)

    frames = [render(index / 24).resize(GIF_SIZE, Image.Resampling.LANCZOS) for index in range(24)]
    palette = frames[0].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    indexed = [palette]
    indexed.extend(
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames[1:]
    )
    indexed[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=indexed[1:],
        duration=130,
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(static_path)
    print(gif_path)


if __name__ == "__main__":
    main()
