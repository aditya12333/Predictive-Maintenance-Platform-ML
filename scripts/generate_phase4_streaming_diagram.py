"""Generate the Phase 4 streaming-reliability diagram and animated GIF."""

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
AMBER = "#A66A1F"
AMBER_LIGHT = "#F6EBDD"

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
    left = (end[0] - 11 * math.cos(angle - 0.45), end[1] - 11 * math.sin(angle - 0.45))
    right = (end[0] - 11 * math.cos(angle + 0.45), end[1] - 11 * math.sin(angle + 0.45))
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
    fill: str = CARD,
) -> None:
    draw.rounded_rectangle(box, radius=12, fill=fill, outline=BORDER, width=2)
    x, y, _, _ = box
    draw.text((x + 18, y + 18), title, font=font(20, bold=True), fill=INK)
    draw.multiline_text((x + 18, y + 57), detail, font=font(15), fill=MUTED, spacing=5)


def render(phase: float) -> Image.Image:
    image = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(image)

    draw.text((48, 42), "PHASE 4 · STREAMING RELIABILITY", font=font(14, mono=True), fill=TEAL)
    draw.text(
        (48, 76),
        "Telemetry is accepted, ordered, and explainable",
        font=font(31, bold=True),
        fill=INK,
    )
    draw.text(
        (48, 121),
        "Implemented flow · no model inference shown yet",
        font=font(16),
        fill=MUTED,
    )

    source = (48, 245, 238, 355)
    stream = (294, 245, 484, 355)
    validate = (540, 245, 730, 355)
    pending = (786, 176, 1124, 286)
    stored = (786, 382, 1124, 492)
    dlq = (540, 515, 730, 625)
    warning = (294, 515, 484, 625)

    card(draw, source, "Telemetry", "engine event\nvalidated envelope")
    card(draw, stream, "Redpanda topic", "durable stream\nmanual offset commit")
    card(draw, validate, "Validate", "schema + payload\nquality gate")
    card(draw, pending, "pending_event", "JSONB buffer · event-time reorder")
    card(draw, stored, "telemetry_event", "released event · durable state")
    card(draw, dlq, "Dead-letter topic", "invalid payload\nkept for investigation", AMBER_LIGHT)
    card(draw, warning, "Quality warning", "idle flush\nordering_window_expired", AMBER_LIGHT)

    main_edges = [
        ((source[2] + 8, 300), (stream[0] - 8, 300)),
        ((stream[2] + 8, 300), (validate[0] - 8, 300)),
        ((validate[2] + 8, 275), (pending[0] - 8, 230)),
        ((pending[0] + 169, pending[3] + 8), (stored[0] + 169, stored[1] - 8)),
    ]
    for start, end in main_edges:
        arrow(draw, start, end, TEAL)
        moving_dot(draw, start, end, phase, TEAL)

    invalid_edge = ((validate[0] + 95, validate[3] + 8), (dlq[0] + 95, dlq[1] - 8))
    arrow(draw, *invalid_edge, AMBER)
    moving_dot(draw, *invalid_edge, phase, AMBER)

    idle_edge = ((pending[0] - 8, 260), (warning[2] + 8, 570))
    arrow(draw, *idle_edge, AMBER)
    moving_dot(draw, *idle_edge, (phase + 0.35) % 1.0, AMBER)

    draw.text((48, 183), "ONLINE TELEMETRY PATH", font=font(12, mono=True), fill=MUTED)
    draw.text((786, 338), "normal release", font=font(13, mono=True), fill=TEAL)
    draw.text((294, 486), "idle timeout", font=font(13, mono=True), fill=AMBER)
    draw.line((48, 650, 1152, 650), fill=BORDER, width=1)
    draw.text(
        (600, 661),
        "BUFFERED · PROCESSED · DEAD_LETTERED · DATA-QUALITY WARNING",
        font=font(12, bold=True),
        fill=MUTED,
        anchor="mm",
    )
    return image


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    static_path = OUTPUT_DIRECTORY / "phase4-streaming-reliability.png"
    gif_path = OUTPUT_DIRECTORY / "phase4-streaming-reliability.gif"
    render(0.4).save(static_path, format="PNG", optimize=True)

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
