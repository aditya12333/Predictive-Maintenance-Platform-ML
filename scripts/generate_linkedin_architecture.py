"""Generate a simple static and animated architecture graphic for LinkedIn."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CANVAS_SIZE = (1200, 675)
GIF_SIZE = (960, 540)
OUTPUT_DIRECTORY = Path("docs/media")

BACKGROUND = "#F5F3ED"
SURFACE = "#FFFFFF"
INK = "#1D2935"
MUTED = "#66727D"
BORDER = "#CCD3D1"
ACCENT = "#24766F"
ACCENT_SOFT = "#E6F0EE"
SECONDARY_SOFT = "#EEF1F4"

FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_MONO = "/System/Library/Fonts/SFNSMono.ttf"


@dataclass(frozen=True)
class Card:
    step: str
    title: str
    detail: str
    box: tuple[int, int, int, int]


CARDS = (
    Card("01", "Engine telemetry", "Sensor events\n+ edge buffer", (40, 233, 194, 365)),
    Card("02", "FastAPI ingestion", "Durable receipt\n202 Accepted", (233, 233, 387, 365)),
    Card("03", "Event stream", "Buffer · retry\nreplay", (426, 233, 580, 365)),
    Card("04", "Processing worker", "Validate · features\npredict", (619, 233, 773, 365)),
    Card("05", "PostgreSQL", "State · alerts\naudit", (812, 233, 966, 365)),
    Card("06", "Fleet dashboard", "Priorities\n+ decisions", (1005, 233, 1159, 365)),
)

OBJECT_STORAGE = (393, 459, 613, 563)
MODEL_LIFECYCLE = (666, 459, 886, 563)


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO if mono else FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(path, size=size)


def center_y(box: tuple[int, int, int, int]) -> int:
    return (box[1] + box[3]) // 2


def center_x(box: tuple[int, int, int, int]) -> int:
    return (box[0] + box[2]) // 2


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = BORDER,
) -> None:
    draw.line((start, end), fill=color, width=2)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 9
    spread = math.pi / 7
    left = (
        end[0] - length * math.cos(angle - spread),
        end[1] - length * math.sin(angle - spread),
    )
    right = (
        end[0] - length * math.cos(angle + spread),
        end[1] - length * math.sin(angle + spread),
    )
    draw.line((end, left), fill=color, width=2)
    draw.line((end, right), fill=color, width=2)


def draw_flow_dot(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    phase: float,
) -> None:
    x = start[0] + (end[0] - start[0]) * phase
    y = start[1] + (end[1] - start[1]) * phase
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=ACCENT)


def draw_card(draw: ImageDraw.ImageDraw, card: Card) -> None:
    draw.rounded_rectangle(card.box, radius=10, fill=SURFACE, outline=BORDER, width=1)
    x1, y1, _, _ = card.box
    step_box = (x1 + 14, y1 + 14, x1 + 48, y1 + 33)
    draw.rounded_rectangle(step_box, radius=5, fill=ACCENT_SOFT)
    draw.text((x1 + 31, y1 + 23), card.step, font=font(10, mono=True), fill=ACCENT, anchor="mm")
    draw.text((x1 + 14, y1 + 48), card.title, font=font(15, bold=True), fill=INK)
    draw.multiline_text(
        (x1 + 14, y1 + 82),
        card.detail,
        font=font(13),
        fill=MUTED,
        spacing=4,
    )


def render_frame(phase: float) -> Image.Image:
    image = Image.new("RGB", CANVAS_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text((40, 48), "PREDICTIVE MAINTENANCE PLATFORM", font=font(12, mono=True), fill=ACCENT)
    draw.text(
        (40, 79),
        "From engine telemetry to maintenance decisions",
        font=font(29, bold=True),
        fill=INK,
    )
    draw.text(
        (40, 122),
        "Target architecture · design approved · implementation in progress",
        font=font(14),
        fill=MUTED,
    )
    draw.text((40, 190), "ONLINE DECISION FLOW", font=font(11, mono=True), fill=MUTED)

    for card in CARDS:
        draw_card(draw, card)

    for current, following in zip(CARDS, CARDS[1:], strict=False):
        start = (current.box[2] + 7, center_y(current.box))
        end = (following.box[0] - 7, center_y(following.box))
        draw_arrow(draw, start, end)
        draw_flow_dot(draw, start, end, phase)

    draw.text((393, 420), "OFFLINE LEARNING LOOP", font=font(11, mono=True), fill=MUTED)

    draw.rounded_rectangle(OBJECT_STORAGE, radius=10, fill=SECONDARY_SOFT, outline=BORDER, width=1)
    draw.text((409, 480), "Object storage", font=font(16, bold=True), fill=INK)
    draw.text((409, 520), "Raw + trusted history", font=font(13), fill=MUTED)

    draw.rounded_rectangle(MODEL_LIFECYCLE, radius=10, fill=ACCENT_SOFT, outline=BORDER, width=1)
    draw.text((682, 480), "Training + MLflow", font=font(16, bold=True), fill=INK)
    draw.text((682, 520), "Train · evaluate · approve", font=font(13), fill=MUTED)

    stream_to_storage = (
        (center_x(CARDS[2].box), CARDS[2].box[3] + 7),
        (center_x(OBJECT_STORAGE), OBJECT_STORAGE[1] - 7),
    )
    storage_to_training = (
        (OBJECT_STORAGE[2] + 7, center_y(OBJECT_STORAGE)),
        (MODEL_LIFECYCLE[0] - 7, center_y(MODEL_LIFECYCLE)),
    )
    training_to_worker = (
        (center_x(MODEL_LIFECYCLE), MODEL_LIFECYCLE[1] - 7),
        (center_x(CARDS[3].box), CARDS[3].box[3] + 7),
    )

    for start, end in (stream_to_storage, storage_to_training):
        draw_arrow(draw, start, end)
        draw_flow_dot(draw, start, end, phase)
    draw_arrow(draw, *training_to_worker, color=ACCENT)
    draw_flow_dot(draw, *training_to_worker, phase)

    draw.line((40, 608, 1160, 608), fill=BORDER, width=1)
    footer = "Event-time ordered   ·   Idempotent processing   ·   Human-approved models"
    draw.text((600, 636), footer, font=font(13, bold=True), fill=MUTED, anchor="mm")
    return image


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    static_path = OUTPUT_DIRECTORY / "predictive-maintenance-architecture.png"
    render_frame(0.45).save(static_path, format="PNG", optimize=True)

    animation_path = OUTPUT_DIRECTORY / "predictive-maintenance-architecture.gif"
    rgb_frames = [
        render_frame(frame_index / 32).resize(GIF_SIZE, Image.Resampling.LANCZOS)
        for frame_index in range(32)
    ]
    palette = rgb_frames[0].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    gif_frames = [palette]
    gif_frames.extend(
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in rgb_frames[1:]
    )
    gif_frames[0].save(
        animation_path,
        format="GIF",
        save_all=True,
        append_images=gif_frames[1:],
        duration=120,
        loop=0,
        optimize=True,
        disposal=1,
    )

    print(static_path)
    print(animation_path)


if __name__ == "__main__":
    main()
