"""Render the project's architecture as a high-resolution PNG and a looping GIF.

Run: .venv/bin/python scripts/generate_final_architecture.py
Bundled logos make rendering offline and reproducible (see logos/SOURCES.md).
Edges are shared by the static drawing and animation. Markers travel only on
connectors, never through component cards. The animation is illustrative, not
a claim about system timing, or automatic synchronization of the local registry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "architecture"
LOGOS = OUT / "logos"
WIDTH, HEIGHT = 2400, 2050
BG = "#F4F6FA"
INK = "#172A43"
TEXT = "#42546C"
MUTED = "#66778B"
WHITE = "#FFFFFF"
BORDER = "#DCE4EE"
BLUE = "#2374CC"
TEAL = "#16817D"
PURPLE = "#7952B3"
ORANGE = "#BE651D"
Point = tuple[float, float]
Box = tuple[int, int, int, int]


@lru_cache
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    raise RuntimeError("Install Arial or DejaVu Sans to render this diagram")


@dataclass(frozen=True)
class Card:
    key: str
    box: Box
    title: str
    role: str
    lines: tuple[str, ...]
    logos: tuple[str, ...]
    color: str
    compact: bool = False


@dataclass(frozen=True)
class Edge:
    key: str
    points: tuple[Point, ...]
    color: str
    dashed: bool = False

    @property
    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.points, self.points[1:], strict=False))

    def point(self, progress: float) -> Point:
        distance = min(1.0, max(0.0, progress)) * self.length
        for start, end in zip(self.points, self.points[1:], strict=False):
            length = math.dist(start, end)
            if distance <= length:
                ratio = distance / length
                return (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                )
            distance -= length
        return self.points[-1]


X = (80, 460, 840, 1220, 1600, 1980)


def standard_box(column: int, y: int, height: int = 280) -> Box:
    return X[column], y, X[column] + 300, y + height


CARDS = [
    Card(
        "demo",
        standard_box(0, 320),
        "Demo telemetry",
        "INPUT",
        ("Synthetic sensor stream", "FD001-like readings", "Sent from dashboard"),
        ("react", "typescript"),
        BLUE,
    ),
    Card(
        "ingest",
        standard_box(1, 320),
        "FastAPI",
        "INGEST",
        ("POST /v1/telemetry", "Pydantic contracts", "Schema validation"),
        ("fastapi",),
        BLUE,
    ),
    Card(
        "broker",
        standard_box(2, 320),
        "Redpanda",
        "BUFFER",
        ("Kafka-compatible broker", "telemetry.events topic", "Consumer + dead letters"),
        ("redpanda",),
        BLUE,
    ),
    Card(
        "worker",
        standard_box(3, 320),
        "Inference worker",
        "PREDICT",
        ("Order & validate events", "22 numeric features", "LightGBM → RUL cycles"),
        ("python", "lightgbm"),
        BLUE,
    ),
    Card(
        "database",
        standard_box(4, 320),
        "PostgreSQL",
        "PERSIST",
        ("Telemetry & predictions", "Alerts & quality issues", "SQLAlchemy + Alembic"),
        ("postgresql",),
        BLUE,
    ),
    Card(
        "dashboard",
        standard_box(5, 320),
        "Operations UI",
        "VISUALIZE",
        ("Next.js · React · TS", "Fleet, RUL & history", "Reads through FastAPI"),
        ("nextjs", "react", "typescript"),
        BLUE,
    ),
    Card(
        "metrics",
        (80, 750, 700, 895),
        "Metrics & logs",
        "OBSERVABILITY",
        (
            "Prometheus: API / worker metrics",
            "Structured JSON application logs",
        ),
        ("prometheus",),
        TEAL,
        True,
    ),
    Card(
        "reports",
        (840, 750, 1520, 895),
        "Quality & model monitoring",
        "REPORTS",
        (
            "Quality · availability · PSI drift · label-based MAE",
            "AWS: cron → JSON reports; local: monitoring DAG",
        ),
        ("python",),
        TEAL,
        True,
    ),
    Card(
        "labels",
        (1660, 750, 2280, 895),
        "Actual RUL labels",
        "DELAYED FEEDBACK",
        (
            "Manual outcome entry → stored in PostgreSQL",
            "MAE needs actual outcomes, not predictions alone",
        ),
        ("postgresql",),
        TEAL,
        True,
    ),
    Card(
        "data",
        standard_box(0, 1080, 260),
        "NASA C-MAPSS",
        "HISTORICAL DATA",
        ("FD001 engine lifecycles", "Training RUL labels", "Official test + RUL files"),
        (),
        PURPLE,
    ),
    Card(
        "features",
        standard_box(1, 1080, 260),
        "Data & features",
        "PREPARE",
        ("Polars → Parquet", "Dataset manifests", "Shared features-v1"),
        ("polars", "parquet"),
        PURPLE,
    ),
    Card(
        "airflow",
        standard_box(2, 1080, 260),
        "Apache Airflow",
        "ORCHESTRATE",
        ("Prepare / train / package", "Register / benchmark", "Manual approval sensor"),
        ("airflow",),
        PURPLE,
    ),
    Card(
        "train",
        standard_box(3, 1080, 260),
        "Model comparison",
        "TRAIN & EVALUATE",
        ("LR · Adaptive Lasso", "XGBoost · LightGBM", "MAE · RMSE · NASA"),
        ("sklearn", "lightgbm"),
        PURPLE,
    ),
    Card(
        "registry",
        standard_box(4, 1080, 260),
        "MLflow",
        "TRACK & REGISTER",
        ("Experiments & registry", "Metrics, artifacts, lineage", "Versioned candidates"),
        ("mlflow",),
        PURPLE,
    ),
    Card(
        "approve",
        standard_box(5, 1080, 260),
        "Approve & promote",
        "HUMAN REVIEW",
        ("Benchmark + evidence", "Audited approval decision", "champion / previous"),
        ("mlflow",),
        PURPLE,
    ),
    Card(
        "ci",
        (80, 1570, 525, 1820),
        "GitHub Actions",
        "VERIFY",
        ("Pytest + PostgreSQL integration", "Ruff · Mypy · frontend build", "Compose validation"),
        ("githubactions",),
        ORANGE,
    ),
    Card(
        "containers",
        (665, 1570, 1110, 1820),
        "Docker",
        "PACKAGE",
        (
            "API / worker and dashboard images",
            "Built from versioned source",
            "Staging smoke test gates deployment",
        ),
        ("docker",),
        ORANGE,
    ),
    Card(
        "ecr",
        (1250, 1570, 1695, 1820),
        "Amazon ECR",
        "PUBLISH",
        (
            "Private container image registry",
            "Commit-SHA image tags",
            "Release images pulled by server",
        ),
        ("ecr",),
        ORANGE,
    ),
    Card(
        "lightsail",
        (1835, 1570, 2280, 1820),
        "AWS Lightsail",
        "DEPLOY",
        (
            "GitHub Actions → SSH deployment",
            "Docker Compose service updates",
            "Hosts the live runtime shown above",
        ),
        ("lightsail", "docker"),
        ORANGE,
    ),
]

EDGES = [
    *[Edge(f"live-{i}", ((X[i] + 300, 450), (X[i + 1], 450)), BLUE) for i in range(5)],
    Edge("metrics", ((1280, 600), (1280, 680), (390, 680), (390, 750)), TEAL),
    Edge("reports", ((1640, 600), (1640, 680), (1400, 680), (1400, 750)), TEAL),
    Edge("outcomes", ((1660, 823), (1520, 823)), TEAL),
    Edge("local-policy", ((1120, 895), (1120, 965), (990, 965), (990, 1080)), PURPLE, True),
    *[Edge(f"training-{i}", ((X[i] + 300, 1210), (X[i + 1], 1210)), PURPLE) for i in range(5)],
    Edge(
        "champion",
        ((2280, 1190), (2360, 1190), (2360, 190), (1370, 190), (1370, 320)),
        PURPLE,
        True,
    ),
    Edge("build", ((525, 1690), (665, 1690)), ORANGE),
    Edge("publish", ((1110, 1690), (1250, 1690)), ORANGE),
    Edge("deploy", ((1695, 1690), (1835, 1690)), ORANGE),
]


def text(draw, xy, value, size=23, color=TEXT, bold=False):
    draw.text(xy, value, font=font(size, bold), fill=color)


def fitted_text(draw, xy, value, width, size=23, color=TEXT, bold=False):
    """Reject clipped text instead of silently producing a broken export."""
    assert draw.textlength(value, font=font(size, bold)) <= width, value
    text(draw, xy, value, size, color, bold)


def center_label(draw, xy, value, color, size=17):
    label_width = draw.textlength(value, font=font(size, True))
    text(draw, (xy[0] - label_width / 2, xy[1]), value, size, color, True)


@lru_cache
def logo(name: str, size: int) -> Image.Image:
    with Image.open(LOGOS / f"{name}.png") as source:
        picture = source.convert("RGBA")
    picture.thumbnail((size, size), Image.Resampling.LANCZOS)
    return picture


def stamp_logo(image, name, x, y, size=58):
    picture = logo(name, size)
    image.paste(
        picture,
        (int(x + (size - picture.width) / 2), int(y + (size - picture.height) / 2)),
        picture,
    )


def draw_card(image, card):
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = card.box
    draw.rounded_rectangle((left, top + 5, right, bottom + 5), 17, fill="#E8EDF4")
    draw.rounded_rectangle(card.box, 17, fill=WHITE, outline=BORDER, width=2)
    draw.rounded_rectangle((left + 20, top, left + 78, top + 5), 2, fill=card.color)
    if card.compact:
        stamp_logo(image, card.logos[0], left + 22, top + 24, 56)
        text(draw, (left + 96, top + 20), card.role, 15, card.color, True)
        fitted_text(draw, (left + 96, top + 42), card.title, right - left - 120, 25, INK, True)
        for index, line in enumerate(card.lines):
            fitted_text(draw, (left + 24, top + 85 + index * 27), line, right - left - 48, 21)
        return
    for index, name in enumerate(card.logos):
        stamp_logo(image, name, left + 24 + index * 69, top + 21, 56)
    if not card.logos:
        # Dataset monogram is a diagram symbol, not a fabricated NASA logo.
        draw.rounded_rectangle((left + 24, top + 21, left + 132, top + 77), 9, fill="#EEE8F7")
        text(draw, (left + 35, top + 34), "FD001", 25, PURPLE, True)
    text(draw, (left + 24, top + 93), card.role, 15, card.color, True)
    fitted_text(draw, (left + 24, top + 118), card.title, right - left - 48, 26, INK, True)
    for index, line in enumerate(card.lines):
        fitted_text(draw, (left + 24, top + 159 + index * 29), line, right - left - 48, 22)
    assert top + 159 + len(card.lines) * 29 - 7 <= bottom, card.key


def draw_edge(draw, edge):
    if edge.dashed:
        distance = 0.0
        while distance < edge.length:
            a = edge.point(distance / edge.length)
            b = edge.point(min(distance + 9, edge.length) / edge.length)
            draw.line((a, b), fill=edge.color, width=3)
            distance += 17
    else:
        draw.line(edge.points, fill=edge.color, width=4, joint="curve")
    end = edge.points[-1]
    previous = edge.points[-2]
    angle = math.atan2(end[1] - previous[1], end[0] - previous[0])
    points = [end] + [
        (end[0] - 13 * math.cos(angle + offset), end[1] - 13 * math.sin(angle + offset))
        for offset in (-0.48, 0.48)
    ]
    draw.polygon(points, fill=edge.color)


def panel(draw, box, title, color, scope):
    left, top, right, _ = box
    draw.rounded_rectangle(box, 24, fill="#FCFDFE", outline=BORDER, width=2)
    text(draw, (left + 36, top + 25), title, 28, color, True)
    scope_width = draw.textlength(scope, font=font(20))
    text(draw, (right - scope_width - 36, top + 30), scope, 20, MUTED)


def render_static() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 9), fill=INK)
    text(draw, (80, 46), "Predictive Maintenance", 59, INK, True)
    text(
        draw,
        (83, 118),
        "End-to-end ML platform  /  From sensor readings to RUL predictions",
        27,
        MUTED,
    )
    draw.rounded_rectangle((1850, 47, 2280, 135), 15, fill=INK)
    text(draw, (1873, 65), "NASA C-MAPSS · FD001", 25, WHITE, True)
    text(draw, (1873, 101), "Selected model: LightGBM", 21, "#C9D7E8")

    panel(
        draw,
        (40, 230, 2320, 925),
        "01  STREAMING INFERENCE",
        BLUE,
        "AWS Lightsail runtime · Docker Compose",
    )
    panel(
        draw,
        (40, 1000, 2320, 1380),
        "02  MODEL DEVELOPMENT & GOVERNANCE",
        PURPLE,
        "Local training and approval workflow",
    )
    panel(
        draw,
        (40, 1480, 2320, 1840),
        "03  CONTINUOUS INTEGRATION & DELIVERY",
        ORANGE,
        "GitHub → ECR → Lightsail",
    )

    for edge in EDGES:
        draw_edge(draw, edge)
    for card in CARDS:
        draw_card(image, card)

    for index, label in enumerate(("POST", "publish", "consume", "write", "FastAPI")):
        center_label(draw, (X[index] + 340, 419), label, BLUE, 15)
    text(draw, (740, 647), "Metrics from API + worker", 20, TEAL)
    text(draw, (1425, 709), "DB records", 18, TEAL)
    center_label(draw, (1590, 792), "labels", TEAL)
    text(
        draw,
        (1170, 945),
        "LOCAL DAG ONLY: two failed monitoring windows → candidate run",
        20,
        PURPLE,
    )
    text(
        draw,
        (1460, 158),
        "Manual registry sync to AWS; worker loads approved champion on startup",
        20,
        PURPLE,
    )

    # Explain the two cross-cutting contracts without routing arrows through cards.
    text(draw, (81, 1405), "FEATURE CONTRACT", 16, PURPLE, True)
    text(
        draw,
        (283, 1401),
        "cycle + 21 sensors · same ordering & validation in training and serving",
        22,
        TEXT,
    )
    text(draw, (1460, 1401), "Prediction states: available / degraded / withheld", 22, TEXT)
    for x, label in ((595, "build"), (1180, "push"), (1765, "pull + SSH")):
        center_label(draw, (x, 1658), label, ORANGE)

    # Footer keeps important deployment qualifications part of the exported image.
    sections = [
        (
            80,
            "STORAGE",
            ("PostgreSQL volume + model/report files", "on Lightsail disk. No S3 bucket."),
        ),
        (
            850,
            "MODEL LIFECYCLE",
            (
                "Train and approve locally; copy the registry to AWS.",
                "Model updates are separate from container releases.",
            ),
        ),
        (
            1670,
            "HOW TO READ",
            (
                "Solid: data / artifacts. Dashed: control / setup.",
                "Animation illustrates flow, not measured latency.",
            ),
        ),
    ]
    for x, heading, lines in sections:
        text(draw, (x, 1890), heading, 17, MUTED, True)
        for index, line in enumerate(lines):
            text(draw, (x, 1922 + index * 29), line, 21, TEXT)
    text(
        draw,
        (80, 2005),
        "Technology logos identify their respective products. Sources bundled with the diagram.",
        16,
        MUTED,
    )
    return image


def validate_geometry():
    """Check every animated route, including bends, against every card interior."""
    assert len({edge.key for edge in EDGES}) == len(EDGES)
    for edge in EDGES:
        assert edge.length > 0
        for a, b in zip(edge.points, edge.points[1:], strict=False):
            assert a[0] == b[0] or a[1] == b[1], edge.key
        # One sample per pixel, including endpoints; borders themselves are allowed.
        for sample in range(math.ceil(edge.length) + 1):
            x, y = edge.point(sample / math.ceil(edge.length))
            assert 0 <= x < WIDTH and 0 <= y < HEIGHT, edge.key
            for card in CARDS:
                left, top, right, bottom = card.box
                assert not (left < x < right and top < y < bottom), (edge.key, card.key)


def render_gif(base: Image.Image):
    frame_count = 72
    scale = 0.8
    size = round(WIDTH * scale), round(HEIGHT * scale)
    # One shared palette avoids color flicker between frames, particularly in logos.
    small = base.resize(size, Image.Resampling.LANCZOS)
    palette = small.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = []
    for frame_index in range(frame_count):
        frame = small.copy()
        draw = ImageDraw.Draw(frame)
        for index, edge in enumerate(EDGES):
            period = frame_count if edge.length > 500 else frame_count // 3
            progress = (frame_index / period + index * 0.19) % 1.0
            # Keep marker radii outside cards at both endpoints.
            inset = 11 / edge.length
            progress = inset + progress * (1 - 2 * inset)
            x, y = edge.point(progress)
            x, y = x * scale, y * scale
            radius = 6.5
            draw.ellipse(
                (x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2), fill=WHITE
            )
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=edge.color)
        frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
    frames[0].save(
        OUT / "final-architecture.gif",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=False,
        disposal=1,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    validate_geometry()
    base = render_static()
    base.save(OUT / "final-architecture.png", optimize=True)
    render_gif(base)
    print(f"Saved PNG ({WIDTH} × {HEIGHT}) and GIF (72 frames) to {OUT}")
    print(f"Verified all {len(EDGES)} connector routes stay outside component cards.")


if __name__ == "__main__":
    main()
