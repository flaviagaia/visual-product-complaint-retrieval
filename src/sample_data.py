from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Iterable

import pandas as pd
from PIL import Image, ImageDraw


IMAGE_SPECS = {
    "cracked_screen_phone.png": {
        "bg": "#f5f7fb",
        "product": "#3b4a6b",
        "accent": "#d9485f",
        "shape": "phone_crack",
    },
    "leaking_cleaner_bottle.png": {
        "bg": "#eef8ff",
        "product": "#3f88c5",
        "accent": "#1f6f8b",
        "shape": "bottle_leak",
    },
    "broken_headphone_hinge.png": {
        "bg": "#fcf7ef",
        "product": "#5a3e36",
        "accent": "#d16a5a",
        "shape": "headphone_break",
    },
    "dented_blender_jar.png": {
        "bg": "#f2fbf4",
        "product": "#708b75",
        "accent": "#d97b4f",
        "shape": "blender_dent",
    },
    "scratched_pan_surface.png": {
        "bg": "#f8f8f8",
        "product": "#444444",
        "accent": "#f29f05",
        "shape": "pan_scratch",
    },
    "torn_shirt_seam.png": {
        "bg": "#fef6f8",
        "product": "#6a4c93",
        "accent": "#cc5c7a",
        "shape": "shirt_tear",
    },
}


def _draw_image(image_path: Path, shape_name: str, bg: str, product: str, accent: str) -> None:
    image = Image.new("RGB", (320, 240), bg)
    draw = ImageDraw.Draw(image)

    if shape_name == "phone_crack":
        draw.rounded_rectangle((110, 25, 210, 215), radius=18, fill=product)
        draw.rectangle((122, 42, 198, 198), fill="#d9e2ec")
        draw.line((155, 55, 168, 98, 140, 140, 177, 176, 158, 196), fill=accent, width=4)
    elif shape_name == "bottle_leak":
        draw.rectangle((120, 40, 200, 175), fill=product)
        draw.rectangle((138, 20, 182, 45), fill=product)
        draw.polygon([(160, 175), (145, 215), (175, 215)], fill=accent)
        draw.ellipse((145, 180, 180, 220), outline=accent, width=5)
    elif shape_name == "headphone_break":
        draw.arc((65, 40, 255, 170), start=180, end=360, fill=product, width=14)
        draw.rounded_rectangle((70, 135, 115, 210), radius=14, fill=product)
        draw.rounded_rectangle((205, 135, 250, 210), radius=14, fill=product)
        draw.line((190, 100, 210, 135), fill=accent, width=8)
    elif shape_name == "blender_dent":
        draw.rectangle((130, 170, 190, 210), fill=product)
        draw.polygon([(105, 55), (215, 55), (198, 175), (122, 175)], fill="#dce9df", outline=product)
        draw.ellipse((183, 95, 212, 128), outline=accent, width=6)
    elif shape_name == "pan_scratch":
        draw.ellipse((60, 60, 225, 205), fill=product)
        draw.rectangle((220, 112, 285, 140), fill=product)
        for offset in range(0, 4):
            draw.line((95 + offset * 18, 100, 130 + offset * 18, 160), fill=accent, width=4)
    elif shape_name == "shirt_tear":
        draw.polygon([(100, 50), (220, 50), (250, 110), (210, 110), (195, 210), (125, 210), (110, 110), (70, 110)], fill=product)
        draw.line((160, 108, 148, 130, 168, 147, 158, 170), fill=accent, width=5)

    image.save(image_path)


def _write_images(image_dir: Path, image_files: Iterable[str]) -> None:
    for image_name in image_files:
        image_path = image_dir / image_name
        if image_path.exists():
            continue
        spec = IMAGE_SPECS[image_name]
        _draw_image(image_path, spec["shape"], spec["bg"], spec["product"], spec["accent"])


def build_demo_dataframe(image_dir: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "complaint_id": "VC-1001",
                "product_category": "smartphone",
                "brand": "Nova Mobile",
                "title": "Tela chegou rachada",
                "complaint_text": "Cliente recebeu smartphone com tela rachada no canto superior e linhas visíveis após ligar o aparelho.",
                "defect_type": "screen_damage",
                "severity": "high",
                "channel": "marketplace",
                "customer_segment": "premium",
                "expected_resolution": "replacement",
                "image_file": "cracked_screen_phone.png",
            },
            {
                "complaint_id": "VC-1002",
                "product_category": "cleaning",
                "brand": "Casa Clara",
                "title": "Frasco vazando na caixa",
                "complaint_text": "A embalagem do limpador chegou vazando, molhando a caixa e com indícios de tampa mal fechada.",
                "defect_type": "leakage",
                "severity": "medium",
                "channel": "ecommerce",
                "customer_segment": "mass_market",
                "expected_resolution": "refund_or_replacement",
                "image_file": "leaking_cleaner_bottle.png",
            },
            {
                "complaint_id": "VC-1003",
                "product_category": "audio",
                "brand": "Wave Sound",
                "title": "Fone com haste quebrada",
                "complaint_text": "Headphone chegou com a dobradiça direita quebrada e o arco desalinhado, impedindo o uso correto.",
                "defect_type": "structural_break",
                "severity": "high",
                "channel": "retail",
                "customer_segment": "mid_market",
                "expected_resolution": "replacement",
                "image_file": "broken_headphone_hinge.png",
            },
            {
                "complaint_id": "VC-1004",
                "product_category": "kitchen",
                "brand": "Blend Home",
                "title": "Copo do liquidificador amassado",
                "complaint_text": "O copo do liquidificador veio com deformação lateral e dificuldade para encaixar a tampa.",
                "defect_type": "dent",
                "severity": "medium",
                "channel": "marketplace",
                "customer_segment": "mid_market",
                "expected_resolution": "replacement",
                "image_file": "dented_blender_jar.png",
            },
            {
                "complaint_id": "VC-1005",
                "product_category": "cookware",
                "brand": "Chef Iron",
                "title": "Panela com riscos internos",
                "complaint_text": "A frigideira antiaderente chegou com vários riscos internos e pontos de desgaste no revestimento.",
                "defect_type": "surface_damage",
                "severity": "medium",
                "channel": "ecommerce",
                "customer_segment": "premium",
                "expected_resolution": "refund_or_replacement",
                "image_file": "scratched_pan_surface.png",
            },
            {
                "complaint_id": "VC-1006",
                "product_category": "fashion",
                "brand": "Urban Cotton",
                "title": "Costura da camisa rasgada",
                "complaint_text": "Camisa chegou com rasgo na costura frontal e tecido tensionado próximo aos botões.",
                "defect_type": "tear",
                "severity": "medium",
                "channel": "retail",
                "customer_segment": "mass_market",
                "expected_resolution": "refund_or_replacement",
                "image_file": "torn_shirt_seam.png",
            },
        ]
    ).assign(image_path=lambda df: df["image_file"].map(lambda name: str((image_dir / name).resolve())))


def ensure_demo_dataset(base_dir: str | Path) -> Path:
    base_path = Path(base_dir)
    raw_dir = base_path / "data" / "raw"
    image_dir = raw_dir / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "complaints_catalog.csv"
    dataframe = build_demo_dataframe(image_dir)
    _write_images(image_dir, dataframe["image_file"].tolist())
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".csv",
        dir=raw_dir,
        delete=False,
    ) as tmp_file:
        dataframe.to_csv(tmp_file.name, index=False)
        temp_path = Path(tmp_file.name)
    temp_path.replace(csv_path)
    return csv_path
