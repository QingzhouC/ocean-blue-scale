from pathlib import Path
import math
import json

import numpy as np
import pandas as pd
from skimage.color import rgb2lab


# ============================================================
# 1. 训练数据
#
# 全部按 C1 → C10 输入
# ============================================================

REFERENCE_PALETTES = {

    "blue": [
        "#DDE8FF",
        "#CDDDFD",
        "#A3C1FF",
        "#7AA6FF",
        "#3C7DFF",
        "#105CF4",
        "#0A4ACB",
        "#09368F",
        "#06225A",
        "#041538",
    ],

    "orange_red": [
        "#FFE2DD",
        "#FFD7CF",
        "#FFA997",
        "#FF7F64",
        "#E54E2C",
        "#C62F0E",
        "#9F260C",
        "#721A06",
        "#4A1105",
        "#320C04",
    ],

    "pink": [
        "#FFDDEC",
        "#FFCFE6",
        "#FEA3CE",
        "#FE74B6",
        "#E4428F",
        "#C61E6E",
        "#991755",
        "#7F1346",
        "#570D30",
        "#30081A",
    ],

    "green": [
        "#CAFFEF",
        "#99F3D9",
        "#2FDDAA",
        "#0DBD89",
        "#009267",
        "#007855",
        "#005E43",
        "#004531",
        "#003526",
        "#003022",
    ],

    "yellow": [
        "#FFF2D8",
        "#FFE9BD",
        "#FFDD99",
        "#FFCC66",
        "#FFBB33",
        "#E69900",
        "#C08002",
        "#996600",
        "#744E01",
        "#4D3300",
    ],
}


# ============================================================
# 2. 输出
# ============================================================

OUTPUT_FOLDER = Path("results")
OUTPUT_FOLDER.mkdir(exist_ok=True)

MODEL_JSON = OUTPUT_FOLDER / "universal_palette_model.json"
NORMALIZED_CSV = OUTPUT_FOLDER / "normalized_palette_training_data.csv"


# ============================================================
# 3. HEX → RGB
# ============================================================

def hex_to_rgb(hex_color):

    h = hex_color.strip().lstrip("#")

    return np.array([
        int(h[0:2], 16),
        int(h[2:4], 16),
        int(h[4:6], 16),
    ]) / 255.0


# ============================================================
# 4. Lab → LCH
# ============================================================

def lab_to_lch(lab):

    L, a, b = lab

    C = math.sqrt(
        a ** 2
        +
        b ** 2
    )

    H = math.degrees(
        math.atan2(
            b,
            a
        )
    )

    if H < 0:
        H += 360

    return np.array([
        L,
        C,
        H
    ])


# ============================================================
# 5. HEX → LCH
# ============================================================

def hex_to_lch(hex_color):

    rgb = hex_to_rgb(
        hex_color
    )

    lab = rgb2lab(
        rgb.reshape(
            1,
            1,
            3
        )
    )[0, 0]

    return lab_to_lch(
        lab
    )


# ============================================================
# 6. Cubic Bezier 拟合
#
# 已知 P0 / P3
# 通过最小二乘法反推出 P1 / P2
# ============================================================

def fit_bezier(values):

    values = np.array(
        values,
        dtype=float
    )

    count = len(values)

    t_values = np.linspace(
        0,
        1,
        count
    )

    p0 = values[0]
    p3 = values[-1]

    A = []
    Y = []

    for t, value in zip(
        t_values,
        values
    ):

        b1 = (
            3
            *
            (1 - t) ** 2
            *
            t
        )

        b2 = (
            3
            *
            (1 - t)
            *
            t ** 2
        )

        known = (
            (1 - t) ** 3
            *
            p0

            +

            t ** 3
            *
            p3
        )

        A.append([
            b1,
            b2
        ])

        Y.append(
            value - known
        )

    A = np.array(
        A
    )

    Y = np.array(
        Y
    )

    solution, _, _, _ = (
        np.linalg.lstsq(
            A,
            Y,
            rcond=None
        )
    )

    p1 = solution[0]
    p2 = solution[1]

    return [
        float(p0),
        float(p1),
        float(p2),
        float(p3),
    ]


# ============================================================
# 7. 单套 Palette 归一化
# ============================================================

def normalize_palette(
    colors_c1_to_c10
):

    # 转成 C10 → C1
    colors = list(
        reversed(
            colors_c1_to_c10
        )
    )

    lch = np.array([
        hex_to_lch(color)
        for color
        in colors
    ])

    # C6 = index 4
    core = lch[4]

    core_L = core[0]
    core_C = core[1]

    # ========================================================
    # Dark side
    #
    # C10 → C6
    # ========================================================

    dark = lch[:5]

    dark_L_ratio = (
        dark[:, 0]
        /
        core_L
    )

    dark_C_ratio = (
        dark[:, 1]
        /
        core_C
    )

    # ========================================================
    # Light side
    #
    # C6 → C1
    # ========================================================

    light = lch[4:]

    light_L_progress = (
        (
            light[:, 0]
            -
            core_L
        )
        /
        (
            100
            -
            core_L
        )
    )

    light_C_ratio = (
        light[:, 1]
        /
        core_C
    )

    return {
        "dark_L":
            dark_L_ratio,

        "dark_C":
            dark_C_ratio,

        "light_L":
            light_L_progress,

        "light_C":
            light_C_ratio,

        "core_L":
            core_L,

        "core_C":
            core_C,

        "core_H":
            core[2],
    }


# ============================================================
# 8. 建立通用模型
# ============================================================

def build_universal_model():

    models = {}

    for (
        name,
        colors
    ) in REFERENCE_PALETTES.items():

        models[name] = normalize_palette(
            colors
        )

    # ========================================================
    # 将所有 Palette 堆叠
    # ========================================================

    dark_L = np.vstack([
        model["dark_L"]
        for model
        in models.values()
    ])

    dark_C = np.vstack([
        model["dark_C"]
        for model
        in models.values()
    ])

    light_L = np.vstack([
        model["light_L"]
        for model
        in models.values()
    ])

    light_C = np.vstack([
        model["light_C"]
        for model
        in models.values()
    ])

    # ========================================================
    # 使用 Median
    #
    # 而不是 Mean
    #
    # 减少个别特殊 Palette 对模型的影响
    # ========================================================

    median_dark_L = np.median(
        dark_L,
        axis=0
    )

    median_dark_C = np.median(
        dark_C,
        axis=0
    )

    median_light_L = np.median(
        light_L,
        axis=0
    )

    median_light_C = np.median(
        light_C,
        axis=0
    )

    # ========================================================
    # Bezier
    # ========================================================

    model = {

        "dark_lightness_bezier":
            fit_bezier(
                median_dark_L
            ),

        "light_lightness_bezier":
            fit_bezier(
                median_light_L
            ),

        "dark_chroma_bezier":
            fit_bezier(
                median_dark_C
            ),

        "light_chroma_bezier":
            fit_bezier(
                median_light_C
            ),

        # Hue 不建立跨色相曲线
        "hue_strategy":
            "LOCK_CORE_HUE",

        "training_palette_count":
            len(
                REFERENCE_PALETTES
            ),
    }

    return (
        model,
        models,
        median_dark_L,
        median_dark_C,
        median_light_L,
        median_light_C,
    )


# ============================================================
# 9. 保存所有归一化训练数据
# ============================================================

def save_training_data(
    individual_models
):

    rows = []

    dark_levels = [
        "C10",
        "C9",
        "C8",
        "C7",
        "C6",
    ]

    light_levels = [
        "C6",
        "C5",
        "C4",
        "C3",
        "C2",
        "C1",
    ]

    for (
        palette_name,
        model
    ) in individual_models.items():

        for i, level in enumerate(
            dark_levels
        ):

            rows.append({
                "Palette":
                    palette_name,

                "Side":
                    "Dark",

                "Level":
                    level,

                "L_Value":
                    model[
                        "dark_L"
                    ][i],

                "C_Value":
                    model[
                        "dark_C"
                    ][i],
            })

        for i, level in enumerate(
            light_levels
        ):

            rows.append({
                "Palette":
                    palette_name,

                "Side":
                    "Light",

                "Level":
                    level,

                "L_Value":
                    model[
                        "light_L"
                    ][i],

                "C_Value":
                    model[
                        "light_C"
                    ][i],
            })

    pd.DataFrame(
        rows
    ).to_csv(
        NORMALIZED_CSV,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 10. 主程序
# ============================================================

def main():

    (
        model,
        individual_models,
        dark_L,
        dark_C,
        light_L,
        light_C,
    ) = build_universal_model()

    # ========================================================
    # JSON
    # ========================================================

    with open(
        MODEL_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            model,
            f,
            indent=4
        )

    save_training_data(
        individual_models
    )

    # ========================================================
    # 输出
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "UNIVERSAL PALETTE MODEL"
    )

    print(
        "=" * 70
    )

    print(
        "\nTraining palettes:"
    )

    for name in REFERENCE_PALETTES:
        print(
            " -",
            name
        )

    print(
        "\nMedian Dark Lightness:"
    )

    print(
        np.round(
            dark_L,
            6
        )
    )

    print(
        "\nMedian Light Lightness:"
    )

    print(
        np.round(
            light_L,
            6
        )
    )

    print(
        "\nMedian Dark Chroma:"
    )

    print(
        np.round(
            dark_C,
            6
        )
    )

    print(
        "\nMedian Light Chroma:"
    )

    print(
        np.round(
            light_C,
            6
        )
    )

    print(
        "\nBezier Parameters"
    )

    print(
        "-" * 70
    )

    for (
        name,
        values
    ) in model.items():

        if isinstance(
            values,
            list
        ):

            print(
                f"{name}:"
            )

            print(
                np.round(
                    values,
                    6
                )
            )

    print(
        "\nHue Strategy:"
    )

    print(
        model[
            "hue_strategy"
        ]
    )

    print(
        "\nSaved:"
    )

    print(
        MODEL_JSON
    )

    print(
        NORMALIZED_CSV
    )


if __name__ == "__main__":
    main()