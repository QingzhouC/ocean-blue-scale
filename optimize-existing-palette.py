from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.color import rgb2lab, lab2rgb


# ============================================================
# 1. 输入现有品牌色板
#
# 顺序必须是：
# C10 → C1
#
# C6 为核心品牌色
# ============================================================

ORIGINAL_PALETTE = [
    "#e8ffed",  # C10
    "#aff0be",  # C9
    "#7be093",  # C8
    "#50d46f",  # C7
    "#27c44c",  # C6
    "#1db842",  # C5
    "#139c33",  # C4
    "#0a8226",  # C3
    "#04661b",  # C2
    "#004d12",  # C1
]


LEVELS = [
    "C10",
    "C9",
    "C8",
    "C7",
    "C6",
    "C5",
    "C4",
    "C3",
    "C2",
    "C1",
]


CORE_INDEX = 4


# ============================================================
# 2. Bézier 优化参数
# ============================================================

# Lightness
DARK_L_BEZIER = (
    0.25,
    0.08,
    0.72,
    0.92
)

LIGHT_L_BEZIER = (
    0.25,
    0.10,
    0.68,
    0.96
)


# 让 Chroma 在 C6 附近保持更强，
# 然后向两端更自然地下降
DARK_C_BEZIER = (
    0.22,
    0.05,
    0.68,
    0.94
)

LIGHT_C_BEZIER = (
    0.20,
    0.06,
    0.62,
    0.92
)


# Hue
HUE_BEZIER = (
    0.30,
    0.25,
    0.70,
    0.75
)


# Hue 优化强度
#
# 0 = 完全保持原 Hue
# 1 = 完全使用平滑 Hue
HUE_SMOOTHING_STRENGTH = 0.30


# ============================================================
# 3. 输出
#
# 每次运行结果按 001、002、003 … 递增保存
# 图片存放在 images/<序号>/ 子文件夹中，与 CSV 区分
# ============================================================

OUTPUT_FOLDER = Path("final")
OUTPUT_FOLDER.mkdir(exist_ok=True)

IMAGE_FOLDER = OUTPUT_FOLDER / "images"
IMAGE_FOLDER.mkdir(exist_ok=True)


def get_next_output_paths():

    existing = sorted(
        OUTPUT_FOLDER.glob(
            "optimized_existing_palette*.csv"
        )
    )

    max_num = 0

    for f in existing:

        stem = f.stem

        try:

            num = int(
                stem.replace(
                    "optimized_existing_palette",
                    ""
                ).lstrip("_")
            )

            max_num = max(
                max_num,
                num
            )

        except ValueError:
            continue

    next_num = (
        max_num + 1
    )

    name = (
        f"optimized_existing_palette"
        f"_{next_num:03d}"
    )

    csv_path = (
        OUTPUT_FOLDER
        /
        f"{name}.csv"
    )

    run_image_folder = (
        IMAGE_FOLDER
        /
        f"{next_num:03d}"
    )

    run_image_folder.mkdir(exist_ok=True)

    palette_image_path = (
        run_image_folder
        /
        "optimized_existing_palette.png"
    )

    lightness_curve_path = (
        run_image_folder
        /
        "lightness_comparison.png"
    )

    chroma_curve_path = (
        run_image_folder
        /
        "chroma_comparison.png"
    )

    hue_curve_path = (
        run_image_folder
        /
        "hue_comparison.png"
    )

    return (
        csv_path,
        palette_image_path,
        lightness_curve_path,
        chroma_curve_path,
        hue_curve_path
    )


# ============================================================
# 4. HEX ↔ RGB
# ============================================================

def hex_to_rgb(hex_color):

    h = (
        hex_color
        .strip()
        .lstrip("#")
    )

    return np.array([
        int(h[0:2], 16),
        int(h[2:4], 16),
        int(h[4:6], 16),
    ]) / 255.0


def rgb_to_hex(rgb):

    rgb = np.clip(
        rgb,
        0,
        1
    )

    rgb = np.round(
        rgb * 255
    ).astype(int)

    return "#{:02X}{:02X}{:02X}".format(
        rgb[0],
        rgb[1],
        rgb[2]
    )


# ============================================================
# 5. Lab ↔ LCH
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


def lch_to_lab(lch):

    L, C, H = lch

    H_rad = math.radians(
        H
    )

    a = (
        C
        *
        math.cos(
            H_rad
        )
    )

    b = (
        C
        *
        math.sin(
            H_rad
        )
    )

    return np.array([
        L,
        a,
        b
    ])


# ============================================================
# 6. HEX ↔ LCH
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


def lch_to_hex(lch):

    lab = lch_to_lab(
        lch
    )

    rgb = lab2rgb(
        lab.reshape(
            1,
            1,
            3
        )
    )[0, 0]

    return rgb_to_hex(
        rgb
    )


# ============================================================
# 7. Hue 圆形差值
# ============================================================

def hue_difference(
    hue,
    reference_hue
):

    return (
        (
            hue
            -
            reference_hue
            +
            180
        )
        %
        360
        -
        180
    )


# ============================================================
# 8. Cubic Bézier
# ============================================================

def cubic_bezier_value(
    t,
    p0,
    p1,
    p2,
    p3
):

    return (
        ((1 - t) ** 3)
        *
        p0

        +

        3
        *
        ((1 - t) ** 2)
        *
        t
        *
        p1

        +

        3
        *
        (1 - t)
        *
        (t ** 2)
        *
        p2

        +

        (t ** 3)
        *
        p3
    )


# ============================================================
# 9. Bézier easing
#
# 输入：
# t = 0 → 1
#
# 输出：
# 一个经过 Bézier 调整后的进度
#
# 起点固定 0
# 终点固定 1
# ============================================================

def bezier_ease(
    t,
    controls
):

    _, y1, _, y2 = (
        controls
    )

    return cubic_bezier_value(
        t,
        0,
        y1,
        y2,
        1
    )


# ============================================================
# 10. LCH 色域安全转换
#
# 如果 Chroma 过高，
# 自动降低 C
# ============================================================

def lch_to_hex_gamut_safe(
    L,
    C,
    H
):

    current_C = max(
        0,
        C
    )

    for _ in range(100):

        lab = lch_to_lab([
            L,
            current_C,
            H
        ])

        rgb = lab2rgb(
            lab.reshape(
                1,
                1,
                3
            )
        )[0, 0]

        if (
            np.all(rgb >= 0)
            and
            np.all(rgb <= 1)
        ):

            return (
                rgb_to_hex(
                    rgb
                ),
                np.array([
                    L,
                    current_C,
                    H
                ])
            )

        current_C *= 0.97

    return (
        rgb_to_hex(
            np.clip(
                rgb,
                0,
                1
            )
        ),

        np.array([
            L,
            current_C,
            H
        ])
    )


# ============================================================
# 11. 读取原始色板 LCH
# ============================================================

def get_original_lch():

    return np.array([
        hex_to_lch(
            color
        )
        for color
        in ORIGINAL_PALETTE
    ])


# ============================================================
# 12. 优化 Palette
# ============================================================

def optimize_palette():

    original_lch = (
        get_original_lch()
    )

    core = (
        original_lch[
            CORE_INDEX
        ]
    )

    core_L = core[0]
    core_C = core[1]
    core_H = core[2]

    optimized = []

    # ========================================================
    # Dark Side
    # C10 → C6
    # ========================================================

    dark_original = (
        original_lch[:5]
    )

    dark_start = (
        dark_original[0]
    )

    dark_count = len(
        dark_original
    )

    for i in range(
        dark_count
    ):

        t = (
            i
            /
            (
                dark_count - 1
            )
        )

        # ----------------------------------------------------
        # Lightness
        # ----------------------------------------------------

        t_L = bezier_ease(
            t,
            DARK_L_BEZIER
        )

        L = (
            dark_start[0]
            +
            (
                core_L
                -
                dark_start[0]
            )
            *
            t_L
        )

        # ----------------------------------------------------
        # Chroma
        # ----------------------------------------------------

        t_C = bezier_ease(
            t,
            DARK_C_BEZIER
        )

        C = (
            dark_start[1]
            +
            (
                core_C
                -
                dark_start[1]
            )
            *
            t_C
        )

        # ----------------------------------------------------
        # Hue
        # 轻微平滑
        # ----------------------------------------------------

        original_H = (
            dark_original[i, 2]
        )

        hue_delta = hue_difference(
            dark_start[2],
            core_H
        )

        t_H = bezier_ease(
            t,
            HUE_BEZIER
        )

        smooth_H = (
            core_H
            +
            hue_delta
            *
            (
                1
                -
                t_H
            )
        ) % 360

        H = (
            original_H
            +
            hue_difference(
                smooth_H,
                original_H
            )
            *
            HUE_SMOOTHING_STRENGTH
        ) % 360

        # C6 完全保留
        if i == CORE_INDEX:

            optimized.append(
                core.copy()
            )

        else:

            optimized.append(
                np.array([
                    L,
                    C,
                    H
                ])
            )

    # ========================================================
    # Light Side
    # C6 → C1
    # ========================================================

    light_original = (
        original_lch[4:]
    )

    light_end = (
        light_original[-1]
    )

    light_count = len(
        light_original
    )

    # C6 已经添加
    for i in range(
        1,
        light_count
    ):

        t = (
            i
            /
            (
                light_count - 1
            )
        )

        # ----------------------------------------------------
        # Lightness
        # ----------------------------------------------------

        t_L = bezier_ease(
            t,
            LIGHT_L_BEZIER
        )

        L = (
            core_L
            +
            (
                light_end[0]
                -
                core_L
            )
            *
            t_L
        )

        # ----------------------------------------------------
        # Chroma
        # ----------------------------------------------------

        t_C = bezier_ease(
            t,
            LIGHT_C_BEZIER
        )

        C = (
            core_C
            +
            (
                light_end[1]
                -
                core_C
            )
            *
            t_C
        )

        # ----------------------------------------------------
        # Hue
        # ----------------------------------------------------

        original_H = (
            light_original[i, 2]
        )

        hue_delta = hue_difference(
            light_end[2],
            core_H
        )

        t_H = bezier_ease(
            t,
            HUE_BEZIER
        )

        smooth_H = (
            core_H
            +
            hue_delta
            *
            t_H
        ) % 360

        H = (
            original_H
            +
            hue_difference(
                smooth_H,
                original_H
            )
            *
            HUE_SMOOTHING_STRENGTH
        ) % 360

        optimized.append(
            np.array([
                L,
                C,
                H
            ])
        )

    optimized = np.array(
        optimized
    )

    return (
        original_lch,
        optimized
    )


# ============================================================
# 13. 生成 HEX
# ============================================================

def create_final_dataframe():

    (
        original_lch,
        optimized_lch
    ) = optimize_palette()

    rows = []

    for i in range(
        len(
            LEVELS
        )
    ):

        if i == CORE_INDEX:

            optimized_hex = (
                ORIGINAL_PALETTE[i]
                .upper()
            )

            final_lch = (
                original_lch[i]
            )

        else:

            (
                optimized_hex,
                final_lch
            ) = (
                lch_to_hex_gamut_safe(
                    optimized_lch[i, 0],
                    optimized_lch[i, 1],
                    optimized_lch[i, 2]
                )
            )

        rows.append({

            "Level":
                LEVELS[i],

            "Original_HEX":
                ORIGINAL_PALETTE[i]
                .upper(),

            "Optimized_HEX":
                optimized_hex,

            "Original_L":
                round(
                    original_lch[i, 0],
                    4
                ),

            "Optimized_L":
                round(
                    final_lch[0],
                    4
                ),

            "Original_C":
                round(
                    original_lch[i, 1],
                    4
                ),

            "Optimized_C":
                round(
                    final_lch[1],
                    4
                ),

            "Original_H":
                round(
                    original_lch[i, 2],
                    4
                ),

            "Optimized_H":
                round(
                    final_lch[2] % 360,
                    4
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 14. Palette 对比图
# ============================================================

def create_palette_image(
    df,
    image_path
):

    fig, ax = plt.subplots(
        figsize=(
            15,
            6
        )
    )

    ax.set_xlim(
        0,
        10
    )

    ax.set_ylim(
        0,
        2
    )

    ax.axis(
        "off"
    )

    for i, row in df.iterrows():

        # Original
        ax.add_patch(
            plt.Rectangle(
                (
                    i,
                    1
                ),
                1,
                1,
                facecolor=row[
                    "Original_HEX"
                ]
            )
        )

        # Optimized
        ax.add_patch(
            plt.Rectangle(
                (
                    i,
                    0
                ),
                1,
                1,
                facecolor=row[
                    "Optimized_HEX"
                ]
            )
        )

        ax.text(
            i + 0.5,
            1.68,
            row[
                "Level"
            ],
            ha="center",
            va="center",
            fontsize=10,
            color="white"
            if i <= 4
            else
            "black"
        )

        ax.text(
            i + 0.5,
            1.48,
            row[
                "Original_HEX"
            ],
            ha="center",
            va="center",
            fontsize=8,
            color="white"
            if i <= 4
            else
            "black"
        )

        ax.text(
            i + 0.5,
            0.68,
            row[
                "Level"
            ],
            ha="center",
            va="center",
            fontsize=10,
            color="white"
            if i <= 4
            else
            "black"
        )

        ax.text(
            i + 0.5,
            0.48,
            row[
                "Optimized_HEX"
            ],
            ha="center",
            va="center",
            fontsize=8,
            color="white"
            if i <= 4
            else
            "black"
        )

    ax.text(
        -0.15,
        1.5,
        "Original",
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

    ax.text(
        -0.15,
        0.5,
        "Optimized",
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 15. 曲线对比
# ============================================================

def create_curve_comparison(
    df,
    original_column,
    optimized_column,
    title,
    ylabel,
    path
):

    x = np.arange(
        10
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            5
        )
    )

    ax.plot(
        x,
        df[
            original_column
        ],
        marker="o",
        label="Original"
    )

    ax.plot(
        x,
        df[
            optimized_column
        ],
        marker="o",
        label="Optimized"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        LEVELS
    )

    ax.set_title(
        title
    )

    ax.set_ylabel(
        ylabel
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=250
    )

    plt.close()


# ============================================================
# 16. Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "EXISTING BRAND PALETTE OPTIMISATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nCore Brand Colour:"
    )

    print(
        ORIGINAL_PALETTE[
            CORE_INDEX
        ]
    )

    (
        csv_path,
        palette_image_path,
        lightness_curve_path,
        chroma_curve_path,
        hue_curve_path
    ) = (
        get_next_output_paths()
    )

    df = (
        create_final_dataframe()
    )

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    create_palette_image(
        df,
        palette_image_path
    )

    create_curve_comparison(
        df,
        "Original_L",
        "Optimized_L",
        "Lightness Comparison",
        "L",
        lightness_curve_path
    )

    create_curve_comparison(
        df,
        "Original_C",
        "Optimized_C",
        "Chroma Comparison",
        "C",
        chroma_curve_path
    )

    create_curve_comparison(
        df,
        "Original_H",
        "Optimized_H",
        "Hue Comparison",
        "Hue",
        hue_curve_path
    )

    print(
        "\n"
    )

    print(
        df[
            [
                "Level",
                "Original_HEX",
                "Optimized_HEX",
                "Original_L",
                "Optimized_L",
                "Original_C",
                "Optimized_C",
                "Original_H",
                "Optimized_H",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nOutput:"
    )

    print(
        csv_path
    )

    print(
        palette_image_path
    )

    print(
        lightness_curve_path
    )

    print(
        chroma_curve_path
    )

    print(
        hue_curve_path
    )


if __name__ == "__main__":
    main()
