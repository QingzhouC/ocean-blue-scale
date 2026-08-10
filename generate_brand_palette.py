from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.color import rgb2lab, lab2rgb


# ============================================================
# 1. 输入品牌色
# ============================================================

BRAND_COLOR = "#005aeb"

CORE_LEVEL = "C6"

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


# ============================================================
# 2. Hue-Adaptive 参数
# ============================================================

# Hue 影响范围
#
# 越小：
# 越接近附近色相自己的曲线
#
# 越大：
# 不同色相之间融合更多
#
HUE_SIGMA = 55.0


# Hue-Adaptive 强度
#
# 0.0 = 完全使用多组色卡的通用模型
# 1.0 = 完全根据 Hue 自适应
#
# 建议先使用 0.60
ADAPT_STRENGTH = 0.60


# 如果品牌色 Hue 距离某个训练核心色非常接近，
# 直接使用该训练色卡的模型。
#
# 这样 #105CF4 可以最大程度还原原蓝色色阶。
ANCHOR_HUE_TOLERANCE = 1.0


# ============================================================
# 3. 训练色卡
#
# 全部按照 C10 → C1
# ============================================================

REFERENCE_PALETTES = {

    "blue": [
        "#001A4D",  # C10
        "#062567",  # C9
        "#0A399C",  # C8
        "#064FD4",  # C7
        "#105CF4",  # C6
        "#1174FE",  # C5
        "#6098FF",  # C4
        "#9FC0FF",  # C3
        "#D0E0FF",  # C2
        "#E8F0FF",  # C1
    ],

    "red": [
        "#4D0006",
        "#66080E",
        "#9B141D",
        "#D3202D",
        "#F42738",
        "#FF4858",
        "#FF7B84",
        "#FFADB3",
        "#FFD5D8",
        "#FFE8EA",
    ],

    "pink": [
        "#4D0040",
        "#660255",
        "#96057E",
        "#C909A9",
        "#E60FC2",
        "#EB47CE",
        "#F47DDD",
        "#FBAEED",
        "#FFD5F7",
        "#FFE8FB",
    ],

    "green": [
        "#004D12",
        "#035F17",
        "#01801F",
        "#00A426",
        "#00B82B",
        "#33C44D",
        "#6DD67D",
        "#A3EAB0",
        "#D1F9D9",
        "#E8FFED",
    ],

    "yellow": [
        "#4D4000",
        "#685702",
        "#9C8202",
        "#D3B000",
        "#F4CB00",
        "#F4D339",
        "#F8DF6F",
        "#FCECA4",
        "#FFF6D1",
        "#FFFBE8",
    ],
}

# ============================================================
# 4. 输出
# ============================================================

OUTPUT_FOLDER = Path("results")
OUTPUT_FOLDER.mkdir(exist_ok=True)

IMAGE_FOLDER = OUTPUT_FOLDER / "images"
IMAGE_FOLDER.mkdir(exist_ok=True)


def get_next_output_paths():

    existing = sorted(
        OUTPUT_FOLDER.glob(
            "brand_palette*.csv"
        )
    )

    max_num = 0

    for f in existing:

        stem = f.stem

        try:

            num = int(
                stem.replace(
                    "brand_palette",
                    ""
                )
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
        f"brand_palette"
        f"{next_num:03d}"
    )

    csv_path = (
        OUTPUT_FOLDER
        /
        f"{name}.csv"
    )

    image_path = (
        IMAGE_FOLDER
        /
        f"{name}.png"
    )

    return (
        csv_path,
        image_path
    )


# ============================================================
# 5. HEX ↔ RGB
# ============================================================

def hex_to_rgb(hex_color):

    hex_color = (
        hex_color
        .strip()
        .lstrip("#")
    )

    return np.array([
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    ]) / 255.0


def rgb_to_hex(rgb):

    rgb = np.clip(
        rgb,
        0,
        1
    )

    values = np.round(
        rgb * 255
    ).astype(int)

    return "#{:02X}{:02X}{:02X}".format(
        values[0],
        values[1],
        values[2]
    )


# ============================================================
# 6. Lab ↔ LCH
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
# 7. HEX → LCH
# ============================================================

def hex_to_lch(hex_color):

    rgb = hex_to_rgb(
        hex_color
    )

    rgb_image = rgb.reshape(
        1,
        1,
        3
    )

    lab = rgb2lab(
        rgb_image
    )[0, 0]

    return lab_to_lch(
        lab
    )


# ============================================================
# 8. LCH → RGB / HEX
# ============================================================

def lab_to_raw_rgb(lab):

    lab_image = lab.reshape(
        1,
        1,
        3
    )

    return lab2rgb(
        lab_image
    )[0, 0]


def lch_to_hex_gamut_safe(lch):

    L, C, H = lch

    current_C = C

    for _ in range(100):

        lab = lch_to_lab([
            L,
            current_C,
            H
        ])

        rgb = lab_to_raw_rgb(
            lab
        )

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

    rgb = np.clip(
        rgb,
        0,
        1
    )

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


# ============================================================
# 9. Hue 圆形计算
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


def hue_distance(
    hue1,
    hue2
):

    return abs(
        hue_difference(
            hue1,
            hue2
        )
    )


# ============================================================
# 10. Cubic Bezier
# ============================================================

def cubic_bezier(
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
# 11. Cubic Bezier 拟合
# ============================================================

def fit_scalar_bezier(
    values,
    t_values
):

    values = np.array(
        values,
        dtype=float
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
            ((1 - t) ** 2)
            *
            t
        )

        b2 = (
            3
            *
            (1 - t)
            *
            (t ** 2)
        )

        known = (
            ((1 - t) ** 3)
            *
            p0

            +

            (t ** 3)
            *
            p3
        )

        A.append([
            b1,
            b2
        ])

        Y.append(
            value
            -
            known
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

    return (
        float(p0),
        float(solution[0]),
        float(solution[1]),
        float(p3)
    )


# ============================================================
# 12. 单套 Palette 倒推 Bezier
#
# 完全保留你第一版的逻辑
# ============================================================

def build_single_palette_model(
    colors
):

    reference_lch = np.array([
        hex_to_lch(
            color
        )
        for color
        in colors
    ])

    # C6 = index 4
    core = (
        reference_lch[4]
    )

    core_L = core[0]
    core_C = core[1]
    core_H = core[2]

    # ========================================================
    # Dark
    # ========================================================

    dark = (
        reference_lch[:5]
    )

    dark_L = (
        dark[:, 0]
        /
        core_L
    )

    dark_C = (
        dark[:, 1]
        /
        core_C
    )

    dark_H = np.array([
        hue_difference(
            hue,
            core_H
        )

        for hue
        in dark[:, 2]
    ])

    # ========================================================
    # Light
    # ========================================================

    light = (
        reference_lch[4:]
    )

    light_L = (
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

    light_C = (
        light[:, 1]
        /
        core_C
    )

    light_H = np.array([
        hue_difference(
            hue,
            core_H
        )

        for hue
        in light[:, 2]
    ])

    dark_t = np.linspace(
        0,
        1,
        5
    )

    light_t = np.linspace(
        0,
        1,
        6
    )

    return {

        "core_H":
            float(
                core_H
            ),

        "dark_L":
            fit_scalar_bezier(
                dark_L,
                dark_t
            ),

        "dark_C":
            fit_scalar_bezier(
                dark_C,
                dark_t
            ),

        "dark_H":
            fit_scalar_bezier(
                dark_H,
                dark_t
            ),

        "light_L":
            fit_scalar_bezier(
                light_L,
                light_t
            ),

        "light_C":
            fit_scalar_bezier(
                light_C,
                light_t
            ),

        "light_H":
            fit_scalar_bezier(
                light_H,
                light_t
            ),
    }


# ============================================================
# 13. 建立所有 Hue Anchor 模型
# ============================================================

MODEL_KEYS = [
    "dark_L",
    "dark_C",
    "dark_H",
    "light_L",
    "light_C",
    "light_H",
]


def build_anchor_models():

    models = {}

    for (
        name,
        colors
    ) in (
        REFERENCE_PALETTES.items()
    ):

        models[name] = (
            build_single_palette_model(
                colors
            )
        )

    return models


# ============================================================
# 14. 建立通用 Median 模型
#
# 作为 Hue Adaptive 的稳定底座。
# ============================================================

def build_universal_model(
    anchor_models
):

    universal = {}

    for key in MODEL_KEYS:

        controls = np.array([
            model[key]
            for model
            in anchor_models.values()
        ])

        universal[key] = tuple(
            np.median(
                controls,
                axis=0
            )
        )

    return universal


# ============================================================
# 15. Hue 权重
#
# 每套模型的权重由：
#
# 当前 Brand Hue
# 和
# 参考 Palette C6 Hue
#
# 的圆形距离决定。
#
# 不是硬切换。
# ============================================================

def calculate_hue_weights(
    brand_H,
    anchor_models
):

    distances = {}

    # --------------------------------------------------------
    # 先检查是否接近某个 Anchor
    # --------------------------------------------------------

    for (
        name,
        model
    ) in anchor_models.items():

        distance = hue_distance(
            brand_H,
            model[
                "core_H"
            ]
        )

        distances[name] = (
            distance
        )

        if (
            distance
            <=
            ANCHOR_HUE_TOLERANCE
        ):

            weights = {
                n: 0.0
                for n
                in anchor_models
            }

            weights[name] = 1.0

            return (
                weights,
                True
            )

    # --------------------------------------------------------
    # Gaussian 连续权重
    # --------------------------------------------------------

    raw = {}

    for (
        name,
        distance
    ) in distances.items():

        raw[name] = math.exp(

            -

            (
                distance ** 2
            )

            /

            (
                2
                *
                HUE_SIGMA ** 2
            )
        )

    total = sum(
        raw.values()
    )

    if total == 0:

        count = len(
            raw
        )

        return (
            {
                name:
                    1 / count
                for name
                in raw
            },
            False
        )

    weights = {

        name:
            value / total

        for (
            name,
            value
        ) in raw.items()
    }

    return (
        weights,
        False
    )


# ============================================================
# 16. 根据 Hue 连续生成 Bezier 控制点
# ============================================================

def build_reference_model(
    brand_color
):

    brand_lch = (
        hex_to_lch(
            brand_color
        )
    )

    brand_H = (
        brand_lch[2]
    )

    anchor_models = (
        build_anchor_models()
    )

    universal_model = (
        build_universal_model(
            anchor_models
        )
    )

    (
        weights,
        exact_anchor
    ) = (
        calculate_hue_weights(
            brand_H,
            anchor_models
        )
    )

    # ========================================================
    # 如果正好命中一个 Anchor
    #
    # 直接使用该 Palette 倒推模型
    # ========================================================

    if exact_anchor:

        anchor_name = max(
            weights,
            key=weights.get
        )

        selected = (
            anchor_models[
                anchor_name
            ]
        )

        model = {
            key:
                selected[key]
            for key
            in MODEL_KEYS
        }

        print(
            "\nExact Hue Anchor:"
        )

        print(
            anchor_name
        )

        return model

    # ========================================================
    # Hue Adaptive Model
    # ========================================================

    local_model = {}

    for key in MODEL_KEYS:

        blended = np.zeros(
            4,
            dtype=float
        )

        for (
            name,
            anchor_model
        ) in (
            anchor_models.items()
        ):

            blended += (
                np.array(
                    anchor_model[key]
                )
                *
                weights[name]
            )

        local_model[key] = tuple(
            blended
        )

    # ========================================================
    # Adaptive + Universal
    #
    # 避免只有 5 个训练 Hue 时曲线过度摆动
    # ========================================================

    model = {}

    for key in MODEL_KEYS:

        local = np.array(
            local_model[key]
        )

        universal = np.array(
            universal_model[key]
        )

        final = (
            ADAPT_STRENGTH
            *
            local

            +

            (
                1
                -
                ADAPT_STRENGTH
            )
            *
            universal
        )

        model[key] = tuple(
            final
        )

    # ========================================================
    # 输出当前 Hue 权重
    # ========================================================

    print(
        "\nHue-Adaptive Model"
    )

    print(
        "-" * 60
    )

    print(
        f"Brand Hue: "
        f"{brand_H:.2f}°"
    )

    print(
        "\nHue Weights:"
    )

    sorted_weights = sorted(
        weights.items(),
        key=lambda x:
            x[1],
        reverse=True
    )

    for (
        name,
        weight
    ) in sorted_weights:

        print(
            f"{name:12s}"
            f" : "
            f"{weight:.2%}"
        )

    print(
        "\nAdaptive Strength:"
    )

    print(
        ADAPT_STRENGTH
    )

    print(
        "\nGenerated Bezier:"
    )

    for key in MODEL_KEYS:

        print(
            key,
            "=",
            np.round(
                model[key],
                4
            )
        )

    return model


# ============================================================
# 17. 根据品牌色生成 Palette
#
# 这一部分保持你喜欢的第一版逻辑。
# ============================================================

def generate_palette(
    brand_color
):

    model = (
        build_reference_model(
            brand_color
        )
    )

    brand_lch = (
        hex_to_lch(
            brand_color
        )
    )

    brand_L = (
        brand_lch[0]
    )

    brand_C = (
        brand_lch[1]
    )

    brand_H = (
        brand_lch[2]
    )

    rows = []

    # ========================================================
    # Dark C10 → C6
    # ========================================================

    dark_levels = [
        "C10",
        "C9",
        "C8",
        "C7",
        "C6",
    ]

    dark_t_values = np.linspace(
        0,
        1,
        5
    )

    for (
        level,
        t
    ) in zip(
        dark_levels,
        dark_t_values
    ):

        L_ratio = cubic_bezier(
            t,
            *model[
                "dark_L"
            ]
        )

        C_ratio = cubic_bezier(
            t,
            *model[
                "dark_C"
            ]
        )

        H_delta = cubic_bezier(
            t,
            *model[
                "dark_H"
            ]
        )

        L = (
            brand_L
            *
            L_ratio
        )

        C = (
            brand_C
            *
            C_ratio
        )

        H = (
            brand_H
            +
            H_delta
        ) % 360

        if level == "C6":

            final_hex = (
                brand_color
                .upper()
            )

            final_lch = (
                brand_lch
                .copy()
            )

        else:

            (
                final_hex,
                final_lch
            ) = (
                lch_to_hex_gamut_safe([
                    L,
                    C,
                    H
                ])
            )

        rows.append({

            "Level":
                level,

            "HEX":
                final_hex,

            "L":
                round(
                    final_lch[0],
                    4
                ),

            "C":
                round(
                    final_lch[1],
                    4
                ),

            "H":
                round(
                    final_lch[2]
                    %
                    360,
                    4
                ),

        })

    # ========================================================
    # Light C6 → C1
    # ========================================================

    light_levels = [
        "C6",
        "C5",
        "C4",
        "C3",
        "C2",
        "C1",
    ]

    light_t_values = np.linspace(
        0,
        1,
        6
    )

    for (
        level,
        t
    ) in zip(
        light_levels[1:],
        light_t_values[1:]
    ):

        L_progress = cubic_bezier(
            t,
            *model[
                "light_L"
            ]
        )

        C_ratio = cubic_bezier(
            t,
            *model[
                "light_C"
            ]
        )

        H_delta = cubic_bezier(
            t,
            *model[
                "light_H"
            ]
        )

        L = (
            brand_L

            +

            L_progress
            *
            (
                100
                -
                brand_L
            )
        )

        C = (
            brand_C
            *
            C_ratio
        )

        H = (
            brand_H
            +
            H_delta
        ) % 360

        (
            final_hex,
            final_lch
        ) = (
            lch_to_hex_gamut_safe([
                L,
                C,
                H
            ])
        )

        rows.append({

            "Level":
                level,

            "HEX":
                final_hex,

            "L":
                round(
                    final_lch[0],
                    4
                ),

            "C":
                round(
                    final_lch[1],
                    4
                ),

            "H":
                round(
                    final_lch[2]
                    %
                    360,
                    4
                ),

        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 18. 生成色卡图片
# ============================================================

def create_palette_image(
    df,
    image_path
):

    fig, ax = plt.subplots(
        figsize=(
            15,
            5
        )
    )

    ax.set_xlim(
        0,
        10
    )

    ax.set_ylim(
        0,
        1
    )

    ax.axis(
        "off"
    )

    for i, row in df.iterrows():

        color = (
            row[
                "HEX"
            ]
        )

        rgb = (
            hex_to_rgb(
                color
            )
        )

        luminance = (
            0.2126
            *
            rgb[0]

            +

            0.7152
            *
            rgb[1]

            +

            0.0722
            *
            rgb[2]
        )

        text_color = (
            "white"
            if luminance < 0.55
            else
            "black"
        )

        ax.add_patch(

            plt.Rectangle(

                (
                    i,
                    0
                ),

                1,
                1,

                facecolor=color

            )

        )

        ax.text(
            i + 0.5,
            0.61,
            row["Level"],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=text_color
        )

        ax.text(
            i + 0.5,
            0.47,
            color,
            ha="center",
            va="center",
            fontsize=9,
            color=text_color
        )

        ax.text(
            i + 0.5,
            0.33,
            f"L {row['L']:.1f}",
            ha="center",
            va="center",
            fontsize=7,
            color=text_color
        )

        ax.text(
            i + 0.5,
            0.25,
            f"C {row['C']:.1f}",
            ha="center",
            va="center",
            fontsize=7,
            color=text_color
        )

        ax.text(
            i + 0.5,
            0.17,
            f"H {row['H']:.1f}",
            ha="center",
            va="center",
            fontsize=7,
            color=text_color
        )

    plt.tight_layout()

    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 19. Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "HUE-ADAPTIVE BRAND PALETTE GENERATOR"
    )

    print(
        "=" * 70
    )

    print(
        "\nInput Brand Colour:"
    )

    print(
        BRAND_COLOR
    )

    brand_lch = (
        hex_to_lch(
            BRAND_COLOR
        )
    )

    print(
        "\nBrand LCH:"
    )

    print(
        f"L = "
        f"{brand_lch[0]:.4f}"
    )

    print(
        f"C = "
        f"{brand_lch[1]:.4f}"
    )

    print(
        f"H = "
        f"{brand_lch[2]:.4f}"
    )

    csv_path, image_path = (
        get_next_output_paths()
    )

    df = generate_palette(
        BRAND_COLOR
    )

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    create_palette_image(
        df,
        image_path
    )

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "GENERATED PALETTE"
    )

    print(
        "=" * 70
    )

    print(
        df.to_string(
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
        image_path
    )


if __name__ == "__main__":
    main()