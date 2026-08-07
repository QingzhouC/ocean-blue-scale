from pathlib import Path
import math

import numpy as np
import pandas as pd

from skimage.color import rgb2lab, lab2rgb


# ============================================================
# 1. B-Design 蓝色标准色卡
# ============================================================

COLORS = [
    "#041538",  # C10
    "#06225A",  # C9
    "#09368F",  # C8
    "#0A4ACB",  # C7
    "#105CF4",  # C6 核心色
    "#3C7DFF",  # C5
    "#7AA6FF",  # C4
    "#A3C1FF",  # C3
    "#CDDDFD",  # C2
    "#DDE8FF",  # C1
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


# ============================================================
# 2. 输出设置
# ============================================================

OUTPUT_FOLDER = Path("results")
OUTPUT_FOLDER.mkdir(exist_ok=True)

LCH_CSV = OUTPUT_FOLDER / "bdesign_lch_points.csv"
FIT_CSV = OUTPUT_FOLDER / "bdesign_bezier_fit.csv"
CONTROL_CSV = OUTPUT_FOLDER / "bdesign_bezier_controls.csv"


# ============================================================
# 3. HEX → RGB
# ============================================================

def hex_to_rgb(hex_color):

    hex_color = hex_color.lstrip("#")

    return np.array([
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    ]) / 255.0


# ============================================================
# 4. RGB → Lab
# ============================================================

def rgb_to_lab(rgb):

    rgb_image = rgb.reshape(1, 1, 3)

    lab = rgb2lab(rgb_image)[0, 0]

    return lab


# ============================================================
# 5. Lab → LCH
# ============================================================

def lab_to_lch(lab):

    L, a, b = lab

    C = math.sqrt(
        a ** 2 +
        b ** 2
    )

    H = math.degrees(
        math.atan2(b, a)
    )

    if H < 0:
        H += 360

    return np.array([
        L,
        C,
        H
    ])


# ============================================================
# 6. HEX → LCH
# ============================================================

def hex_to_lch(hex_color):

    rgb = hex_to_rgb(hex_color)

    lab = rgb_to_lab(rgb)

    return lab_to_lch(lab)


# ============================================================
# 7. Hue 解包
#
# Hue 是圆形数据：
# 359° 和 1° 实际只差 2°
#
# 所以拟合前需要处理角度连续性。
# ============================================================

def unwrap_hue(hues):

    radians = np.radians(hues)

    unwrapped = np.unwrap(radians)

    return np.degrees(unwrapped)


# ============================================================
# 8. Cubic Bezier
#
# B(t)
# =
# (1-t)^3 P0
# +
# 3(1-t)^2 t P1
# +
# 3(1-t)t^2 P2
# +
# t^3 P3
# ============================================================

def cubic_bezier(t, p0, p1, p2, p3):

    return (
        ((1 - t) ** 3) * p0
        +
        3 * ((1 - t) ** 2) * t * p1
        +
        3 * (1 - t) * (t ** 2) * p2
        +
        (t ** 3) * p3
    )


# ============================================================
# 9. 拟合 Cubic Bezier 控制点
#
# 固定：
#
# P0 = C10
# P3 = C1
#
# 求：
#
# P1
# P2
# ============================================================

def fit_bezier(points):

    count = len(points)

    # 假设 10 个标准色在曲线上等距采样
    t_values = np.linspace(
        0,
        1,
        count
    )

    p0 = points[0]
    p3 = points[-1]

    A = []
    Y = []

    for t, target in zip(
        t_values,
        points
    ):

        b1 = (
            3
            * ((1 - t) ** 2)
            * t
        )

        b2 = (
            3
            * (1 - t)
            * (t ** 2)
        )

        known = (
            ((1 - t) ** 3) * p0
            +
            (t ** 3) * p3
        )

        A.append([
            b1,
            b2
        ])

        Y.append(
            target - known
        )

    A = np.array(A)

    Y = np.array(Y)

    solution, residuals, rank, singular_values = (
        np.linalg.lstsq(
            A,
            Y,
            rcond=None
        )
    )

    p1 = solution[0]
    p2 = solution[1]

    return (
        p0,
        p1,
        p2,
        p3,
        t_values
    )


# ============================================================
# 10. 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # HEX → LCH
    # --------------------------------------------------------

    lch_points = np.array([
        hex_to_lch(color)
        for color in COLORS
    ])

    # --------------------------------------------------------
    # Hue 单独进行 unwrap
    # --------------------------------------------------------

    original_hues = (
        lch_points[:, 2].copy()
    )

    unwrapped_hues = unwrap_hue(
        original_hues
    )

    lch_points[:, 2] = (
        unwrapped_hues
    )

    # --------------------------------------------------------
    # 保存原始 LCH 数据
    # --------------------------------------------------------

    lch_rows = []

    for i in range(
        len(COLORS)
    ):

        lch_rows.append({
            "Level": LEVELS[i],
            "HEX": COLORS[i],

            "L":
                round(
                    lch_points[i, 0],
                    6
                ),

            "C":
                round(
                    lch_points[i, 1],
                    6
                ),

            "H_original":
                round(
                    original_hues[i],
                    6
                ),

            "H_unwrapped":
                round(
                    unwrapped_hues[i],
                    6
                ),
        })

    lch_df = pd.DataFrame(
        lch_rows
    )

    lch_df.to_csv(
        LCH_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 拟合 Cubic Bezier
    # --------------------------------------------------------

    (
        p0,
        p1,
        p2,
        p3,
        t_values
    ) = fit_bezier(
        lch_points
    )

    # --------------------------------------------------------
    # 控制点输出
    # --------------------------------------------------------

    controls_df = pd.DataFrame([
        {
            "Point": "P0",
            "L": p0[0],
            "C": p0[1],
            "H": p0[2]
        },
        {
            "Point": "P1",
            "L": p1[0],
            "C": p1[1],
            "H": p1[2]
        },
        {
            "Point": "P2",
            "L": p2[0],
            "C": p2[1],
            "H": p2[2]
        },
        {
            "Point": "P3",
            "L": p3[0],
            "C": p3[1],
            "H": p3[2]
        },
    ])

    controls_df.to_csv(
        CONTROL_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 计算拟合值和误差
    # --------------------------------------------------------

    fit_rows = []

    total_error = 0

    for i, t in enumerate(
        t_values
    ):

        fitted = cubic_bezier(
            t,
            p0,
            p1,
            p2,
            p3
        )

        actual = (
            lch_points[i]
        )

        error_L = (
            fitted[0]
            - actual[0]
        )

        error_C = (
            fitted[1]
            - actual[1]
        )

        error_H = (
            fitted[2]
            - actual[2]
        )

        distance = math.sqrt(
            error_L ** 2
            +
            error_C ** 2
            +
            error_H ** 2
        )

        total_error += (
            distance
        )

        fit_rows.append({
            "Level":
                LEVELS[i],

            "HEX":
                COLORS[i],

            "t":
                round(
                    t,
                    6
                ),

            "Actual_L":
                round(
                    actual[0],
                    6
                ),

            "Fitted_L":
                round(
                    fitted[0],
                    6
                ),

            "Error_L":
                round(
                    error_L,
                    6
                ),

            "Actual_C":
                round(
                    actual[1],
                    6
                ),

            "Fitted_C":
                round(
                    fitted[1],
                    6
                ),

            "Error_C":
                round(
                    error_C,
                    6
                ),

            "Actual_H":
                round(
                    actual[2],
                    6
                ),

            "Fitted_H":
                round(
                    fitted[2],
                    6
                ),

            "Error_H":
                round(
                    error_H,
                    6
                ),

            "LCH_Distance":
                round(
                    distance,
                    6
                ),
        })

    fit_df = pd.DataFrame(
        fit_rows
    )

    fit_df.to_csv(
        FIT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 计算平均误差
    # --------------------------------------------------------

    mean_error = (
        total_error
        /
        len(COLORS)
    )

    max_error = (
        fit_df[
            "LCH_Distance"
        ].max()
    )

    # --------------------------------------------------------
    # 找 Chroma 峰值
    # --------------------------------------------------------

    chroma_values = (
        lch_points[:, 1]
    )

    peak_index = (
        np.argmax(
            chroma_values
        )
    )

    peak_level = (
        LEVELS[
            peak_index
        ]
    )

    peak_chroma = (
        chroma_values[
            peak_index
        ]
    )

    # --------------------------------------------------------
    # 打印结果
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("B-DESIGN LCH ANALYSIS")
    print("=" * 60)

    print("\n01 — Standard Colour LCH")
    print("-" * 60)

    print(
        lch_df.to_string(
            index=False
        )
    )

    print("\n")
    print("02 — Cubic Bezier Control Points")
    print("-" * 60)

    print(
        controls_df.to_string(
            index=False
        )
    )

    print("\n")
    print("03 — Chroma Peak")
    print("-" * 60)

    print(
        f"Peak Level : {peak_level}"
    )

    print(
        f"Peak Chroma: {peak_chroma:.6f}"
    )

    print("\n")
    print("04 — Fit Error")
    print("-" * 60)

    print(
        f"Mean LCH Distance: {mean_error:.6f}"
    )

    print(
        f"Max LCH Distance : {max_error:.6f}"
    )

    print("\n")
    print("05 — Output Files")
    print("-" * 60)

    print(
        LCH_CSV
    )

    print(
        CONTROL_CSV
    )

    print(
        FIT_CSV
    )

    print("\n")
    print("=" * 60)


if __name__ == "__main__":
    main()