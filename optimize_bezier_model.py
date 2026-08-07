from pathlib import Path
import math
import json

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from skimage.color import rgb2lab


# ============================================================
# 1. 基础设置
# ============================================================

OUTPUT_FOLDER = Path("results")
OUTPUT_FOLDER.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_FOLDER / "optimized_bezier_model.json"
COMPARISON_PATH = OUTPUT_FOLDER / "optimization_comparison.csv"


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
# 2. 训练 Palette
#
# 全部统一为 C10 → C1
# ============================================================

TRAINING_PALETTES = {

    "blue": [
        "#041538",
        "#06225A",
        "#09368F",
        "#0A4ACB",
        "#105CF4",
        "#3C7DFF",
        "#7AA6FF",
        "#A3C1FF",
        "#CDDDFD",
        "#DDE8FF",
    ],

    "orange_red": [
        "#320C04",
        "#4A1105",
        "#721A06",
        "#9F260C",
        "#C62F0E",
        "#E54E2C",
        "#FF7F64",
        "#FFA997",
        "#FFD7CF",
        "#FFE2DD",
    ],

    "pink": [
        "#30081A",
        "#570D30",
        "#7F1346",
        "#991755",
        "#C61E6E",
        "#E4428F",
        "#FE74B6",
        "#FEA3CE",
        "#FFCFE6",
        "#FFDDEC",
    ],

    "green": [
        "#003022",
        "#003526",
        "#004531",
        "#005E43",
        "#007855",
        "#009267",
        "#0DBD89",
        "#2FDDAA",
        "#99F3D9",
        "#CAFFEF",
    ],

    "yellow": [
        "#4D3300",
        "#744E01",
        "#996600",
        "#C08002",
        "#E69900",
        "#FFBB33",
        "#FFCC66",
        "#FFDD99",
        "#FFE9BD",
        "#FFF2D8",
    ],
}


# ============================================================
# 3. Loss 权重
#
# Lightness 最重要
# Chroma 第二
# Hue 只允许小幅校准
# ============================================================

WEIGHT_L = 1.00
WEIGHT_C = 0.80
WEIGHT_H = 0.25


# ============================================================
# 4. 正则强度
#
# 数字越高：
# 越接近第一版
#
# 数字越低：
# 优化器自由度越高
# ============================================================

LAMBDA_REG = 0.18


# ============================================================
# 5. 误差归一化尺度
#
# 防止 L / C / H 单位不同导致某一项占据 Loss
# ============================================================

L_SCALE = 10.0
C_SCALE = 15.0
H_SCALE = 10.0


# ============================================================
# 6. HEX → RGB
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


# ============================================================
# 7. Lab → LCH
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
# 8. HEX → LCH
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
# 9. Hue 差值
#
# 防止 359° 和 1° 被认为相差 358°
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
        ((1 - t) ** 3) * p0

        +

        3
        * ((1 - t) ** 2)
        * t
        * p1

        +

        3
        * (1 - t)
        * (t ** 2)
        * p2

        +

        (t ** 3)
        * p3
    )


# ============================================================
# 11. 拟合第一版 Blue Baseline
#
# 这部分就是你第一版的原始逻辑。
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
            value - known
        )

    solution, _, _, _ = (
        np.linalg.lstsq(
            np.array(A),
            np.array(Y),
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
# 12. 第一版蓝色模型
# ============================================================

def build_blue_baseline():

    colors = TRAINING_PALETTES[
        "blue"
    ]

    lch = np.array([
        hex_to_lch(color)
        for color
        in colors
    ])

    # C6 index = 4
    core = lch[4]

    core_L = core[0]
    core_C = core[1]
    core_H = core[2]

    # --------------------------------------------------------
    # Dark C10 → C6
    # --------------------------------------------------------

    dark = lch[:5]

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

    # --------------------------------------------------------
    # Light C6 → C1
    # --------------------------------------------------------

    light = lch[4:]

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
# 13. 将模型转换成 12 个优化变量
#
# 每条曲线：
#
# P0 固定
# P1 优化
# P2 优化
# P3 固定
#
# 6 × 2 = 12
# ============================================================

MODEL_KEYS = [
    "dark_L",
    "dark_C",
    "dark_H",
    "light_L",
    "light_C",
    "light_H",
]


def model_to_parameters(
    model
):

    parameters = []

    for key in MODEL_KEYS:

        controls = model[key]

        parameters.append(
            controls[1]
        )

        parameters.append(
            controls[2]
        )

    return np.array(
        parameters,
        dtype=float
    )


# ============================================================
# 14. 12 参数 → 完整模型
# ============================================================

def parameters_to_model(
    parameters,
    baseline
):

    model = {}

    index = 0

    for key in MODEL_KEYS:

        p0 = baseline[key][0]
        p3 = baseline[key][3]

        p1 = parameters[index]
        p2 = parameters[index + 1]

        model[key] = (
            float(p0),
            float(p1),
            float(p2),
            float(p3)
        )

        index += 2

    return model


# ============================================================
# 15. 用模型预测某一套 Palette 的 LCH
#
# 注意：
# 优化阶段不转 RGB。
#
# 直接在 LCH 空间比较，
# 避免 RGB clipping 干扰数学模型。
# ============================================================

def predict_palette_lch(
    core_lch,
    model
):

    brand_L = core_lch[0]
    brand_C = core_lch[1]
    brand_H = core_lch[2]

    predictions = []

    # ========================================================
    # Dark C10 → C6
    # ========================================================

    dark_t = np.linspace(
        0,
        1,
        5
    )

    for t in dark_t:

        L_ratio = cubic_bezier(
            t,
            *model["dark_L"]
        )

        C_ratio = cubic_bezier(
            t,
            *model["dark_C"]
        )

        H_delta = cubic_bezier(
            t,
            *model["dark_H"]
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

        predictions.append([
            L,
            C,
            H
        ])

    # ========================================================
    # Light C5 → C1
    # ========================================================

    light_t = np.linspace(
        0,
        1,
        6
    )

    # index 0 = C6
    # 已经加入
    for t in light_t[1:]:

        L_progress = cubic_bezier(
            t,
            *model["light_L"]
        )

        C_ratio = cubic_bezier(
            t,
            *model["light_C"]
        )

        H_delta = cubic_bezier(
            t,
            *model["light_H"]
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

        predictions.append([
            L,
            C,
            H
        ])

    return np.array(
        predictions
    )


# ============================================================
# 16. 将所有真实 Palette 转 LCH
# ============================================================

def prepare_training_data():

    training = {}

    for (
        name,
        colors
    ) in TRAINING_PALETTES.items():

        actual_lch = np.array([
            hex_to_lch(
                color
            )
            for color
            in colors
        ])

        training[name] = {
            "colors":
                colors,

            "actual_lch":
                actual_lch,

            # C6
            "core_lch":
                actual_lch[4],
        }

    return training


# ============================================================
# 17. 单个颜色的误差
# ============================================================

def colour_error(
    actual,
    predicted
):

    delta_L = (
        predicted[0]
        -
        actual[0]
    )

    delta_C = (
        predicted[1]
        -
        actual[1]
    )

    delta_H = hue_difference(
        predicted[2],
        actual[2]
    )

    error = (
        WEIGHT_L
        *
        (
            delta_L
            /
            L_SCALE
        ) ** 2

        +

        WEIGHT_C
        *
        (
            delta_C
            /
            C_SCALE
        ) ** 2

        +

        WEIGHT_H
        *
        (
            delta_H
            /
            H_SCALE
        ) ** 2
    )

    return (
        error,
        delta_L,
        delta_C,
        delta_H
    )


# ============================================================
# 18. 总 Palette Loss
# ============================================================

def reconstruction_loss(
    model,
    training
):

    total = 0.0
    count = 0

    for data in training.values():

        predicted = predict_palette_lch(
            data["core_lch"],
            model
        )

        actual = data[
            "actual_lch"
        ]

        # C6 本身固定，不需要参与优化
        for i in range(10):

            if i == 4:
                continue

            error, _, _, _ = (
                colour_error(
                    actual[i],
                    predicted[i]
                )
            )

            total += error
            count += 1

    return (
        total
        /
        max(
            count,
            1
        )
    )


# ============================================================
# 19. Optimization Objective
#
# Reconstruction Error
#
# +
#
# Regularization
#
# 防止偏离第一版过多
# ============================================================

def objective(
    parameters,
    baseline,
    baseline_parameters,
    training
):

    model = parameters_to_model(
        parameters,
        baseline
    )

    reconstruction = (
        reconstruction_loss(
            model,
            training
        )
    )

    # --------------------------------------------------------
    # 参数相对于第一版的变化
    #
    # 为避免不同参数尺度不同，
    # 使用相对尺度进行标准化。
    # --------------------------------------------------------

    scale = np.maximum(
        np.abs(
            baseline_parameters
        ),
        1.0
    )

    parameter_change = (
        (
            parameters
            -
            baseline_parameters
        )
        /
        scale
    )

    regularization = np.mean(
        parameter_change ** 2
    )

    return (
        reconstruction
        +
        LAMBDA_REG
        *
        regularization
    )


# ============================================================
# 20. 保存 Before / After 对比
# ============================================================

def create_comparison(
    baseline,
    optimized,
    training
):

    rows = []

    for (
        palette_name,
        data
    ) in training.items():

        actual = data[
            "actual_lch"
        ]

        baseline_pred = (
            predict_palette_lch(
                data["core_lch"],
                baseline
            )
        )

        optimized_pred = (
            predict_palette_lch(
                data["core_lch"],
                optimized
            )
        )

        for i, level in enumerate(
            LEVELS
        ):

            if i == 4:

                baseline_error = 0
                optimized_error = 0

            else:

                (
                    baseline_error,
                    _,
                    _,
                    _
                ) = colour_error(
                    actual[i],
                    baseline_pred[i]
                )

                (
                    optimized_error,
                    _,
                    _,
                    _
                ) = colour_error(
                    actual[i],
                    optimized_pred[i]
                )

            rows.append({

                "Palette":
                    palette_name,

                "Level":
                    level,

                "Actual_L":
                    round(
                        actual[i, 0],
                        4
                    ),

                "Actual_C":
                    round(
                        actual[i, 1],
                        4
                    ),

                "Actual_H":
                    round(
                        actual[i, 2],
                        4
                    ),

                "Baseline_L":
                    round(
                        baseline_pred[i, 0],
                        4
                    ),

                "Baseline_C":
                    round(
                        baseline_pred[i, 1],
                        4
                    ),

                "Baseline_H":
                    round(
                        baseline_pred[i, 2],
                        4
                    ),

                "Optimized_L":
                    round(
                        optimized_pred[i, 0],
                        4
                    ),

                "Optimized_C":
                    round(
                        optimized_pred[i, 1],
                        4
                    ),

                "Optimized_H":
                    round(
                        optimized_pred[i, 2],
                        4
                    ),

                "Baseline_Error":
                    round(
                        baseline_error,
                        6
                    ),

                "Optimized_Error":
                    round(
                        optimized_error,
                        6
                    ),
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 21. 主程序
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "BEZIER MODEL OPTIMIZATION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # 第一版作为起点
    # --------------------------------------------------------

    baseline = (
        build_blue_baseline()
    )

    baseline_parameters = (
        model_to_parameters(
            baseline
        )
    )

    training = (
        prepare_training_data()
    )

    before_loss = (
        reconstruction_loss(
            baseline,
            training
        )
    )

    print(
        "\nBaseline reconstruction loss:"
    )

    print(
        f"{before_loss:.8f}"
    )

    # --------------------------------------------------------
    # 优化
    # --------------------------------------------------------

    result = minimize(

        objective,

        baseline_parameters,

        args=(
            baseline,
            baseline_parameters,
            training
        ),

        method="L-BFGS-B",

        options={
            "maxiter": 5000,
            "ftol": 1e-12,
            "gtol": 1e-8,
        }
    )

    optimized_parameters = (
        result.x
    )

    optimized_model = (
        parameters_to_model(
            optimized_parameters,
            baseline
        )
    )

    after_loss = (
        reconstruction_loss(
            optimized_model,
            training
        )
    )

    # --------------------------------------------------------
    # 输出结果
    # --------------------------------------------------------

    print(
        "\nOptimization success:"
    )

    print(
        result.success
    )

    print(
        "\nMessage:"
    )

    print(
        result.message
    )

    print(
        "\nBaseline Loss:"
    )

    print(
        f"{before_loss:.8f}"
    )

    print(
        "\nOptimized Loss:"
    )

    print(
        f"{after_loss:.8f}"
    )

    improvement = (
        (
            before_loss
            -
            after_loss
        )
        /
        before_loss
        *
        100
    )

    print(
        "\nImprovement:"
    )

    print(
        f"{improvement:.2f}%"
    )

    # ========================================================
    # 控制点
    # ========================================================

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "OPTIMIZED CONTROL POINTS"
    )

    print(
        "=" * 70
    )

    for key in MODEL_KEYS:

        print(
            f"\n{key}:"
        )

        print(
            "Baseline :",
            np.round(
                baseline[key],
                6
            )
        )

        print(
            "Optimized:",
            np.round(
                optimized_model[key],
                6
            )
        )

    # ========================================================
    # 保存 JSON
    # ========================================================

    serializable_model = {

        key: [
            float(v)
            for v
            in optimized_model[key]
        ]

        for key in MODEL_KEYS
    }

    serializable_model[
        "baseline_loss"
    ] = float(
        before_loss
    )

    serializable_model[
        "optimized_loss"
    ] = float(
        after_loss
    )

    serializable_model[
        "improvement_percent"
    ] = float(
        improvement
    )

    serializable_model[
        "lambda_regularization"
    ] = LAMBDA_REG

    serializable_model[
        "weights"
    ] = {
        "L": WEIGHT_L,
        "C": WEIGHT_C,
        "H": WEIGHT_H,
    }

    with open(
        MODEL_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            serializable_model,
            f,
            indent=4
        )

    # ========================================================
    # 对比 CSV
    # ========================================================

    comparison = create_comparison(
        baseline,
        optimized_model,
        training
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "OUTPUT"
    )

    print(
        "=" * 70
    )

    print(
        MODEL_PATH
    )

    print(
        COMPARISON_PATH
    )


if __name__ == "__main__":
    main()