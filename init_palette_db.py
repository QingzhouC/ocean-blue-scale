"""
初始化训练色卡数据库 (SQLite)

运行此脚本会创建 / 重建 palettes.db，
将所有参考色卡写入 reference_palettes 表。

用法:
    python init_palette_db.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "palettes.db"

# ============================================================
# 训练色卡数据
#
# 全部按照 C10 → C1
# ============================================================

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

    "cyan": [
        "#00404D",
        "#045667",
        "#08819A",
        "#08AECE",
        "#0FC8ED",
        "#4AD0EE",
        "#82DDF1",
        "#B2EBF7",
        "#D6F6FD",
        "#E8FBFF",
    ],

    "orange": [
        "#4D2600",
        "#753C02",
        "#A15406",
        "#C96B0C",
        "#F48210",
        "#F5983B",
        "#F7AE65",
        "#FAC591",
        "#FCDDBD",
        "#FFF4E8",
    ],

    "light_blue": [
        "#00264D",
        "#043568",
        "#07529C",
        "#0171D4",
        "#1082F4",
        "#1C95FA",
        "#65B1FE",
        "#A2CFFF",
        "#D1E8FF",
        "#E8F4FF",
    ],

    "purple_blue": [
        "#00044D",
        "#090F75",
        "#1A20A1",
        "#3038C9",
        "#4E56F4",
        "#6C73F5",
        "#8B90F7",
        "#AAAEFA",
        "#CACCFC",
        "#E8E9FF",
    ],

    "purple": [
        "#33004D",
        "#460268",
        "#6A069C",
        "#920AD4",
        "#A810F4",
        "#B940F8",
        "#CF75FD",
        "#E4AAFF",
        "#F2D4FF",
        "#F7E8FF",
    ],
}


def init_database():
    """创建 / 重建 palettes.db"""

    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE reference_palettes (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT    NOT NULL,
            level     TEXT    NOT NULL,
            hex       TEXT    NOT NULL,
            position  INTEGER NOT NULL,
            UNIQUE(name, level)
        )
        """
    )

    rows = []

    for name, colors in REFERENCE_PALETTES.items():

        for position, (level, hex_color) in enumerate(
            zip(LEVELS, colors)
        ):

            rows.append(
                (name, level, hex_color, position)
            )

    cursor.executemany(
        """
        INSERT INTO reference_palettes
            (name, level, hex, position)
        VALUES
            (?, ?, ?, ?)
        """,
        rows,
    )

    conn.commit()

    # --------------------------------------------------------
    # 打印摘要
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT name, COUNT(*) AS count
        FROM reference_palettes
        GROUP BY name
        ORDER BY name
        """
    )

    print(
        "Database created:"
    )

    print(
        DB_PATH
    )

    print(
        "\nPalettes:"
    )

    for name, count in cursor.fetchall():

        print(
            f"  {name:15s}"
            f" : "
            f"{count} colors"
        )

    conn.close()


if __name__ == "__main__":
    init_database()
