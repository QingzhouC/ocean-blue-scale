"""
初始化训练色卡数据库 (SQLite)

运行此脚本会创建 / 重建 palettes.db，
将所有参考色卡写入 reference_palettes 表。

用法:
    python3 init_palette_db.py
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
        "#072567",  # C9
        "#0C399C",  # C8
        "#0E4FD4",  # C7
        "#105CF4",  # C6
        "#1274FE",  # C5
        "#6298FF",  # C4
        "#A0BFFF",  # C3
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
        "#C908A9",
        "#E60FC2",
        "#EB47CE",
        "#F47DDD",
        "#FCAEED",
        "#FFD5F7",
        "#FFE8FB",
    ],

    "green": [
        "#004D12",
        "#035F17",
        "#01801F",
        "#00A426",
        "#00B82B",
        "#34C44D",
        "#6CD67D",
        "#A3EAAF",
        "#D1F9D9",
        "#E8FFED",
    ],

    "yellow": [
        "#4D4000",
        "#685702",
        "#9C8202",
        "#D4B000",
        "#F4CB00",
        "#F4D339",
        "#F8DF6F",
        "#FCECA4",
        "#FFF6D1",
        "#FFFBE8",
    ],

    "cyan": [
        "#00404D",
        "#035667",
        "#068199",
        "#07AECE",
        "#0FC8ED",
        "#4AD0EE",
        "#82DDF1",
        "#B2EBF8",
        "#D6F6FD",
        "#E8FBFF",
    ],

    "orange": [
        "#4D2700",
        "#683606",
        "#9C520A",
        "#D4700D",
        "#F48210",
        "#F69430",
        "#FDB063",
        "#FFCE9C",
        "#FFE8CE",
        "#FFF4E8",
    ],

    "light_blue": [
        "#00274D",
        "#083667",
        "#0B529C",
        "#0E70D4",
        "#1082F4",
        "#1E94FA",
        "#66B1FE",
        "#A2CFFF",
        "#D1E8FF",
        "#E8F4FF",
    ],
    
    "purple_blue": [
        "#00044D",
        "#141167",
        "#2E2A9B",
        "#4645D3",
        "#4E56F4",
        "#5F6CFC",
        "#898FFF",
        "#B4B7FF",
        "#D7D9FF",
        "#E8E9FF",
    ],

    "purple": [
        "#33004D",
        "#460268",
        "#6A069C",
        "#910BD4",
        "#A810F4",
        "#B840F8",
        "#CF75FD",
        "#E4AAFF",
        "#F2D4FF",
        "#F7E8FF",
    ],

    "yellow": [
        "#4d3300",
        "#744e01",
        "#996600",
        "#c08002",
        "#e69900",
        "#ffbb33",
        "#ffcc66",
        "#ffdd99",
        "#ffe9bd",
        "#fff2d8",
    ],

    "green": [
        "#003022",
        "#003526",
        "#004531",
        "#005e43",
        "#007855",
        "#009267",
        "#0dbd89",
        "#2fddaa",
        "#99f3d9",
        "#caffef",
    ],

    "green": [
        "#00310b",
        "#00350d",
        "#004811",
        "#006217",
        "#008520",
        "#00b82b",
        "#00dd33",
        "#30fe5f",
        "#a9ffbe",
        "#d8ffe2",
    ],

    "yellow": [
        "#574800",
        "#705d00",
        "#947b00",
        "#c2a100",
        "#e0bb00",
        "#f4cb00",
        "#f8d426",
        "#f9de58",
        "#fbe784",
        "#fdefb0",
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
