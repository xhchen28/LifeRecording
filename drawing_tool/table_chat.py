# import matplotlib.pyplot as plt
# from matplotlib import font_manager

# plt.xkcd()  # 手绘风格

# fig, ax = plt.subplots(figsize=(6, 3))
# ax.axis('off')

# columns = ["Function", "Parameter"]
# rows = [
#     ["double-stream", "open"],
#     ["mtp", "2"],
#     ["mtp-accept_ratio", "1.7"],
#     ["node", "4"],
#     ["DP-size", "8"],
#     ["TP-size", "1"],
#     ["EP-size", "32"],
#     ["attention-engine", "flashmla"],
#     ["ctx_len", "32768"],
#     ["model", "deepseek_v32"]
# ]

# # 表格
# table = ax.table(
#     cellText=rows,
#     colLabels=columns,
#     loc='center',
#     cellLoc='center'
# )

# table.scale(1.2, 1.4)
# table.auto_set_font_size(False)
# table.set_fontsize(12)

# # ✅ 指定系统支持对号的字体（例如 DejaVu Sans）
# prop = font_manager.FontProperties(family="DejaVu Sans")
# for key, cell in table.get_celld().items():
#     cell.set_text_props(fontproperties=prop)

# # ax.set_title("Handwritten-style Performance Table", fontsize=14, pad=15)
# plt.subplots_adjust(top=0.85, bottom=0.1)
# plt.show()

import matplotlib.pyplot as plt
from matplotlib import font_manager
import random

plt.xkcd()  # 启用手绘风格

fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')

columns = ["Function", "Parameter"]
rows = [
]

# 表格绘制
table = ax.table(
    cellText=rows,
    colLabels=columns,
    loc='center',
    cellLoc='center'
)
table.scale(1.3, 1.8)
table.auto_set_font_size(False)
table.set_fontsize(12)

# 指定支持 Unicode 的字体
prop = font_manager.FontProperties(family="DejaVu Sans")
for key, cell in table.get_celld().items():
    cell.set_text_props(fontproperties=prop)

# 🎨 设置第一行数据（即 rows[0]）为特殊颜色
highlight_color = "#fff2a8"  # 柔和黄色
text_color = "#000000"

for j in range(len(columns)):  # 第一行的两个单元格
    table[0, j].set_facecolor(highlight_color)
    table[0, j].get_text().set_color(text_color)
    table[0, j].get_text().set_weight("bold")

# ---------- 🎯 特别优化标题 ----------
title_text = ""
x_offset = random.uniform(-0.02, 0.02)
y_offset = random.uniform(-0.02, 0.02)

ax.text(
    0.5 + x_offset, 
    1.05 + y_offset, 
    title_text,
    ha='center', va='bottom',
    fontsize=16,
    weight='bold'
)

plt.subplots_adjust(top=0.82, bottom=0.12)
plt.tight_layout()
plt.show()