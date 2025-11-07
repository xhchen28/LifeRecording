# # # # import matplotlib.pyplot as plt
# # # # import numpy as np
# # # # import matplotlib.patheffects as path_effects

# # # # plt.xkcd()  # 卡通风格
# # # # x = np.linspace(0, 10, 100)
# # # # y = np.cos(x)


# # # # fig, ax = plt.subplots(figsize=(8, 5))
# # # # (line,) = ax.plot(x, y, linewidth=2, color='orange')
# # # # line.set_path_effects([
# # # #     path_effects.SimpleLineShadow(offset=(1.5, -1.5), shadow_color='gray'),
# # # #     path_effects.Normal()
# # # # ])
# # # # ax.set_title("Cute Handwritten Line Chart")
# # # # ax.set_facecolor("#fffbe6")  # 类似纸张背景
# # # # plt.show()




# # # import matplotlib.pyplot as plt
# # # import matplotlib.patheffects as path_effects
# # # import math

# # # plt.xkcd()  # 启用卡通风格

# # # # 用纯 Python 列表生成数据
# # # # x = [i * 0.1 for i in range(100)]        # 0 到 10，每步 0.1
# # # # y = [math.cos(v) for v in x]             # y = cos(x)
# # # x = [32, 52, 64, 96, 128, 160, 192, 224, 256, 288]
# # # y = [5952.273489460, 9647.711535062, 10975.257735666, 13524.76261611, 15990.763350017, 16814.000491307, 18488.48805467, 19273.356378398, 20624.659376392, 21588.449908947]

# # # fig, ax = plt.subplots(figsize=(8, 5))
# # # (line,) = ax.plot(x, y, linewidth=2, color='orange')

# # # # 添加阴影效果
# # # line.set_path_effects([
# # #     path_effects.SimpleLineShadow(offset=(1.5, -1.5), shadow_color='gray'),
# # #     path_effects.Normal()
# # # ])

# # # ax.set_title("Throughput vs. Batch Size")
# # # ax.set_facecolor("#fffbe6")  # 类似纸张背景
# # # plt.show()


# # import matplotlib.pyplot as plt
# # import matplotlib.patheffects as path_effects

# # plt.xkcd()  # 启用卡通风格

# # # 数据
# # x = [32, 52, 64, 96, 128, 160, 192, 224, 256, 288]
# # y = [5952.273489460, 9647.711535062, 10975.257735666, 13524.76261611,
# #      15990.763350017, 16814.000491307, 18488.48805467,
# #      19273.356378398, 20624.659376392, 21588.449908947]

# # fig, ax = plt.subplots(figsize=(8, 5))
# # (line,) = ax.plot(x, y, linewidth=2, color='orange', label='Measured Throughput')

# # # 添加一条虚线（例如理想线）
# # (line2,) = ax.plot(
# #     52,
# #     linestyle='--', color='gray', linewidth=2, label='Ideal Trend'
# # )

# # # 阴影效果（只对主折线）
# # line.set_path_effects([
# #     path_effects.SimpleLineShadow(offset=(1.5, -1.5), shadow_color='gray'),
# #     path_effects.Normal()
# # ])

# # ax.set_title("Throughput vs. Batch Size")
# # ax.set_facecolor("#fffbe6")  # 类似纸张背景
# # ax.set_xlabel("Batch Size")
# # ax.set_ylabel("Throughput (MB/s)")
# # ax.legend()
# # plt.show()


# import matplotlib.pyplot as plt
# import matplotlib.patheffects as path_effects

# plt.xkcd()  # 启用卡通风格

# # 数据
# x = [32, 52, 64, 96, 128, 160, 192, 224, 256, 288]
# y = [5952.273489460, 9647.711535062, 10975.257735666, 13524.76261611,
#      15990.763350017, 16814.000491307, 18488.48805467,
#      19273.356378398, 20624.659376392, 21588.449908947]

# fig, ax = plt.subplots(figsize=(8, 5))
# (line,) = ax.plot(x, y, linewidth=2, color='orange', label='Measured Throughput')

# # 添加阴影效果
# line.set_path_effects([
#     path_effects.SimpleLineShadow(offset=(1.5, -1.5), shadow_color='gray'),
#     path_effects.Normal()
# ])

# # 👉 在 x=128 处画一条虚直线
# ax.axvline(x=128, color='gray', linestyle='--', linewidth=2)
# # （你可以改成其他位置，比如 ax.axvline(x=192, ...)）

# # 也可以加注释
# ax.text(128, max(y)*0.9, 'x = 128', rotation=90, va='center', ha='right')

# ax.set_title("Throughput vs. Batch Size")
# ax.set_facecolor("#fffbe6")  # 类似纸张背景
# ax.set_xlabel("Batch Size")
# ax.set_ylabel("Throughput (MB/s)")
# ax.legend()
# plt.show()



import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

plt.xkcd()  # 启用卡通风格

# 数据
x = [32, 52, 64, 96, 128, 160, 192, 224, 256, 288]
y = [i * 2 for i in [32, 52, 64, 96, 128, 160, 192, 224, 256, 288]]

fig, ax = plt.subplots(figsize=(9, 6))
(line,) = ax.plot(x, y, linewidth=2, color='orange', label='Measured Throughput')

# 添加阴影效果
line.set_path_effects([
    path_effects.SimpleLineShadow(offset=(1.5, -1.5), shadow_color='gray'),
    path_effects.Normal()
])

# 👉 在 y=18000 处画一条横向虚线
ax.axhline(y=9647.711535062, color='gray', linestyle='--', linewidth=2)
# 也可以换成 ax.axhline(y=np.mean(y), ...) 表示平均值

# 添加文字标注
ax.text(130, 9647, 'Batch Size = 52', ha='right', va='bottom')

ax.set_title("Throughput vs. Batch Size")
ax.set_facecolor("#fffbe6")  # 类似纸张背景
ax.set_xlabel("Batch Size")
ax.set_ylabel("Throughput (tokens/s)")
ax.legend()
plt.show()
