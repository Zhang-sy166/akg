import matplotlib.pyplot as plt

# 阶段 1: 初始上升
x1 = [1, 2, 3, 4]
y1 = [1, 1.5, 2.5, 2.7]

# 阶段 2: 回退后的上升 (横坐标 4 再次出现)
x2 = [4, 5, 6, 7]
y2 = [0.5, 1, 2, 3]

plt.figure(figsize=(8, 5))

# 绘制第一段
plt.plot(x1, y1, marker='o', color='C0', label='Phase 1')
# 绘制第二段
plt.plot(x2, y2, marker='o', color='C0')

# 绘制连接虚线 (表示回退动作)
plt.plot([4, 4], [y1[-1], y2[0]], linestyle='--', color='gray')

# 添加标注 (对应图中 id0, id3 等)
for i, txt in enumerate(['l0-id0', 'l1-id0', 'l2-id0', 'l3-id0']):
    plt.annotate(txt, (x1[i], y1[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('Evolution Time')
plt.ylabel('Speed Up')
plt.title('Rollback Visualization')
plt.grid(True, linestyle=':', alpha=0.6)
# plt.show()
plt.savefig("rollback_visualization.png")