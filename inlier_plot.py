import matplotlib.pyplot as plt

# 1. 准备最新数据 (对应 frames 20 到 28)
frames = list(range(20, 29)) 

# 根据最新截图提取的数据
raw_inliers =      [974, 985, 1023, 895, 0, 0, 4, 212, 175]
vanilla_inliers =  [26, 36, 86, 90, 24, 109, 211, 305, 320]
adaptive_inliers = [1001, 985, 960, 895, 24, 261, 320, 450, 423]

# 2. 创建图表
plt.figure(figsize=(10, 6), dpi=300)

# 3. 绘制"追踪丢失"的危险区域 (假设阈值依然是 300)
plt.fill_between(frames, -50, 300, color='red', alpha=0.08, label='Tracking Lost Zone (<300)')
plt.axhline(y=300, color='red', linestyle='--', linewidth=1.5)

# 4. 绘制折线
plt.plot(frames, raw_inliers, label='Raw', 
         color='gray', marker='s', linestyle='-.', linewidth=2, markersize=7)

plt.plot(frames, vanilla_inliers, label='Vanilla Enhancer', 
         color='red', marker='X', linestyle='-', linewidth=2.5, markersize=9)

plt.plot(frames, adaptive_inliers, label='Ours (Adaptive Enhancer)', 
         color='blue', marker='o', linestyle='-', linewidth=2.5, markersize=8)

# 5. 图表细节设置
plt.title('Number of Inliers during Sudden Light Turn-off', fontsize=16, fontweight='bold')
plt.xlabel('Frame Index', fontsize=14)
plt.ylabel('Number of Inliers', fontsize=14)

# Y 轴范围设置
plt.ylim(-20, 1100)
plt.xticks(frames)

# 添加网格线
plt.grid(True, linestyle=':', alpha=0.7)

# 设置图例位置 (放在中间偏右，避免遮挡数据)
plt.legend(fontsize=12, loc='upper right', framealpha=0.9)

# 添加文本标注，指出关灯瞬间 (第23到24帧之间发生断崖下跌)
# plt.annotate('Lights Off', xy=(23.8, 500), xytext=(21.5, 400),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
#              fontsize=12, fontweight='bold')

plt.tight_layout()

# 6. 保存和显示
plt.savefig('lights_off_drop_updated.png', format='png', bbox_inches='tight')
plt.show()