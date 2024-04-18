
image_path = r"C:\Users\yangx\OneDrive - KUKA AG\Bilder\c.jpg"

import pygame
import sys

# 初始化Pygame
pygame.init()

# 设置窗口大小和标题
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption(image_path)

# 加载文件图标
file_icon = pygame.image.load(image_path)

# 设置图标的初始位置
file_x = 0
file_y = WINDOW_HEIGHT // 2

# 设置动画帧率
clock = pygame.time.Clock()

# 主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 绘制背景
    window.fill((255, 255, 255))

    # 绘制文件图标
    window.blit(file_icon, (file_x, file_y))

    # 更新文件图标的位置
    file_x += 5  # 移动速度为5个像素/帧

    # 如果文件图标超出窗口范围，重置位置
    if file_x > WINDOW_WIDTH:
        file_x = 0

    # 刷新显示
    pygame.display.flip()

    # 控制帧率
    clock.tick(60)

# 退出Pygame
pygame.quit()
sys.exit()
