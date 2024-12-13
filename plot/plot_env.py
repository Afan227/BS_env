
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from configs.config_env import MESH_NUM, AREA_SIZE


class CreateTopography:
    def __init__(self, x_length, y_length,z_mode):
        """创建地形图的size，z_mode表示地面是平整还是其他形式"""
        self.x_length = x_length
        self.y_length = x_length
        self.z_mode = z_mode

    def create_basic_topo(self):
        # 创建基础地形
        if self.z_mode == 0:    # 平整地面
            x = np.linspace(0, self.x_length, MESH_NUM)  # 在 x 轴上生成 100 个点，范围从 0 到 10
            y = np.linspace(0, self.y_length, MESH_NUM)  # 在 y 轴上生成 100 个点，范围从 0 到 10
            x, y = np.meshgrid(x, y)  # 将 x 和 y 变为 2D 网格
            z = np.zeros_like(x)  # 根据 x 和 y 计算地形的 z 值（高度）
        else:
            x = np.linspace(0, self.x_length, MESH_NUM)  # 在 x 轴上生成 100 个点，范围从 0 到 10
            y = np.linspace(0, self.y_length, MESH_NUM)  # 在 y 轴上生成 100 个点，范围从 0 到 10
            x, y = np.meshgrid(x, y)  # 将 x 和 y 变为 2D 网格
            z = np.sin(x) * np.cos(y) * 0.5  # 使用正弦余弦函数，生成较平缓的地形
            z += np.random.normal(0, 0.1, z.shape)  # 加入轻微噪声，生成更自然的起伏


x = np.linspace(0, AREA_SIZE, MESH_NUM)  # 在 x 轴上生成 100 个点，范围从 0 到 10
y = np.linspace(0, AREA_SIZE, MESH_NUM)  # 在 y 轴上生成 100 个点，范围从 0 到 10
x, y = np.meshgrid(x, y)  # 将 x 和 y 变为 2D 网格
z = np.sin(x) * np.cos(y) * 0.5  # 使用正弦余弦函数，生成较平缓的地形
z += np.random.normal(0, 0.1, z.shape)  # 加入轻微噪声，生成更自

# 创建建筑物（长方体）
def add_building(ax, x, y, width, length, height):
    """在地形上添加长方体建筑物"""
    # 定义建筑物的顶点
    vertices = [
        [x, y, 0], [x + width, y, 0], [x + width, y + length, 0], [x, y + length, 0],  # 底部四个点
        [x, y, height], [x + width, y, height], [x + width, y + length, height], [x, y + length, height]  # 顶部四个点
    ]
    # 面的定义
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
        [vertices[3], vertices[0], vertices[4], vertices[7]],  # 左面
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
        [vertices[0], vertices[1], vertices[2], vertices[3]]   # 底面
    ]
    # 添加到图形
    ax.add_collection3d(Poly3DCollection(faces, color="gray", alpha=0.8))

# 创建树木（圆柱体和圆锥体）
def add_tree(ax, x, y, trunk_height, crown_height, radius):
    """在地形上添加树木"""
    # 树干
    theta = np.linspace(0, 2 * np.pi, 30)  # 圆周角
    z_trunk = np.linspace(0, trunk_height, 30)  # 树干高度
    theta, z_trunk = np.meshgrid(theta, z_trunk)  # 创建网格
    x_trunk = radius * np.cos(theta) + x  # 树干的 x 坐标
    y_trunk = radius * np.sin(theta) + y  # 树干的 y 坐标
    ax.plot_surface(x_trunk, y_trunk, z_trunk, color="brown")  # 绘制树干

    # 树冠
    z_crown = np.linspace(trunk_height, trunk_height + crown_height, 30)  # 树冠的高度
    r_crown = np.linspace(radius, 0, 30)  # 树冠的半径
    theta = np.linspace(0, 2 * np.pi, 30)  # 圆周角

    # 树冠的网格
    theta, r_crown = np.meshgrid(theta, r_crown)  # 创建网格
    x_crown = r_crown * np.cos(theta) + x  # 树冠的 x 坐标
    y_crown = r_crown * np.sin(theta) + y  # 树冠的 y 坐标

    # 将树冠的高度调整为与半径相匹配的二维数组
    z_crown = np.outer(np.linspace(trunk_height, trunk_height + crown_height, 30), np.ones(30))
    print(x_crown)
    print(y_crown)
    print(z_crown)
    ax.plot_surface(x_crown, y_crown, z_crown, color="green")  # 绘制树冠


# 直接计算某个点的总高度（地形 + 建筑物 + 树木）
def get_total_height(x_val, y_val, x, y, z, buildings, trees):
    """通过 x 和 y 值找到对应的总高度（地形 + 建筑物 + 树木高度）"""
    # 计算 (x_val, y_val) 在网格中的最近位置
    idx_x = np.abs(x[0, :] - x_val).argmin()  # 找到 x 上最接近的索引
    idx_y = np.abs(y[:, 0] - y_val).argmin()  # 找到 y 上最接近的索引

    # 先获取地形高度
    total_height = z[idx_y, idx_x]

    # 检查建筑物是否存在于该位置并加上建筑物高度
    for building in buildings:
        bx, by, bw, bl, bh = building["x"], building["y"], building["width"], building["length"], building["height"]
        if bx <= x_val <= bx + bw and by <= y_val <= by + bl:
            total_height += bh  # 建筑物高度加到总高度

    # 检查树木是否存在于该位置并加上树木高度
    for tree in trees:
        tx, ty, trunk_height, crown_height, radius = tree["x"], tree["y"], tree["trunk_height"], tree["crown_height"], \
        tree["radius"]
        if np.sqrt((x_val - tx) ** 2 + (y_val - ty) ** 2) <= radius:
            total_height += trunk_height + crown_height  # 树木高度加到总高度

    return total_height
buildings = [
    {"x": 100, "y": 100, "width": 100, "length": 20, "height": 50},
    {"x": 700, "y": 700, "width": 30, "length": 100, "height": 20}
]
# 树木信息（位置、树干高度、树冠高度）
trees = [
    {"x": 5, "y": 5, "trunk_height": 15, "crown_height": 5, "radius": 3},
    {"x": 500, "y": 500, "trunk_height": 20, "crown_height": 8, "radius": 10}
]


# 判断两点之间是否有建筑物或树木遮挡
def check_obstacle(start, end, buildings, trees):
    """
    判断两点之间是否有建筑物或树木遮挡。
    :param start: 起始点 (x1, y1, z1)
    :param end: 目标点 (x2, y2, z2)
    :param buildings: 建筑物列表
    :param trees: 树木列表
    :return: True if there is an obstacle, False if not
    """
    # 计算两点之间的直线（射线）方向向量
    direction = np.array(end) - np.array(start)
    distance = np.linalg.norm(direction)
    direction /= distance  # 归一化方向向量

    # 逐步沿射线方向检查建筑物和树木
    steps = 100  # 可以调整步长来提高精度
    for step in range(1, steps + 1):
        current_point = np.array(start) + direction * (step / steps) * distance  # 当前检查的点
        x, y, z = current_point

        # 检查建筑物
        for building in buildings:
            bx, by, bw, bl, bh = building["x"], building["y"], building["width"], building["length"], building["height"]
            if bx <= x <= bx + bw and by <= y <= by + bl and 0 <= z <= bh:
                return True  # 发现建筑物遮挡

        # 检查树木
        for tree in trees:
            tx, ty, trunk_height, crown_height, radius = tree["x"], tree["y"], tree["trunk_height"], tree[
                "crown_height"], tree["radius"]
            if np.sqrt((x - tx) ** 2 + (y - ty) ** 2) <= radius and 0 <= z <= trunk_height + crown_height:
                return True  # 发现树木遮挡

    return False  # 没有找到任何遮挡



# 绘图
fig = plt.figure(figsize=(10, 7))  # 创建图形对象
ax = fig.add_subplot(111, projection='3d')  # 创建 3D 坐标轴

# 绘制地形
ax.plot_surface(x, y, z, color='grey', alpha=0.4)  # 绘制地形，使用 terrain 色图，alpha 设置透明度


# 添加建筑物
for building in buildings:
    add_building(ax, building["x"], building["y"], building["width"], building["length"], building["height"])
# 添加树木
for tree in trees:
    add_tree(ax, tree["x"], tree["y"], trunk_height=tree["trunk_height"], crown_height=tree["crown_height"], radius=tree["radius"])

# 添加一个红点，假设我们在 (5, 5) 位置添加红点，z值是该点的总高度
x_red, y_red = 100, 100  # 红点的坐标
z_red = get_total_height(x_red, y_red, x, y, z, buildings, trees)  # 获取红点的总高度
ax.scatter(x_red, y_red, z_red, color='red', s=50)  # 使用scatter函数添加红点，s为点的大小


# 调整视角
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Terrain with Buildings and Trees")
ax.set_xlim([0, 1000])
ax.set_ylim([0, 1000])
ax.set_zlim([0, 200])
plt.show()  # 显示图形

# 获取某个坐标点的总高度
x_val = 5  # 指定查询的 x 坐标
y_val = 5  # 指定查询的 y 坐标
total_height = get_total_height(x_val, y_val, x, y, z, buildings, trees)
print(f"The total height at point ({x_val}, {y_val}) is: {total_height}")
