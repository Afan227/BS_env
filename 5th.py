import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class CityBuildingGenerator:
    def __init__(self, area_size=10000, num_buildings=100):
        self.area_size = area_size  # 设置区域大小 10000m * 10000m
        self.num_buildings = num_buildings  # 要生成的建筑物数量
        self.buildings = []  # 存储建筑物数据

    def generate_building(self):
        """生成单个建筑物的随机位置和尺寸"""
        # 随机位置：建筑物中心坐标
        x = random.uniform(0, self.area_size)
        y = random.uniform(0, self.area_size)

        # 随机尺寸：建筑物宽度和长度，范围可以根据城市建筑物的实际情况来设定
        width = random.uniform(50, 200)  # 宽度 10m 到 200m
        length = random.uniform(50, 200)  # 长度 10m 到 200m

        # 随机高度：高度根据实际城市情况设定
        height = random.uniform(10, 30)  # 高度 10m 到 100m

        return {"x": x - width / 2, "y": y - length / 2, "width": width, "length": length, "height": height}

    def check_overlap(self, new_building):
        """检查新生成的建筑物是否与现有建筑物重叠"""
        for building in self.buildings:
            # 获取现有建筑物的位置和尺寸
            x1, y1, w1, l1 = building["x"], building["y"], building["width"], building["length"]
            x2, y2, w2, l2 = new_building["x"], new_building["y"], new_building["width"], new_building["length"]

            # 检查是否重叠：如果两建筑物的水平边界有交集，则视为重叠
            if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + l2 and y1 + l1 > y2):
                return True
        return False

    def generate_city(self):
        """生成整个城市的建筑物，并避免重叠"""
        for _ in range(self.num_buildings):
            while True:
                # 随机生成一个建筑物
                new_building = self.generate_building()

                # 检查是否与现有建筑物重叠
                if not self.check_overlap(new_building):
                    self.buildings.append(new_building)
                    break  # 生成成功，跳出循环

    def _plot_building(self, ax, building):
        """绘制单个建筑物"""
        x, y, width, length, height = building.values()
        vertices = [
            [x, y, 0], [x + width, y, 0], [x + width, y + length, 0], [x, y + length, 0],
            [x, y, height], [x + width, y, height], [x + width, y + length, height], [x, y + length, height]
        ]
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
            [vertices[3], vertices[0], vertices[4], vertices[7]],  # 左面
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
            [vertices[0], vertices[1], vertices[2], vertices[3]]  # 底面
        ]
        ax.add_collection3d(Poly3DCollection(faces, color="gray", alpha=0.8))

    def plot_city(self):
        """绘制整个城市的建筑物和地面"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        x = np.linspace(0, self.area_size, 1000)
        y = np.linspace(0, self.area_size, 1000)
        X, Y = np.meshgrid(x, y)


        Z = np.sin(X) * np.cos(Y) * 0.5
        Z += np.random.normal(0, 0.1, Z.shape)

        # 绘制地面表面
        ax.plot_surface(X, Y, Z, color='grey', alpha=0.4)

        # 遍历所有建筑物并绘制
        for building in self.buildings:
            self._plot_building(ax, building)

        # 设置坐标轴范围
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set_zlim(0, 150)  # 设置z轴高度范围

        # 添加标题
        ax.set_title("3D City Buildings with Terrain")
        plt.savefig("city_building_with_terrain.png", dpi=300, bbox_inches='tight')
        # 显示图形
        # plt.show()


# 使用示例
city_generator = CityBuildingGenerator(area_size=10000, num_buildings=50)
city_generator.generate_city()
city_generator.plot_city()
