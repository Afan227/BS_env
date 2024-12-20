import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
from configs.config_env import AREA_SIZE



class TerrainEnvironment:
    def __init__(self, x_length, y_length, mesh_num, z_mode, num_buildings, num_parks):
        self.x_length = x_length
        self.y_length = y_length
        self.mesh_num = mesh_num
        self.z_mode = z_mode
        self.num_parks = num_parks  # 公园数量
        self.parks = []
        self.buildings = []
        self.trees = []
        self.bs = []
        self.users = []
        self._generate_terrain()
        self.num_buildings = num_buildings  # 要生成的建筑物数量

    def _generate_terrain(self):
        x = np.linspace(0, self.x_length, self.mesh_num)
        y = np.linspace(0, self.y_length, self.mesh_num)
        self.x, self.y = np.meshgrid(x, y)

        if self.z_mode == 0:  # Flat terrain
            self.z = np.zeros_like(self.x)
        else:  # Undulating terrain
            self.z = np.sin(self.x) * np.cos(self.y) * 0.5
            self.z += np.random.normal(0, 0.1, self.z.shape)

    def add_building(self, x, y, width, length, height):
        self.buildings.append({"x": x, "y": y, "width": width, "length": length, "height": height})

    def add_tree(self, x, y, trunk_height, crown_height, radius):
        self.trees.append(
            {"x": x, "y": y, "trunk_height": trunk_height, "crown_height": crown_height, "radius": radius})

    def add_bs(self, bs):
        self.bs.append({"x": bs[0], "y": bs[1], "z": bs[2]})

    def add_user(self, user):
        self.users.append({"x": user[0], "y": user[1],"z":user[2]})

    def get_total_height(self, x_val, y_val):
        idx_x = np.abs(self.x[0, :] - x_val).argmin()
        idx_y = np.abs(self.y[:, 0] - y_val).argmin()
        total_height = self.z[idx_x, idx_y]

        for building in self.buildings:
            if building["x"] <= x_val <= building["x"] + building["width"] and \
                    building["y"] <= y_val <= building["y"] + building["length"]:
                total_height += building["height"]

        for tree in self.trees:
            if np.sqrt((x_val - tree["x"]) ** 2 + (y_val - tree["y"]) ** 2) <= tree["radius"]:
                total_height += tree["trunk_height"] + tree["crown_height"]

        return total_height

    def check_obstacle(self, start, end):
        direction = np.array(end) - np.array(start)
        distance = np.linalg.norm(direction)
        direction = direction.astype(np.float32)
        direction /= distance
        steps = 100
        for step in range(1, steps + 1):
            current_point = np.array(start) + direction * (step / steps) * distance
            x, y, z = current_point

            for building in self.buildings:
                if building["x"] <= x <= building["x"] + building["width"] and \
                        building["y"] <= y <= building["y"] + building["length"] and \
                        0 <= z <= building["height"]:
                    return True


            for tree in self.trees:
                if np.sqrt((x - tree["x"]) ** 2 + (y - tree["y"]) ** 2) <= tree["radius"] and \
                        0 <= z <= tree["trunk_height"] + tree["crown_height"]:
                    return True
        return False

    def plot_environment(self,sinrs):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(self.x, self.y, self.z, color='grey', alpha=0.4)

        for building in self.buildings:
            self._plot_building(ax, building)

        # 绘制公园地面
        for park in self.parks:
            self._plot_park(ax, park)

        for tree in self.trees:
            self._plot_tree(ax, tree)
        for bs in self.bs:
            self._plot_bs(ax, bs)
        for user in self.users:
            self._plot_user(ax, user)
        self._plot_values(ax,self.users,sinrs)
        ax.set_xlim([0, AREA_SIZE])
        ax.set_ylim([0, AREA_SIZE])
        ax.set_zlim([0, 200])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Terrain with Buildings and Trees")
        plt.show()

    def generate_building(self):
        """生成单个建筑物的随机位置和尺寸"""
        # 随机位置：建筑物中心坐标
        x = random.uniform(0, self.x_length)
        y = random.uniform(0, self.y_length)

        # 随机尺寸：建筑物宽度和长度，范围可以根据城市建筑物的实际情况来设定
        width = random.uniform(50, 200)  # 宽度 10m 到 200m
        length = random.uniform(50, 200)  # 长度 10m 到 200m

        # 随机高度：高度根据实际城市情况设定
        height = random.uniform(10, 30)  # 高度 10m 到 100m

        return {"x": x - width / 2, "y": y - length / 2, "width": width, "length": length, "height": height}

    def is_overlap(self, area1, area2):
        """检查两个矩形区域是否重叠"""
        x1, y1, w1, l1 = area1["x"], area1["y"], area1["width"], area1["length"]
        x2, y2, w2, l2 = area2["x"], area2["y"], area2["width"], area2["length"]
        # 判断两个矩形是否有交集
        if (x1 + w1 > x2 and x1 < x2 + w2 and y1 + l1 > y2 and y1 < y2 + l2):
            return True
        return False

    def generate_city(self):
        """生成城市，避免建筑物重叠"""
        while len(self.buildings) < self.num_buildings:
            new_building = self.generate_building()
            overlap = False
            for existing_building in self.buildings:
                if self.is_overlap(new_building, existing_building):
                    overlap = True
                    break
            if not overlap:
                self.buildings.append(new_building)

    def generate_parks(self):
        """生成公园，避免公园与建筑物重叠"""
        while len(self.parks) < self.num_parks:
            park_x = random.randint(0, self.x_length - 500)
            park_y = random.randint(0, self.y_length - 500)
            park_width = random.randint(1300, 1800)
            park_length = random.randint(1300, 1800)
            new_park = {"x": park_x, "y": park_y, "width": park_width, "length": park_length}

            # 检查新公园与现有建筑物是否重叠
            overlap = False
            for building in self.buildings:
                if self.is_overlap(new_park, building):
                    overlap = True
                    break

            # 如果不重叠，添加公园到城市
            if not overlap:
                self.parks.append(new_park)

    def _plot_building(self, ax, building):
        x, y, width, length, height = building.values()
        vertices = [
            [x, y, 0], [x + width, y, 0], [x + width, y + length, 0], [x, y + length, 0],
            [x, y, height], [x + width, y, height], [x + width, y + length, height], [x, y + length, height]
        ]
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[3], vertices[0], vertices[4], vertices[7]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[2], vertices[3]]
        ]
        ax.add_collection3d(Poly3DCollection(faces, color="gray", alpha=0.8))

    def _plot_park(self, ax, park):
        """绘制公园地面"""
        x, y, width, length = park.values()
        # 绘制公园地面为绿色，在z=0处绘制平面
        park_vertices = [
            [x, y, 0], [x + width, y, 0], [x + width, y + length, 0], [x, y + length, 0]
        ]
        park_faces = [[park_vertices[0], park_vertices[1], park_vertices[2], park_vertices[3]]]
        ax.add_collection3d(Poly3DCollection(park_faces, color="green", alpha=0.5))

    def _plot_tree(self, ax, tree):
        x, y, trunk_height, crown_height, radius = tree.values()

        theta = np.linspace(0, 2 * np.pi, 30)
        z_trunk = np.linspace(0, trunk_height, 30)
        theta, z_trunk = np.meshgrid(theta, z_trunk)
        x_trunk = radius * np.cos(theta) + x
        y_trunk = radius * np.sin(theta) + y
        ax.plot_surface(x_trunk, y_trunk, z_trunk, color="brown")

        z_crown = np.linspace(trunk_height, trunk_height + crown_height, 30)
        r_crown = np.linspace(radius, 0, 30)
        theta = np.linspace(0, 2 * np.pi, 30)  # 圆周角
        theta, r_crown = np.meshgrid(theta, r_crown)
        x_crown = r_crown * np.cos(theta) + x
        y_crown = r_crown * np.sin(theta) + y
        z_crown = np.outer(np.linspace(trunk_height, trunk_height + crown_height, 30), np.ones(30))
        print(x_crown)
        print(y_crown)
        print(z_crown)
        ax.plot_surface(x_crown, y_crown, z_crown, color="green")

    def _plot_bs(self, ax, bs):
        x, y, z = bs.values()
        # 添加一个红点，假设我们在 (5, 5) 位置添加红点，z值是该点的总高度
        z += self.get_total_height(x, y)  # 获取红点的总高度
        ax.scatter(x, y, z, color='red', s=50)  # 使用scatter函数添加红点，s为点的大小

    def _plot_user(self, ax, user):
        x, y,z = user.values()
        # 添加一个绿点，假设我们在 (5, 5) 位置添加绿点代表用户
        ax.scatter(x, y,z, color='green', s=30)  # 使用scatter函数添加绿点，s为点的大小

    def _plot_values(self, ax, users, sinrs):
        for i in range(len(users)):
            x, y,z = users[i].values()
            print(str(sinrs[i]))
            ax.text(x+10,y+10,z+30, s = '{:.2f}'.format(sinrs[i]), fontsize=6, color='black')


    def _plot_line(self, users, bs):
        pass