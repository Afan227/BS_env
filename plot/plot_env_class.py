import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
from configs.config_env import AREA_SIZE


random.seed(1000)
class TerrainEnvironment:
    def __init__(self, x_length, y_length, mesh_num, z_mode, num_buildings, num_parks):
        self.x_length = x_length
        self.y_length = y_length
        self.mesh_num = mesh_num
        self.z_mode = z_mode
        self.buildings = []
        self.bs = []
        self.users = []
        self._generate_terrain()
        self.num_buildings = num_buildings  # 要生成的建筑物数量

    def _generate_terrain(self):
        """
        生成该三维地图的地面地形
        z_mode:0  平面地形
        z_mode:1  起伏地形
        :return: None
        """
        x = np.linspace(0, self.x_length, self.mesh_num)
        y = np.linspace(0, self.y_length, self.mesh_num)
        self.x, self.y = np.meshgrid(x, y)

        if self.z_mode == 0:  # Flat terrain
            self.z = np.zeros_like(self.x)
        else:  # Undulating terrain
            self.z = np.sin(self.x) * np.cos(self.y) * 0.5
            self.z += np.random.normal(0, 0.1, self.z.shape)

    # 添加建筑物位置
    def add_building(self, x, y, width, length, height):
        self.buildings.append({"x": x, "y": y, "width": width, "length": length, "height": height})
    # 添加基站位置
    def add_bs(self, bs):
        self.bs.append({"x": bs[0], "y": bs[1], "z": bs[2]})
    # 添加用户位置
    def add_user(self, user):
        self.users.append({"x": user[0], "y": user[1],"z":user[2]})
    # 计算坐标点的总体高度，一般等于地形高度+建筑物高度
    def get_total_height(self, x_val, y_val):
        idx_x = np.abs(self.x[0, :] - x_val).argmin()
        idx_y = np.abs(self.y[:, 0] - y_val).argmin()
        total_height = self.z[idx_x, idx_y]

        for building in self.buildings:
            if building["x"] <= x_val <= building["x"] + building["width"] and \
                    building["y"] <= y_val <= building["y"] + building["length"]:
                total_height += building["height"]

        return total_height

    # 检验两点之间是否有障碍物，障碍物一般指连线间是否通过建筑物
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

        return False

    def plot_environment(self,c,index_bs,_pathloss):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(self.x, self.y, self.z, color='grey', alpha=0.4)
        cmap = plt.get_cmap("viridis")  # 使用 'viridis' colormap（你可以选择其他的如 'plasma', 'inferno', 'coolwarm' 等）
        norm = plt.Normalize(vmin=0, vmax=3)  # 设置值的范围，colormap 会根据这个范围来映射颜色
        for building in self.buildings:
            self._plot_building(ax, building)
        for i in range(len(self.bs)):
            self._plot_bs(ax, self.bs[i],i)
        for i in range(len(self.users)):
            self._plot_user(ax, self.users[i],i)
        self._plot_lines(ax,self.users,c,self.bs,index_bs,_pathloss)
        ax.set_xlim([0, AREA_SIZE])
        ax.set_ylim([0, AREA_SIZE])
        ax.set_zlim([0, 125])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Terrain with Buildings and Trees")

        # 添加颜色条 (colorbar)，并与 norm 和 cmap 相关联
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 这个空数组只是为了创建 colorbar
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('Cell Capability')  # 设置颜色条的标签
        plt.show()


    def is_overlap(self, area1, area2):
        """检查两个矩形区域是否重叠"""
        x1, y1, w1, l1 = area1["x"], area1["y"], area1["width"], area1["length"]
        x2, y2, w2, l2 = area2["x"], area2["y"], area2["width"], area2["length"]
        # 判断两个矩形是否有交集
        if (x1 + w1 > x2 and x1 < x2 + w2 and y1 + l1 > y2 and y1 < y2 + l2):
            return True
        return False

    def is_overlap_with_user(self, building, user):
        """
        判断建筑物与用户是否重叠。
        :param building: 建筑物的字典或对象，包含建筑物的位置和尺寸
        :param user: 用户的位置（x, y）
        :return: 如果建筑物与用户重叠，返回 True，否则返回 False
        """
        x, y, width, length, _ = building.values()
        user_x, user_y = user  # 用户的位置

        # 检查用户是否在建筑物的矩形区域内
        if x <= user_x <= x + width and y <= user_y <= y + length:
            return True
        return False

    def generate_city(self,building):
        """生成城市，避免建筑物重叠"""
        for i in range(len(building)):
            new_building = building[i]
            new_building["x"] = new_building["x"] - 0.5 * new_building["width"]
            new_building["y"] = new_building["y"] - 0.5 * new_building["length"]
            print(new_building)
            overlap = False
            for existing_building in self.buildings:
                if self.is_overlap(new_building, existing_building):
                    overlap = True
                    break
            # 检查是否与任何用户重叠
            if not overlap:
                for user in self.users:
                    if self.is_overlap_with_user(new_building, user):
                        overlap = True
                        break
            if not overlap:
                for bs in self.bs:
                    if self.is_overlap_with_user(new_building, bs):
                        overlap = True
                        break
            if not overlap:
                self.buildings.append(new_building)

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
        ax.add_collection3d(Poly3DCollection(faces, color="#8B8B8B", alpha=0.8))

    def _plot_bs(self, ax, bs,i):
        x, y, z = bs.values()
        # 添加一个红点，假设我们在 (5, 5) 位置添加红点，z值是该点的总高度
        z += self.get_total_height(x, y)  # 获取红点的总高度
        bs['z'] = z
        ax.scatter(x, y, z, color='red', s=50)  # 使用scatter函数添加红点，s为点的大小
        ax.text(x, y, z+5, f'{i+1}', color='black', fontsize=7, ha='center',zorder = 10)

    def _plot_user(self, ax, user,i):
        x, y,z = user.values()
        # 添加一个绿点，假设我们在 (5, 5) 位置添加绿点代表用户
        ax.scatter(x, y,z, color='green', s=30,zorder=2)  # 使用scatter函数添加绿点，s为点的大小
        ax.text(x, y, z+5, f'{i+1}', color='black', fontsize=7, ha='center',zorder = 10)

    def _plot_lines(self, ax, users, c, bs, index_bs,_pathloss):
        # 定义一个 colormap（颜色映射）来根据值选择颜色
        cmap = plt.get_cmap("viridis")  # 使用 'viridis' colormap（你可以选择其他的如 'plasma', 'inferno', 'coolwarm' 等）
        norm = plt.Normalize(vmin=0, vmax=3)  # 设置值的范围，colormap 会根据这个范围来映射颜色
        for i in range(len(users)):
            color = cmap(norm(c[i]))  # 将值映射到颜色
            x, y,z = users[i].values()
            x_bs,y_bs,z_bs = bs[index_bs[i]].values()
            mid_x = (x + x_bs) / 2
            mid_y = (y + y_bs) / 2
            mid_z = (z + z_bs) / 2
            ax.text(mid_x, mid_y, mid_z, f'{_pathloss[index_bs[i]]:.2f}', color='black', fontsize=7, ha='center')
            ax.plot([x, x_bs], [y, y_bs], [z, z_bs], marker='o', color=color, label=f'{c[i]}',zorder = 10)
