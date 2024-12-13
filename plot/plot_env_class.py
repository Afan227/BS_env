import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class TerrainEnvironment:
    def __init__(self, x_length, y_length, mesh_num, z_mode):
        self.x_length = x_length
        self.y_length = y_length
        self.mesh_num = mesh_num
        self.z_mode = z_mode
        self.buildings = []
        self.trees = []
        self.bs = []
        self.users = []
        self._generate_terrain()

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
        self.users.append({"x": user[0], "y": user[1]})

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

        for tree in self.trees:
            self._plot_tree(ax, tree)
        for bs in self.bs:
            self._plot_bs(ax, bs)
        for user in self.users:
            self._plot_user(ax, user)
        self._plot_values(ax,self.users,sinrs)
        ax.set_xlim([0, 1000])
        ax.set_ylim([0, 1000])
        ax.set_zlim([0, 200])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Terrain with Buildings and Trees")
        plt.show()

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
        x, y = user.values()
        # 添加一个绿点，假设我们在 (5, 5) 位置添加绿点代表用户
        ax.scatter(x, y, color='green', s=30)  # 使用scatter函数添加绿点，s为点的大小

    def _plot_values(self, ax, users, sinrs):
        for i in range(len(users)):
            x, y = users[i].values()
            print(str(sinrs[i]))
            ax.text(x+10,y+10,0, s = '{:.2f}'.format(sinrs[i]), fontsize=6, color='black')


    def _plot_line(self, users, bs):
        pass