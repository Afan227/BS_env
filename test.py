from plot.plot_env_class import TerrainEnvironment
from envs.basestation_env import BaseStationEnv

"""
    获取基站和用户的位置
    这里首先采用固定的位置做测试
    user : (x,y)
    bs   : (x,y,z)
    build: (x,y,width,length,height)
    tree : (x,y,trunk_height, crown_height, radius)
"""
user = [(100,200), (700,800),(600,650),(400,300),(300,400)]
bs = [(500,500,30)]
build = [(100,100,100,20,50)]
tree = [(5,5,15,5,3)]


# 创建地形环境
terrain_env = TerrainEnvironment(x_length=1000, y_length=1000, mesh_num=100, z_mode=1)
terrain_env.add_building(100, 100, 50, 50, 30)
for i in range(len(bs)):
    terrain_env.add_bs(bs[i])
for i in range(len(user)):
    terrain_env.add_user(user[i])

# 创建基站环境，传递地形环境对象
bs_env = BaseStationEnv(
    bs_env=terrain_env
)
"""
    中国移动的测试要求
    极好: sinr > 25
    好点: 16-25
    中点: 11-15
    差 :  3-10
    极差: sinr<3
"""
sinr = bs_env._calculate_sinr()

print(sinr)
bs_env.env.plot_environment(sinr)