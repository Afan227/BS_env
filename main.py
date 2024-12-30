from configs.config_env import *
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

# 创建地形环境
terrain_env = TerrainEnvironment(x_length=AREA_SIZE, y_length=AREA_SIZE, mesh_num=MESH_NUM, z_mode=1, num_buildings=NUM_BUILDING,num_parks=2)
terrain_env.generate_city(building)

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
c, index_bs, _pathloss = bs_env._calculate_capability()

bs_env.env.plot_environment(c,index_bs,_pathloss)