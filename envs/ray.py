import numpy as np


def ray_terrain_intersection(ray_origin, ray_direction, terrain_points):
    """
    检查射线是否与地形相交

    :param ray_origin: 射线起点 (x, y, z)
    :param ray_direction: 射线方向 (dx, dy, dz), 单位向量
    :param terrain_points: 地形点云, 每个点 (x, y, z)
    :return: 是否相交 (True/False), 以及最近交点
    """
    ray_origin = np.array(ray_origin)
    ray_direction = np.array(ray_direction)
    terrain_points = np.array(terrain_points)

    # 计算射线起点到地形点的矢量
    vectors_to_points = terrain_points - ray_origin

    # 射线与每个地形点的投影距离
    projections = np.dot(vectors_to_points, ray_direction)

    # 筛选射线方向上的点
    valid_points = terrain_points[projections > 0]

    # 计算到射线的垂直距离
    distances_to_ray = np.linalg.norm(
        valid_points - (ray_origin + projections[:, None] * ray_direction), axis=1
    )

    # 判断是否有交点（假设交点条件为距离小于一定阈值）
    threshold = 1.0  # 距离阈值
    intersects = distances_to_ray < threshold

    if np.any(intersects):
        # 找到最近的交点
        nearest_point_idx = np.argmin(projections[intersects])
        nearest_intersection = valid_points[nearest_point_idx]
        return True, nearest_intersection
    return False, None