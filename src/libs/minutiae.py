from itertools import combinations

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from libs.basics import display_image, extract_angle, euclidian_distance


def extract_minutiae(image: np.array):


    # Index order list - defines the order in which the pixels in a 3x3 frame are considered.
    idx = [(1, -1), (0, -1), (0, 1), (0, 0), (1, 0), (-1, 0), (-1, 1), (-1, -1), (1, -1)]

    debug = False

    height, width = image.shape

    # Store all minutiae
    bifurcations = []
    terminations = []

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # 3x3 frame extraction based on the previous, current and next values on x and y axis.
            frame = image[i - 1: i + 2, j - 1: j + 2]

            # Custom minutiae detection function.
            # Control for pixels found in the middle of the frame.
            # Once identified, it counts filled pixels separated by at least 1 empty pixel.
            pixel_list = [frame[idx[i]] * (1 - frame[idx[i + 1]]) for i in range(len(idx) - 1)]
            pixel_sum = frame[1, 1] * sum(pixel_list)

            # Based on http://airccse.org/journal/ijcseit/papers/2312ijcseit01.pdf
            # pixel_sum = .5 * sum([abs(frame[idx[i]] - frame[idx[i + 1]]) for i in range(len(indices) - 1)])

            if pixel_sum == 1:
                # Termination
                if debug:
                    # Displays a larger frame for debugging purposes.
                    print(f'Termination: {i}, {j}')
                    display_image(image[i - 2: i + 3, j - 2: j + 3])

                # Add termination coordinates
                terminations.append((i, j))

            elif pixel_sum == 3:
                # Bifurcation
                if debug:
                    # Displays a larger frame for debugging purposes.
                    print(f'Bifurcation: {i}, {j}')
                    display_image(image[i - 2: i + 3, j - 2: j + 3])

                # Add bifurcation coordinates
                bifurcations.append((i, j))

    return terminations, bifurcations


def clean_minutiae(image: np.array, minutiae: list) -> list:


    height, width = image.shape

    minutiae_clean = []
    for x, y in minutiae:
        # If there are directions in which the minutiae with x and y coordinates has only empty
        # pixels, that we label the minutiae as an image border and discard it.
        if (image[x, :y].sum() > 0) and (image[x, y + 1:].sum() > 0) and (image[:x, y].sum() > 0) and \
                (image[x + 1:, y].sum() > 0):
            minutiae_clean.append((x, y))

    return minutiae_clean


def remove_duplicate_minutiae(minutiae: list, min_distance: int = 10) -> list:

    if len(minutiae) == 0:
        return []
    
    from scipy.spatial.distance import cdist
    
    minutiae_array = np.array(minutiae)
    
    # 计算所有点之间的距离矩阵
    distances = cdist(minutiae_array, minutiae_array)
    
    # 创建掩码，标记要保留的点
    keep = np.ones(len(minutiae), dtype=bool)
    
    # 对于每个点，如果附近有更近的点，保留其中一个
    for i in range(len(minutiae)):
        if not keep[i]:
            continue
        
        # 找到所有距离太近的点（排除自己）
        too_close = np.where((distances[i] < min_distance) & (distances[i] > 0))[0]
        
        # 对于距离太近的点，只保留第一个（当前点），移除其他的
        for j in too_close:
            if keep[j]:  # 如果还没被移除
                # 保留索引较小的点（或者可以根据其他标准选择）
                if j > i:
                    keep[j] = False
    
    return [minutiae[i] for i in range(len(minutiae)) if keep[i]]


def eliminate_false_minutiae(minutiae: list, min_distance: int = 15, max_neighbors: int = 3, 
                              neighborhood_radius: int = 20) -> list:

    if len(minutiae) == 0:
        return []
    
    from scipy.spatial.distance import cdist
    
    minutiae_array = np.array(minutiae)
    
    # 计算所有点之间的距离矩阵
    distances = cdist(minutiae_array, minutiae_array)
    
    # 创建掩码，标记要保留的点
    keep = np.ones(len(minutiae), dtype=bool)
    
    # 对于每个点，检查其邻域
    for i in range(len(minutiae)):
        if not keep[i]:
            continue
        
        # 找到邻域内的所有点（在 neighborhood_radius 范围内）
        neighbors_in_radius = np.where((distances[i] <= neighborhood_radius) & (distances[i] > 0))[0]
        
        # 如果邻域内点太多，可能是虚假集中
        if len(neighbors_in_radius) > max_neighbors:
            # 检查是否有距离太近的点（小于 min_distance）
            too_close = np.where((distances[i] < min_distance) & (distances[i] > 0))[0]
            
            if len(too_close) > 0:
                # 在局部集中的点中，保留距离最远的点，移除其他的
                # 计算每个点到其他所有点的平均距离
                avg_distances = np.mean(distances[neighbors_in_radius], axis=1)
                
                # 保留平均距离最大的点（最不集中的点）
                best_idx = neighbors_in_radius[np.argmax(avg_distances)]
                
                # 移除其他点
                for j in neighbors_in_radius:
                    if j != best_idx and keep[j]:
                        keep[j] = False
            else:
                # 如果没有太近的点，但邻域内点太多，可能是真实的密集区域
                # 只移除那些距离小于 min_distance 的点
                for j in neighbors_in_radius:
                    if distances[i, j] < min_distance and keep[j] and j > i:
                        keep[j] = False
        else:
            # 如果邻域内点不多，检查是否有距离太近的点
            too_close = np.where((distances[i] < min_distance) & (distances[i] > 0))[0]
            
            # 对于距离太近的点，保留其中一个（保留索引较小的）
            for j in too_close:
                if keep[j] and j > i:
                    keep[j] = False
    
    return [minutiae[i] for i in range(len(minutiae)) if keep[i]]


def extract_tuple_profile(distances: list, m: tuple, minutiae: list) -> list:


    # Closest minutiae to the current minutiae
    closest_distances = sorted(distances)[1:6]
    closest_indices = [list(distances).index(d) for d in closest_distances]
    closest_minutiae = [minutiae[i] for i in closest_indices]

    # Unique pair ratios.
    # The 10 pairs used for computing the ratios
    # i-i1 : i-i2, i-i1 : i-i3, i-i1 : i-i4, i-i1 : i-i5,
    # i-i2 : i-i3, i-i2 : i-i4, i-i2 : i-i5
    # i-i3 : i-i4, i-i3 : i-i5
    # i-i4 : i-i5
    unique_pairs = list(combinations(closest_distances, 2))
    # 2 decimal rounded ratios of max of the two distances divided by their minimum.
    compute_ratios = [round(max(p[0], p[1]) / min(p[0], p[1]), 2) for p in unique_pairs]

    # Angle computation.
    minutiae_combinations = list(combinations(closest_minutiae, 2))

    # Angle between the segments drawn from m to the two other minutae with varying distances.
    minutiae_angles = [round(extract_angle((m, x), (m, y)), 2) for x, y in minutiae_combinations]

    return [compute_ratios, minutiae_angles]


def process_minutiae(image: np.array, min_distance: int = 10, eliminate_false: bool = True, 
                     false_min_distance: int = 15, max_neighbors: int = 3, return_types: bool = False):

    # Extract minutiae (CN = 1 for terminations, CN = 3 for bifurcations)
    terminations, bifurcations = extract_minutiae(image)

    # Post-processing border minutiae removal.
    terminations = clean_minutiae(image, terminations)
    bifurcations = clean_minutiae(image, bifurcations)

    # 记录类型信息
    # 类型：1=termination, 3=bifurcation
    termination_types = [1] * len(terminations)  # 类型1: termination
    bifurcation_types = [3] * len(bifurcations)  # 类型3: bifurcation

    # 合并所有 minutiae 和类型
    all_minutiae = terminations + bifurcations
    all_types = termination_types + bifurcation_types
    
    # 创建 minutiae 到类型的映射
    minutiae_to_type = dict(zip(all_minutiae, all_types))
    
    # 移除距离太近的重复点
    all_minutiae = remove_duplicate_minutiae(all_minutiae, min_distance=min_distance)
    
    # 更新类型列表（只保留剩余的 minutiae 的类型）
    all_types = [minutiae_to_type[m] for m in all_minutiae if m in minutiae_to_type]
    
    # 消除虚假 minutiae（基于距离阈值和局部集中度）
    if eliminate_false:
        # 再次创建映射（因为 all_minutiae 可能已经改变）
        minutiae_to_type = dict(zip(all_minutiae, all_types))
        
        original_count = len(all_minutiae)
        all_minutiae = eliminate_false_minutiae(
            all_minutiae, 
            min_distance=false_min_distance,
            max_neighbors=max_neighbors
        )
        
        # 更新类型列表（只保留剩余的 minutiae 的类型）
        all_types = [minutiae_to_type[m] for m in all_minutiae if m in minutiae_to_type]

    if return_types:
        return all_minutiae, all_types
    return all_minutiae


def generate_tuple_profile(minutiae: list) -> dict:


    distance_matrix = np.array([[euclidian_distance(i, j) for i in minutiae] for j in minutiae])

    tuples = {}

    for i, m in enumerate(minutiae):
        # When comparing two tuple profiles, one from base and one from test image,
        # they are the same if at least 2 ratios match (and angles).

        # This means that for the tuple profile i is found in a second image under a
        # different tuple's profile.

        # Angles are given a +/- 3.5 degree range to match. To match sourcing device discrepancies.
        ratios_angles = extract_tuple_profile(distance_matrix[i], m, minutiae)
        tuples[m] = np.round(ratios_angles, 2)

    return tuples


def minutiae_points(image: np.array):

    # ORB discretises the angle to increments of 2 * pi / 30 (12 degrees) and construct a lookup table of precomputed
    # BRIEF patterns. As long as the keypoint orientationis consistent across views, the correct set of points S
    # will be used to compute its descriptor.
    orb = cv2.ORB_create()

    # Use ORB to detect keypoints.
    points = orb.detect(image)

    # # Use minutiae extracted via crossing numbers technique as keypoints.
    # minutiae = process_minutiae(image)
    # points = [cv2.KeyPoint(y, x, 1) for (x, y) in minutiae]

    # Describe and compute descriptor extractor
    keypoints, descriptors = orb.compute(image, points)

    return keypoints, descriptors


def plot_minutiae_tree(image: np.array, points: list, size: int = 5, node_size: int = 20, graph_color: str = 'blue'):
    

    plt.figure(figsize=(size, size))
    plt.imshow(image)
    plt.grid(False)

    G = nx.Graph()

    # Create nodes for each coordinate pair
    for i, coord in enumerate(points):
        G.add_node(i, pos=(coord[1], coord[0]))

    # Create edges between subsequent nodes.
    G.add_edges_from([(i, i + 1) for i in range(len(points[:-1]))])

    nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=False, node_size=node_size, color=graph_color,
            edge_color=graph_color)

    plt.show()


def plot_minutiae(image: np.array, terminations: list = None, bifurcations: list = None, size: int = 5) -> None:
    

    # 如果只提供了一个列表（可能是 process_minutiae 返回的合并列表），将其作为 terminations
    if terminations is not None and bifurcations is None:
        # 检查是否是 process_minutiae 返回的格式（单个列表）
        if isinstance(terminations, list) and len(terminations) > 0:
            if isinstance(terminations[0], tuple) and len(terminations[0]) == 2:
                # 这可能是合并的 minutiae 列表，需要重新提取
                # 为了兼容，我们将它作为 terminations 处理
                pass  # 保持原样，作为 terminations 处理

    if bifurcations is None and terminations is None:
        raise Exception("INFO: No 'terminations' or 'bifurcations' parameter given. Nothing to plot.")
    
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(image, cmap='gray')
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # 反转 y 轴以匹配图像坐标
    ax.grid(False)
    ax.set_aspect('equal')

    if terminations is not None:
        print("INFO: Plotting terminations\' coordinates")
        for y, x in terminations:
            circle = plt.Circle((x, y), radius=3, linewidth=1.5, color='red', fill=False)
            ax.add_patch(circle)

    if bifurcations is not None:
        print("INFO: Plotting bifurcations\' coordinates")
        for y, x in bifurcations:
            circle = plt.Circle((x, y), radius=3, linewidth=1.5, color='blue', fill=False)
            ax.add_patch(circle)
    
    plt.tight_layout()
    plt.show()
