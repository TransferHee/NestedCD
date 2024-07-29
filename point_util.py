import numpy as np
import open3d as o3d

from numba import jit
from numba import njit, prange


@jit(nopython=True)
def rotate_nb(s, theta=0, axis='x'):
    """
    Counter Clock wise rotation of a vector s, along the axis by angle theta
    s:= array/list of scalars. Contains the vector coordinates [x,y,z]
    theta:= scalar, <degree> rotation angle for counterclockwise rotation
    axis:= str, rotation axis <x,y,z>
    """
    theta = np.radians(theta)  # degree -> radians
    r = [0., 1., 2.]

    if axis.lower() == 'x':
        r = [s[0],
             s[1] * np.cos(theta) - s[2] * np.sin(theta),
             s[1] * np.sin(theta) + s[2] * np.cos(theta)]
    elif axis.lower() == 'y':
        r = [s[0] * np.cos(theta) + s[2] * np.sin(theta),
             s[1],
             -s[0] * np.sin(theta) + s[2] * np.cos(theta)]
    elif axis.lower() == 'z':
        r = [s[0] * np.cos(theta) - s[1] * np.sin(theta),
             s[0] * np.sin(theta) + s[1] * np.cos(theta),
             s[2]]
    else:
        r = [0., 1., 2.]

    r = np.array(r)

    return r


# rotate point x, y, z degree
@njit(parallel=True)
def rotate_points_nb(points, x=0, y=0, z=0):
    points2 = np.zeros(points.shape)

    for idx in prange(len(points)):
        point = points[idx]
        point = rotate_nb(point, x, 'x')
        point = rotate_nb(point, y, 'y')
        point = rotate_nb(point, z, 'z')

        points2[idx] = point

    return points2


def rotate_pts(pts):
    point = pts.copy()
    theta = np.pi * 2 * np.random.choice(24) / 24
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    point[:, [0, 2]] = point[:, [0, 2]].dot(rotation_matrix)  # random rotation (x,z)
    return point


def jitter_pts(pts, sigma=0.01, clip=0.02):
    point = pts.copy()
    N, C = point.shape
    point += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return point


def translate_pts(pts):
    point = pts.copy()
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    point = np.add(np.multiply(point, xyz1), xyz2).astype('float32')
    return point


def read_3dp(file_path):
    full_data = np.loadtxt(file_path)

    point = full_data[:, 2:5]
    color = full_data[:, 5:] / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point)
    pcd.colors = o3d.utility.Vector3dVector(color)

    return pcd


def append_coordinate(points, colors=None):
    if colors is None:
        colors = np.zeros_like(points)

    x_arr = [[x, 0, 0] for x in np.linspace(0, 1, num=100)]
    x_color = [[1, 0, 0]] * len(x_arr)

    y_arr = [[0, y, 0] for y in np.linspace(0, 1, num=100)]
    y_color = [[0, 1, 0]] * len(y_arr)

    z_arr = [[0, 0, z] for z in np.linspace(0, 1, num=100)]
    z_color = [[0, 0, 1]] * len(z_arr)

    new_points = np.concatenate((points, x_arr, y_arr, z_arr))
    colors = np.concatenate((colors,
                             x_color, y_color, z_color))

    return new_points, colors


def rotate(s, theta=0, axis='x'):
    """
    Counter Clock wise rotation of a vector s, along the axis by angle theta
    s:= array/list of scalars. Contains the vector coordinates [x,y,z]
    theta:= scalar, <degree> rotation angle for counterclockwise rotation
    axis:= str, rotation axis <x,y,z>
    """
    theta = np.radians(theta)  # degree -> radians
    r = 0
    if axis.lower() == 'x':
        r = [s[0],
             s[1] * np.cos(theta) - s[2] * np.sin(theta),
             s[1] * np.sin(theta) + s[2] * np.cos(theta)]
    elif axis.lower() == 'y':
        r = [s[0] * np.cos(theta) + s[2] * np.sin(theta),
             s[1],
             -s[0] * np.sin(theta) + s[2] * np.cos(theta)]
    elif axis.lower() == 'z':
        r = [s[0] * np.cos(theta) - s[1] * np.sin(theta),
             s[0] * np.sin(theta) + s[1] * np.cos(theta),
             s[2]]
    else:
        print("Error! Invalid axis rotation")

    return r


# rotate point x, y, z degree
def rotate_points(points, x=0, y=0, z=0):
    point_arr = []

    for point in points:
        point = rotate(point, x, 'x')
        point = rotate(point, y, 'y')
        point = rotate(point, z, 'z')

        point_arr.append(point)

    points = np.array(point_arr)

    return points


def view_pcd(points=None, colors=None, pcd=None):
    if type(pcd) == o3d.cpu.pybind.geometry.PointCloud:
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)

    if colors is None:
        colors = np.zeros_like(points)
    # points, colors = append_coordinate(points, colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.7)])
    o3d.visualization.draw_geometries([pcd, ])


def write_pcd(points, colors, file_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(file_path, pcd)
