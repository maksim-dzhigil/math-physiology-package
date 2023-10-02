from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from sklearn.neighbors import KDTree
from scipy.spatial import Delaunay
from sklearn.model_selection import train_test_split
from vtk import vtkPolyDataReader, vtkXMLUnstructuredGridReader
from pandas import read_csv
from os.path import join
import pyvista as pv
import numpy as np



def get_xyz4apex(path, patient):
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(join(path, f"heart.colored.{patient:03}.vtu"))
    reader.Update()
    data = reader.GetOutput()
    apex_base = vtk_to_numpy(data.GetPointData().GetArray('Apex_base'))
    lv_colors = vtk_to_numpy(data.GetPointData().GetArray('LV colors'))

    apex0 = vtk_to_numpy(data.GetPoints().GetData())[np.where(apex_base == np.min(apex_base[np.where(lv_colors == 3)]))]
    apex1 = vtk_to_numpy(data.GetPoints().GetData())[np.where(apex_base == np.max(apex_base[np.where(lv_colors == 3)]))]

    xyz_apex0 = np.array([[np.average(apex0[:, 0]), np.average(apex0[:, 1]), np.average(apex0[:, 2])]])
    xyz_apex1 = np.array([[np.average(apex1[:, 0]), np.average(apex1[:, 1]), np.average(apex1[:, 2])]])

    return xyz_apex0, xyz_apex1


def get_bbox_bounds(path, patient):
    if "vtu" in path:
        path = join(path, f"heart.colored.{patient:03}.vtu")
    elif "vtk" in path:
        path = join(path, f"{patient}.vtk")
    model = pv.read(path)

    xmin, xmax, ymin, ymax, zmin, zmax = model.bounds

    min_xyz = np.array([[xmin, ymin, zmin]])
    max_xyz = np.array([[xmax, ymax, zmax]])

    return min_xyz, max_xyz


def extract_vtk_file(path, patient):
    reader = vtkPolyDataReader()
    reader.SetFileName(join(path, f"{patient}.vtk"))
    reader.Update()
    data = reader.GetOutput()
    heart_points = vtk_to_numpy(data.GetPoints().GetData())
    scalars = vtk_to_numpy(data.GetPointData().GetScalars())

    return data, heart_points, scalars


def kdtree_maker(points):
    if len(points) == 1:
        return KDTree(points[0], leaf_size=2)
    kdtrees = []
    for point_cloud in points:
        current_tree = None
        if point_cloud is not None and point_cloud.shape[0] != 0:
            current_tree = KDTree(point_cloud, leaf_size=2)
        kdtrees.append(current_tree)
    return kdtrees


def get_infarct_points(path, patient):
    infarct_data = read_csv(join(path, f"{patient}.csv"))[["x", "y", "z"]]
    infarct_points = np.array(infarct_data).astype("float32")
    return infarct_points


def get_infarct_mask(heart_points, infarct_points):
    heart_tree = kdtree_maker([heart_points])
    dist, ind = heart_tree.query(infarct_points)
    infarct_inds = [i for d, i in zip(dist, ind) if d < 1]
    infarct_mask = np.array([1 if i in infarct_inds else 0 for i in range(heart_points.shape[0])])

    return infarct_mask


def split_infarct_mask(infarct_mask, scalars, infarct=True):
    # return binary mask for [infarct / non-infarct] areas of the heart
    # 3 - left ventricular endocardium
    # 4 - epicardium
    # 6 - right ventricular endocardium
    if infarct:
        lv_mask = np.array(scalars == 3) & np.array(infarct_mask == 1)
        epi_mask = np.array(scalars == 4) & np.array(infarct_mask == 1)
        rv_mask = np.array(scalars == 6) & np.array(infarct_mask == 1)
    else:
        lv_mask = np.array(scalars == 3) & np.array(infarct_mask == 0)
        epi_mask = np.array(scalars == 4) & np.array(infarct_mask == 0)
        rv_mask = np.array(scalars == 6) & np.array(infarct_mask == 0)
    return lv_mask, epi_mask, rv_mask


def dists_inds_getting(points_cloud, tree_1, tree_2):
    # there are [infarct / non-infarct] points in [right / left] ven endo area -->
    # 1. find distances between ([right / left] ven endo points) and ([left / right] ven endo points)
    # 2. find distances between ([right / left] ven endo points) and (epi points)
    # 3. get minimal distances between 1st and 2nd steps
    # distances from 3rd step are distances [infarct / non-infarct] points for ven endo
    pc_dists_1, pc_inds_1 = np.array([]), np.array([])
    pc_dists_2, pc_inds_2 = np.array([]), np.array([])
    actual_dists = np.array([])
    if points_cloud.shape[0] != 0:
        if tree_1 is not None:
            pc_dists_1, pc_inds_1 = tree_1.query(points_cloud)
        if tree_2 is not None:
            pc_dists_2, pc_inds_2 = tree_2.query(points_cloud)
        actual_dists = np.minimum(pc_dists_1, pc_dists_2)

    # for left_ven points and right_ven kdtree returns indices of right_ven_points
    pc_only_inds_1 = np.array([i for d, i in zip(pc_dists_1, pc_inds_1) if d in actual_dists])
    pc_only_dists_1 = np.array([d for d, i in zip(pc_dists_1, pc_inds_1) if d in actual_dists])
    pc_only_inds_2 = np.array([i for d, i in zip(pc_dists_2, pc_inds_2) if d in actual_dists])
    pc_only_dists_2 = np.array([d for d, i in zip(pc_dists_2, pc_inds_2) if d in actual_dists])
    return pc_only_dists_1, pc_only_inds_1, pc_only_dists_2, pc_only_inds_2


def actual_distances(points_cloud, tree_1, tree_2):
    # there are [infarct / non-infarct] points in [right / left] ven endo area -->
    # 1. find distances between ([right / left] ven endo points) and ([left / right] ven endo points)
    # 2. find distances between ([right / left] ven endo points) and (epi points)
    # 3. get minimal distances between 1st and 2nd steps
    # distances from 3rd step are distances [infarct / non-infarct] points for ven endo
    pc_dists_1, _ = np.array([]), np.array([])
    pc_dists_2, _ = np.array([]), np.array([])
    actual_dists = np.array([])
    if points_cloud.shape[0] != 0:
        if tree_1 is not None:
            pc_dists_1, _ = tree_1.query(points_cloud)
        if tree_2 is not None:
            pc_dists_2, _ = tree_2.query(points_cloud)
        actual_dists = np.minimum(pc_dists_1, pc_dists_2)

    return actual_dists
