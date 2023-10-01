from vtk import vtkPolyDataReader
from os.path import join
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from pandas import read_csv


def get_heart(patient_number, path):
    reader = vtkPolyDataReader()
    reader.SetFileName(join(path, f"{patient_number}.vtk"))
    reader.Update()
    geometry = reader.GetOutput()
    scalars = vtk_to_numpy(geometry.GetPointData().GetScalars())
    heart = vtk_to_numpy(geometry.GetPoints().GetData())

    return heart, scalars


def get_infarct(patient_number, path):
    infarct_data = read_csv(join(path, f"{patient_number}.csv"))[["x", "y", "z"]]
    infarct_points = np.array(infarct_data).astype("float32")
    return infarct_points
