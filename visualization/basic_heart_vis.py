from tools.vis_tools import get_heart, get_infarct
import pyvista as pv


PATIENT_NUMBER = 32
HEART_DATA_DIR = "../data/vtk_csv"
INFARCT_DATA_DIR = "../data/vtk_csv"


heart_points, scalars = get_heart(PATIENT_NUMBER, HEART_DATA_DIR)
infarct_points = get_infarct(PATIENT_NUMBER, INFARCT_DATA_DIR)

lv_mesh = pv.PolyData(heart_points[scalars == 3])
rv_mesh = pv.PolyData(heart_points[scalars == 6])
epi_mesh = pv.PolyData(heart_points[scalars == 4])
inf_mesh = pv.PolyData(infarct_points)

p = pv.Plotter()
p.add_mesh(lv_mesh, color="darkblue", opacity=1)
p.add_mesh(rv_mesh, color="red", opacity=1)
p.add_mesh(epi_mesh, color="lightblue", opacity=0.9)
p.add_mesh(inf_mesh, color="black", opacity=0.2)
p.show()