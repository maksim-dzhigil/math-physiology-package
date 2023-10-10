import dolfin as dol
from vedo.dolfin import plot
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import pyvista as pv
import os

XDMF_FOLDER = "/home/makism/Desktop/projs/py/fenics_folder/epi_endo_xdmf"
EPI_ENDO_SOLUTION_FOLDER = "/home/makism/Desktop/projs/py/fenics_folder/laplace_solutions_epi_endo"
LV_RV_SOLUTION_FOLDER = "/home/makism/Desktop/projs/py/fenics_folder/laplace_solutions_lv_rv"

PATIENT = 10
BC_1 = 100.
BC_2 = 200.


def get_fenics_mesh(patient: int) -> dol.cpp.mesh.Mesh:
    mesh = dol.Mesh()
    with dol.XDMFFile(os.path.join(XDMF_FOLDER, f"patient_{patient:03}.xdmf")) as patient_file:
        patient_file.read(mesh)
    return mesh
 

def get_xyz_from_vtk(patient: int) -> (np.ndarray, np.ndarray):
    reader = vtk.vtkXdmfReader()
    reader.SetFileName(os.path.join(XDMF_FOLDER, f"patient_{patient:03}.xdmf"))
    reader.Update()
    surface = reader.GetOutput()
    sep_mask = vtk_to_numpy(surface.GetPointData().GetArray("For Separating"))
    heart = vtk_to_numpy(surface.GetPoints().GetData())

    return heart, sep_mask


def main(patient: int, 
         bc_value_1: float, 
         bc_value_2: float,
         save: bool = True,
         show: bool = True):
    mesh = get_fenics_mesh(patient)
    heart, sep_mask = get_xyz_from_vtk(patient)

    V = dol.FunctionSpace(mesh, 'P', 1)
    u_D_1 = dol.Constant(bc_value_1)
    u_D_2 = dol.Constant(bc_value_2)

    lv = heart[(sep_mask == 3)]
    rv = heart[(sep_mask == 6)]
    endo = heart[(sep_mask == 3) | (sep_mask == 6)]
    epi = heart[(sep_mask == 4) | (sep_mask == 7)]


    def boundary_lv(x, on_boundary):
        return on_boundary and (x not in rv)


    def boundary_rv(x, on_boundary):
        return on_boundary and (x not in lv)


    def boundary_endo(x, on_boundary):
        return on_boundary and (x not in epi)


    def boundary_epi(x, on_boundary):
        return on_boundary and (x not in endo)


    bc_endo = dol.DirichletBC(V, u_D_1, boundary_endo)
    bc_epi = dol.DirichletBC(V, u_D_2, boundary_epi)
    bc_lv = dol.DirichletBC(V, u_D_1, boundary_lv)
    bc_rv = dol.DirichletBC(V, u_D_2, boundary_rv)

    bc_epi_endo = [bc_endo, bc_epi]
    bc_lv_rv = [bc_lv, bc_rv]

    u = dol.TrialFunction(V)
    v = dol.TestFunction(V)
    f = dol.Constant(0.)
    a = dol.dot(dol.grad(u), dol.grad(v))*dol.dx
    L = f*v*dol.dx

    u_epi_endo = dol.Function(V)
    u_lv_rv = dol.Function(V)

    dol.solve(a == L, u_epi_endo, bc_epi_endo)
    dol.solve(a == L, u_lv_rv, bc_lv_rv)

    if save:
        solution_epi_endo = dol.File(os.path.join(EPI_ENDO_SOLUTION_FOLDER, f"epi-endo_laplace_for_patient_{patient:03}_.pvd"))
        solution_epi_endo << u_epi_endo

        solution_lv_rv = dol.File(os.path.join(LV_RV_SOLUTION_FOLDER, f"lv-rv_laplace_for_patient_{patient:03}_.pvd"))
        solution_lv_rv << u_lv_rv

    if show:
        plot(u_epi_endo)
        plot(u_lv_rv)


if __name__ =="__main__":
    main(PATIENT, BC_1, BC_2, show=False)