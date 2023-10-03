import meshio
from os.path import join, isfile, exists
from os import listdir, mkdir


def vtu2xdmf(vtu_path, xdmf_path, patient_number):
    # vtu_path = join(vtu_path, f"heart.colored.{patient_number:03}.vtu")
    heart_geometry = meshio.read(vtu_path)
    if not exists(xdmf_path):
        mkdir(xdmf_path)
    
    xdmf_path = join(xdmf_path, f"patient_{patient_number:03}.xdmf")

    meshio.write(xdmf_path, heart_geometry)


def convert_folder(vtu_path, xdmf_path):
    files_to_convert = sorted([f for f in listdir(vtu_path) if isfile(join(vtu_path, f))])
    for index, file in enumerate(files_to_convert):
        vtu2xdmf(join(vtu_path, file), xdmf_path, index+1)
