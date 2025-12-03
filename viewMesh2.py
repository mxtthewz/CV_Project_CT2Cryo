#!/usr/bin/env python3


import pyvista as pv
import sys
import os

meshName = "mesh_colored"+str(170)+".ply"

def view_ply(filename="mesh_colored.ply"):
    
    print(f"Loading mesh")
    mesh = pv.read(filename)
    
    print(f"Mesh info:")
    print(f"  - Vertices: {mesh.n_points:,}")
    print(f"  - Faces: {mesh.n_cells:,}")
    print(f"  - Bounds: {mesh.bounds}")
    print(f"  - Available data: {list(mesh.point_data.keys())}")
    
    # Color attempts
    has_rgb = ("RGB" in mesh.point_data)
    sepColors = all(c in mesh.point_data for c in ["red", "green", "blue"])
    
    plotter = pv.Plotter()
    
    if has_rgb:
        print("  - Displaying with RGB vertex colors")
        plotter.add_mesh(mesh, show_edges=False, scalars="RGB", rgb=True)
    elif sepColors:
        rgb = np.stack([mesh.point_data["red"], 
                       mesh.point_data["green"], 
                       mesh.point_data["blue"]], axis=1)
        mesh.point_data["RGB"] = rgb
        plotter.add_mesh(mesh, show_edges=False, scalars="RGB", rgb=True)
    else:
        plotter.add_mesh(mesh, show_edges=False, color="lightblue")
    
    plotter.add_axes()
    plotter.add_text(f"Mesh: {os.path.basename(filename)}", position='upper_left', font_size=10)
    plotter.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ply_file = sys.argv[1]
    else:
        ply_file = meshName
    
    view_ply(ply_file)