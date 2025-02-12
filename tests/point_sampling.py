import sys
sys.path.append('.')

import os
import time
import trimesh
import numpy as np
import argparse
import math
from collections import defaultdict

import kiui
from meshiki import Mesh, fps, load_mesh, triangulate, merge_close_vertices

parser = argparse.ArgumentParser()
parser.add_argument('mesh', type=str, help='path to the mesh file')
parser.add_argument('--output', type=str, default='output', help='name to the output file')
opt = parser.parse_args()

# mesh
vertices, faces = load_mesh(opt.mesh, clean=True)
# triangulate
faces = triangulate(faces)
mesh = Mesh(vertices, faces, verbose=True)
mesh.export(f"{opt.output}.obj")

N_SALIENT = 64000
N_UNIFORM = 128000
N_FPS = 8000

# salient points
_t0 = time.time()
salient_points = mesh.salient_point_sample(N_SALIENT)
assert len(salient_points) == N_SALIENT, f"{len(salient_points)}"
trimesh.PointCloud(salient_points).export(f"{opt.output}_salient_points.obj")
print(f"Salient points: {time.time()-_t0:.2f}s")

# uniform points
_t0 = time.time()
uniform_points = mesh.uniform_point_sample(N_UNIFORM)
assert len(uniform_points) == N_UNIFORM, f"{len(uniform_points)}"
trimesh.PointCloud(uniform_points).export(f"{opt.output}_uniform_points.obj")
print(f"Uniform points: {time.time()-_t0:.2f}s")

# FPS of salient points
_t0 = time.time()
fps_points = fps(salient_points, N_FPS, backend='kdline')
assert len(fps_points) == N_FPS, f"{len(fps_points)}"
trimesh.PointCloud(fps_points).export(f"{opt.output}_fps_salient_points.obj")
print(f"FPS of salient points: {time.time()-_t0:.2f}s")

# FPS of uniform points
_t0 = time.time()
fps_points = fps(uniform_points, N_FPS, backend='kdline')
assert len(fps_points) == N_FPS, f"{len(fps_points)}"
trimesh.PointCloud(fps_points).export(f"{opt.output}_fps_points_uniform.obj")
print(f"FPS of uniform points: {time.time()-_t0:.2f}s")