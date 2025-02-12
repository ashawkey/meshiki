import sys
sys.path.append('.')

import os
import trimesh
import numpy as np
import argparse

import kiui
from meshiki import Mesh, load_mesh, triangulate

parser = argparse.ArgumentParser()
parser.add_argument('mesh', type=str, help='path to the mesh file')
parser.add_argument('--output', type=str, default='output.obj', help='path to the output file')
opt = parser.parse_args()

# mesh
vertices, faces = load_mesh(opt.mesh, clean=True)
tri_faces = triangulate(faces)
trimesh.Trimesh(vertices, tri_faces).export(opt.output)
