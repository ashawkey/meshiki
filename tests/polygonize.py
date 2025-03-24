import sys
sys.path.append('.')

import os
import trimesh
import numpy as np
import argparse

import kiui
from meshiki import Mesh

parser = argparse.ArgumentParser()
parser.add_argument('mesh', type=str, help='path to the mesh file')
parser.add_argument('--verbose', action='store_true', help='print verbose output')
parser.add_argument('--output', type=str, default='output.obj', help='path to the output file')
opt = parser.parse_args()


# mesh
mesh = Mesh.load(opt.mesh, verbose=opt.verbose)
mesh.polygonize(thresh_bihedral=10, thresh_convex=355, max_round=100)
mesh.export(opt.output)
