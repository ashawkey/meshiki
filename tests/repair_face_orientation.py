import sys
sys.path.append('.')

import argparse

import kiui
from meshiki import Mesh

parser = argparse.ArgumentParser()
parser.add_argument('mesh', type=str, help='path to the mesh file')
parser.add_argument('--output', type=str, default='output.obj', help='path to the output file')
parser.add_argument('--verbose', action='store_true', help='print verbose output')
opt = parser.parse_args()

# mesh
mesh = Mesh.load(opt.mesh, verbose=opt.verbose)
mesh.repair_face_orientation()
mesh.export(opt.output)
