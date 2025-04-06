import numpy as np
import pickle
import trimesh
from typing import Literal

# cpp extension (named in setup.py and PYBIND11_MODULE)
import _meshiki


# helper functions
def normalize_mesh(vertices, bound=0.95):
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    ori_center = (vmax + vmin) / 2
    ori_scale = 2 * bound / np.max(vmax - vmin)
    vertices = (vertices - ori_center) * ori_scale
    return vertices


def load_mesh(path, bound=0.99, clean=True):
    
    # manually load obj to support quad (only load geom)
    if path.lower().endswith('.obj'):
        with open(path, "r") as f:
            lines = f.readlines()

        def parse_f_v(fv):
            # pass a vertex term of a face, return {v, vt, vn} (-1 if not provided)
            # supported forms:
            # f v1 v2 v3
            # f v1/vt1 v2/vt2 v3/vt3
            # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            # f v1//vn1 v2//vn2 v3//vn3
            xs = [int(x) - 1 if x != "" else -1 for x in fv.split("/")]
            xs.extend([-1] * (3 - len(xs)))
            return xs[0], xs[1], xs[2]

        vertices = []
        faces = []
        
        face_len_set = set()
        for line in lines:
            split_line = line.split()
            # empty line
            if len(split_line) == 0:
                continue
            prefix = split_line[0].lower()
            # v/vn/vt
            if prefix == "v":
                vertices.append([float(v) for v in split_line[1:]])
            elif prefix == "f":
                vs = split_line[1:]
                face = []
                for i in range(len(vs)): # allow polygon
                    face.append(parse_f_v(vs[i])[0])
                faces.append(face)
                face_len_set.add(len(face))
        
        vertices = np.array(vertices, dtype=np.float32)

        if len(face_len_set) == 1:
            faces = np.array(faces, dtype=np.int32)

    else:

        # use trimesh to load other formats
        _data = trimesh.load(path)
        # always convert scene to mesh, and apply all transforms...
        if isinstance(_data, trimesh.Scene):
            # print(f"[INFO] load trimesh: concatenating {len(_data.geometry)} meshes.")
            _concat = []
            # loop the scene graph and apply transform to each mesh
            scene_graph = _data.graph.to_flattened() # dict {name: {transform: 4x4 mat, geometry: str}}
            for k, v in scene_graph.items():
                name = v['geometry']
                if name in _data.geometry and isinstance(_data.geometry[name], trimesh.Trimesh):
                    transform = v['transform']
                    _concat.append(_data.geometry[name].apply_transform(transform))
            _mesh = trimesh.util.concatenate(_concat)
        else:
            _mesh = _data
        
        vertices = _mesh.vertices
        faces = _mesh.faces


    # normalize
    vertices = normalize_mesh(vertices, bound=bound)

    # clean 
    if clean:
        # will copy to c++ and back to python...
        vertices, faces = merge_close_vertices(vertices, faces)

    return vertices, faces


def write_mesh(path, vertices, faces):
    # manually write obj for supporting quad
    if path.lower().endswith('.obj'):
        with open(path, "w") as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for i, face in enumerate(faces):
                f.write("f")
                for v in face:
                    f.write(f" {v+1}")
                f.write("\n")
    else:
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(path)


def triangulate(faces):
    # faces: list of list of 3 or 4 int.
    return _meshiki.triangulate(faces)

def pad_to_quad(faces):
    # faces: list of list of 3 or 4 int.
    return _meshiki.pad_to_quad(faces)

def merge_close_vertices(vertices, faces, thresh=1e-8, verbose=False):
    # thresh: vertices inside this radius will be merged
    vertices, faces = _meshiki.merge_close_vertices(vertices, faces, thresh, verbose)
    vertices = np.asarray(vertices)
    return vertices, faces

def fps(points, num_points, start_idx=None, backend: Literal['naive', 'kdtree', 'kdline'] = 'kdline'):
    # points: np.array of shape (N, 3)
    if start_idx is None:
        start_idx = np.random.randint(0, len(points))

    if backend == 'naive':
        samples = _meshiki.fps(points, num_points, start_idx)
    elif backend == 'kdtree':
        samples = _meshiki.bucket_fps_kdtree(points, num_points, start_idx)
    elif backend == 'kdline': # fastest
        samples = _meshiki.bucket_fps_kdline(points, num_points, start_idx, 5)
    
    samples = np.asarray(samples)
    return samples


class Mesh:
    def __init__(self, vertices, faces, clean=False, verbose=False):
        self.impl = _meshiki.Mesh(vertices, faces, clean, verbose)
        self.sync_impl()

        # count face
        self.face_cnt = {}
        for face in self.faces:
            if len(face) not in self.face_cnt:
                self.face_cnt[len(face)] = 0
            self.face_cnt[len(face)] += 1

        self.trig_only = 3 in self.face_cnt and len(self.face_cnt) == 1

    def sync_impl(self):
        # copy back to self
        self.vertices, self.faces = self.impl.export_mesh()
        self.vertices = np.asarray(self.vertices)
    
    @staticmethod
    def load(path, clean=True, bound=0.99, verbose=False):
        vertices, faces = load_mesh(path, bound=bound, clean=clean)
        return Mesh(vertices, faces, clean, verbose)

    @property
    def verts(self):  # alias for vertices
        return self.vertices

    @property
    def num_quad(self):
        return self.impl.num_quad

    @property
    def quad_ratio(self):
        return self.impl.num_quad / self.impl.num_faces

    @property
    def rect_error(self):
        return self.impl.rect_error

    @property
    def num_components(self):
        return self.impl.num_components
    
    def quadrangulate(self, thresh_bihedral=45, thresh_convex=185):
        assert self.trig_only, "Only support quadrangulateing pure trimesh!"
        # run quadrangulation
        self.impl.quadrangulate(thresh_bihedral, thresh_convex)
        self.sync_impl()
    
    def polygonize(self, thresh_bihedral=1, thresh_convex=181, max_round=100):
        self.impl.polygonize(thresh_bihedral, thresh_convex, max_round)
        self.sync_impl()

    def salient_point_sample(self, num_points, thresh_bihedral=30):
        samples = self.impl.salient_point_sample(num_points, thresh_bihedral)
        samples = np.asarray(samples)
        return samples

    def uniform_point_sample(self, num_points):
        samples = self.impl.uniform_point_sample(num_points)
        samples = np.asarray(samples)
        return samples
    
    def repair_face_orientation(self):
        self.impl.repair_face_orientation()
        self.sync_impl()
    
    def explode(self, delta=0.1):
        self.impl.explode(delta)
        self.sync_impl()
        
    def export(self, path):
        # assume synced with impl
        write_mesh(path, self.vertices, self.faces)
    
    def smart_group_components(self):
        self.impl.smart_group_components()
        # this only changes the component of impl, we don't expose component in python
    
    def export_components_as_trimesh_scene(self):
        # we have to use trimesh scene for components
        components = self.impl.export_components()
        scene = trimesh.Scene()
        for component in components:
            verts, faces = component
            mesh = trimesh.Trimesh(verts, faces)
            scene.add_geometry(mesh)
        return scene