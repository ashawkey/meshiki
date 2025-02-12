# Meshiki

A collection of unusual mesh processing algorithms.

## Install

```bash
# from pypi, will build extension at first time
pip install meshiki

# locally
cd meshiki
pip install . 
```

## Usage

### Trigs-to-Quads

```bash
          Triangulate       We want this!
    Quads ----------> Trigs ------------> Quads
(obj, blender)   (glb, fbx, ...)
```
This algorithm is aimed for converting a triangulated quad-dominant mesh back to a mixed tri/quad mesh, with as many as possible reasonable quad faces.
Our implementation is based on maximum weighted graph matching, and is usually better compared to the built-in tool (`Edit Mode -> Face -> Tris to Quads`) of blender.

```python
from meshiki import Mesh

mesh = Mesh.load('mesh.glb', verbose=False)
mesh.quadrangulate()
mesh.export('mesh.obj') # must use obj for quad faces
```

### Salient point sampling

This algorithm samples salient points from mesh surface as proposed in [Dora](https://github.com/Seed3D/Dora).

```python
from meshiki import Mesh, fps, load_mesh, triangulate

# load mesh
vertices, faces = load_mesh(mesh_path, clean=True)
# make sure it's pure-trig
faces = triangulate(faces)
mesh = Mesh(vertices, faces)
# sample 64K salient points
salient_points = mesh.salient_point_sample(64000, thresh_bihedral=30) # np.ndarray, [64000, 3]
```

We also implement uniform sampling and furthest point sampling:
```python
# sample 128K uniform points
uniform_points = mesh.uniform_point_sample(128000) # np.ndarray, [128000, 3]
# use FPS to subsample 8K points from uniform points
fps_points = fps(uniform_points, N_FPS, backend='kdline') # np.ndarray, [8000, 3]
```

## Acknowledgement

* [QuickFPS](https://github.com/hanm2019/bucket-based_farthest-point-sampling_CPU) and [fpsample](https://github.com/leonardodalinky/fpsample) for fast furthest point sampling.
* [PCG32](https://github.com/wjakob/pcg32).
* [nanoflann](https://github.com/jlblancoc/nanoflann).