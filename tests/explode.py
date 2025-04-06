import sys
sys.path.append('.')

import os
import trimesh
import numpy as np
import argparse

import kiui
from meshiki import Mesh, normalize_mesh

parser = argparse.ArgumentParser()
parser.add_argument('mesh', type=str, help='path to the mesh file')
parser.add_argument('--verbose', action='store_true', help='print verbose output')
parser.add_argument('--trimesh', action='store_true', help='use trimesh and glb group')
parser.add_argument('--output', type=str, default='output_exploded.glb', help='path to the output file')
opt = parser.parse_args()

# the allowed minimum distance between two objects (such that the SDF conversion could separate them)
MIN_DIST = 2 * 2 / 512

if opt.trimesh:
    mesh = trimesh.load(opt.mesh)

    if isinstance(mesh, trimesh.Scene):
        print(f'[INFO] scene: {len(mesh.geometry)} meshes')
        scene = mesh

        ### we don't use the scene graph API, something is wrong and I cannot find out.
        ### instead, we always apply transform to vertices

        ### box normalize into [-1, 1]
        bounds = scene.bounds # [2, 3]
        center = scene.centroid # [3]
        # print(f'[INFO] center = {center}, bounds = {bounds}')
        scale = 0.95 * 1 / np.max(bounds[1] - bounds[0])
        transform_normalize = np.eye(4)
        transform_normalize[:3, 3] = -center
        transform_normalize[:3, :3] = np.diag(np.array([scale, scale, scale]))
        # print(transform_normalize)
        scene.apply_transform(transform_normalize)

        ### apply transform to vertices
        meshes = {}
        scene_graph = scene.graph.to_flattened()
        for k, v in scene_graph.items():
            name = v['geometry']
            if name in scene.geometry and isinstance(scene.geometry[name], trimesh.Trimesh):
                transform = v['transform']
                meshes[name] = scene.geometry[name].apply_transform(transform)
                # drop all textures
                meshes[name].visual = trimesh.visual.ColorVisuals()

        ### explode
        
        # sort objects by distance to center
        name_to_centers = {}
        for name, mesh in meshes.items():
            vmin = np.min(mesh.vertices, axis=0)
            vmax = np.max(mesh.vertices, axis=0)
            name_to_centers[name] = (vmin + vmax) / 2

        name_with_dist = [] # [(name, dist), ...]
        for name, mesh_center in name_to_centers.items():
            dist = np.linalg.norm(mesh_center)
            name_with_dist.append((name, dist))
        name_with_dist.sort(key=lambda x: x[1])
        center_name = name_with_dist[0][0]
        center_pushing = name_to_centers[center_name]
        # print(f'[INFO] center object: {center_name}, pushing center: {center_pushing}')

        # get collision manager (requires pip install python-fcl)
        # manager, node_name_to_colliders = trimesh.collision.scene_to_collision(scene) 
        manager = trimesh.collision.CollisionManager() # will add colliders one-by-one

        # the first object will not be exploded
        manager.add_object(center_name, meshes[center_name])  # DO NOT USE transform! something is buggy...

        # explode other objects, make sure each new object is not colliding with all previous ones
        for name, dist in name_with_dist[1:]:

            mesh_cur = meshes[name]
            
            # get center of the object
            center_cur = name_to_centers[name]

            # get the pushing direction
            direction = center_cur - center_pushing
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            # decide the pushing distance
            # cannot use binary search! it's not monotonic!
            # simply use a linear search...
            for x in np.arange(0, 2, MIN_DIST):
                # push away a little bit
                mesh_cur.vertices += x * direction

                # check if collide
                is_collide = manager.in_collision_single(mesh_cur)
                # print(f'[test] {name} at {x}: collide = {is_collide}')
                if is_collide:
                    continue

                # check if the distance is too small
                min_dist = manager.min_distance_single(mesh_cur)
                # print(f'[test] {name} at {x}: min_dist = {min_dist}, MIN_DIST = {MIN_DIST}')

                # min_dist, data = manager.min_distance_single(mesh_cur, return_data=True)
                # real_min_dist = np.linalg.norm(data.point(list(data.names)[0]) - data.point(list(data.names)[1]))
                # assert abs(min_dist / real_min_dist)  == 1  # if we use transform in manager, it will be wrong...

                if min_dist <= MIN_DIST:
                    continue

                # if not collide and the distance is large enough, we can use this distance
                # print(f'[INFO] {name} at {x}: accepted')
                break
            
            # add the object to the manager
            manager.add_object(name, mesh_cur)

        ### convert to a single mesh
        mesh = trimesh.util.concatenate(list(meshes.values()))
    else:
        print('[INFO] no groups in glb, cannot explode')
        mesh.vertices = normalize_mesh(mesh.vertices)
    mesh.export(opt.output)
else:
    mesh = Mesh.load(opt.mesh, verbose=opt.verbose, clean=False)
    mesh.explode(0.1)
    mesh.export(opt.output)