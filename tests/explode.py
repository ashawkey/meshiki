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
parser.add_argument('--output', type=str, default='output_exploded.glb', help='path to the output file')
parser.add_argument('--max_components', type=int, default=100, help='maximum number of components to explode')
opt = parser.parse_args()

# the allowed minimum distance between two objects (such that the SDF conversion could separate them)
MIN_DIST = 2 * 2 / 512

mesh = trimesh.load(opt.mesh)

def bottom_up_merge_geometry(scene: trimesh.Scene, max_components: int):
    # inplace merge scene by bottom-up merging nodes in the scene graph
    print(f'[INFO] bottom-up merge scene: num_geom = {len(scene.geometry)}')
    node_to_children = scene.graph.transforms.children
    node_to_depth = {}
    queue = [('world', 0)]
    while len(queue) > 0:
        node, depth = queue.pop(0)
        if node not in node_to_depth:
            node_to_depth[node] = depth
            if node in node_to_children:
                for child in node_to_children[node]:
                    queue.append((child, depth + 1))
    depth_to_nodes = {}
    for node, depth in node_to_depth.items():
        if depth not in depth_to_nodes:
            depth_to_nodes[depth] = []
        depth_to_nodes[depth].append(node)
    max_depth = len(depth_to_nodes) - 1
    # print(f'[INFO] max depth = {max_depth}')
    # for depth, nodes in depth_to_nodes.items():
    #     print(f'[INFO] depth {depth} has {len(nodes)} nodes')

    # merge nodes from bottom to top
    num_geom = len(scene.geometry)
    while max_depth > 0:
        nodes_to_merge = depth_to_nodes[max_depth - 1]
        # print(f'[INFO] merge depth = {max_depth - 1}, nodes to merge = {len(nodes_to_merge)}, num_geom = {num_geom}')
        cnt = {}
        for node in nodes_to_merge:
            if node not in node_to_children: 
                continue
            subscene = scene.subscene(node)
            
            # merge all geometries of this node
            if len(subscene.geometry) >= 1:
                
                if len(subscene.geometry) not in cnt:
                    cnt[len(subscene.geometry)] = 0
                cnt[len(subscene.geometry)] += 1

                for geom_name in subscene.geometry:
                    scene.delete_geometry(geom_name)

                # inplace modify the node to avoid breaking the scene graph
                geom_merged = subscene.to_geometry() # might be an empty mesh if the node doesn't have geometry...
                if 'geometry' in scene.graph.transforms.node_data[node]:
                    geom_name = scene.graph.transforms.node_data[node]['geometry']
                else:
                    geom_name = 'geom_' + node
                    scene.graph.transforms.node_data[node]['geometry'] = geom_name
                scene.geometry[geom_name] = geom_merged

                # update the number of geometries
                num_geom -= len(subscene.geometry) - 1
            
            # delete all children of this node
            for child in node_to_children[node]:
                scene.graph.transforms.remove_node(child)
        # print(f'[INFO] cnt = {cnt}')
        max_depth -= 1
        if num_geom <= max_components:
            break
    print(f'[INFO] after merging, num_geom = {num_geom}')

if isinstance(mesh, trimesh.Scene) and len(mesh.geometry) > 1:
    print(f'[INFO] scene: {len(mesh.geometry)} meshes')
    scene = mesh

    # if there are too many components, we try to merge components based on the depth in scene graph
    if len(scene.geometry) > opt.max_components:
        # not always good... highly dependent on the scene graph...
        bottom_up_merge_geometry(scene, opt.max_components)
            
else:
    if isinstance(mesh, trimesh.Scene): mesh = mesh.to_mesh()
    mesh = Mesh(mesh.vertices, mesh.faces, verbose=opt.verbose, clean=False)
    mesh.smart_group_components()
    scene = mesh.export_components_as_trimesh_scene()
    print(f'[INFO] mesh: {len(scene.geometry)} components')

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
        mesh: trimesh.Trimesh = scene.geometry[name].apply_transform(transform)
        # drop all textures since we only need geom for parted data
        mesh.visual = trimesh.visual.ColorVisuals()
        # clean up
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        mesh.update_faces(mesh.unique_faces() & mesh.nondegenerate_faces())
        mesh.fix_normals()
        meshes[name] = mesh

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
    # cannot use binary search! it's not monotonic! simply use a linear search...
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

### convert to a single mesh and export as glb
mesh = trimesh.util.concatenate(list(meshes.values()))
mesh.export(opt.output)