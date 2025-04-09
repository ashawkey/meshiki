import sys
sys.path.append('.')

import os
import glob
import tqdm
import trimesh
import argparse
import numpy as np

import kiui
from meshiki import Mesh

parser = argparse.ArgumentParser()
parser.add_argument('test_path', type=str, help='path to the mesh file or folder')
parser.add_argument('--verbose', action='store_true', help='print verbose output')
parser.add_argument('--force_cc', action='store_true', help='force to use connected components and ignore glb groups')
parser.add_argument('--workspace', type=str, default='output', help='path to the output folder')
opt = parser.parse_args()


class NamedDisjointSet:
    def __init__(self, names):
        # names: list of str
        self.parent = {name: name for name in names}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def merge(self, x, y):
        self.parent[self.find(x)] = self.find(y)
    
    def get_groups(self):
        groups = {}
        for name in self.parent:
            root = self.find(name)
            if root not in groups:
                groups[root] = []
            groups[root].append(name)
        return groups


def is_single_layer_plane(mesh: trimesh.Trimesh, coplane_thresh: float = 1):
    # check if a mesh is just a single-layer plane
    if mesh.is_watertight: return False
    face_normals = mesh.face_normals
    diff = np.linalg.norm(np.abs(face_normals) - np.abs(face_normals[0]))
    return diff.max() < coplane_thresh

def calc_intersection_union(bounds1, bounds2):
    # bounds: [2, 3]
    bmin1, bmax1 = bounds1[0], bounds1[1]
    bmin2, bmax2 = bounds2[0], bounds2[1]
    # intersection
    bmin = np.maximum(bmin1, bmin2)
    bmax = np.minimum(bmax1, bmax2)
    if np.any(bmin >= bmax):
        intersection = 0
    else:
        intersection = np.prod(bmax - bmin)
    # union
    vol1 = np.prod(bmax1 - bmin1)
    vol2 = np.prod(bmax2 - bmin2)
    union = vol1 + vol2 - intersection
    return intersection, union

def is_coplanar_and_convex(vertices, coplanar_thresh: float = 1):
    # vertices: [N, 3], assume ordered to form a coplanar polygon
    # note the last vertex is the same as the first vertex, i.e. ABCA
    # we are actually not requiring perfect coplanarity, the thresh of 1 means 60 degree tolerance...
    
    # Need at least 3 vertices to form a polygon
    if len(vertices) < 3:
        return False
    
    # Convert to numpy array if not already
    points = np.array(vertices)
    n_points = len(points)

    # Normal of the first triangle
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    # Test if coplanar using the normal of fan-cut triangles
    for i in range(2, n_points - 2):
        v1 = points[i] - points[0]
        v2 = points[i+1] - points[0]
        normal_cur = np.cross(v1, v2)
        normal_cur = normal_cur / (np.linalg.norm(normal_cur) + 1e-12)
        diff = np.linalg.norm(np.abs(normal_cur) - np.abs(normal))
        if diff > coplanar_thresh:
            # print(f'not coplanar: {normal_cur} != {normal} (diff = {diff:.4f}) at {i}-{i+1}')
            return False # not coplanar
    
    # Find basis vectors for the 2D plane
    # First basis vector can be the normalized vector from points[0] to points[1]
    basis1 = v1 / (np.linalg.norm(v1) + 1e-12)
    # Second basis vector is perpendicular to both normal and basis1
    basis2 = np.cross(normal, basis1)
    basis2 = basis2 / (np.linalg.norm(basis2) + 1e-12)
    
    # Project all points onto the 2D plane
    points_2d = np.zeros((n_points, 2))
    for i in range(n_points):
        v = points[i] - points[0]
        points_2d[i, 0] = np.dot(v, basis1)
        points_2d[i, 1] = np.dot(v, basis2)
    
    # Check if polygon is convex by using the cross product
    # For a convex polygon, all cross products should have the same sign
    sign = 0
    for i in range(n_points - 1):
        j = (i + 1) % n_points
        k = (i + 2) % n_points
        
        # Vectors from point i to j and j to k
        v1 = points_2d[j] - points_2d[i]
        v2 = points_2d[k] - points_2d[j]
        
        # 2D cross product
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        
        # Check for consistent sign of cross product
        if abs(cross_product) > 1e-2:  # Skip collinear points
            current_sign = np.sign(cross_product)
            if sign == 0:
                sign = current_sign
            elif sign != current_sign:
                # print(f'not convex: {i}-{j} and {j}-{k}, cross_product = {cross_product:.4f}')
                return False  # not convex
    
    return True


def stitch_nonwatertight_mesh(mesh: trimesh.Trimesh, eps: float = 1e-2):
    # mesh will be inplace modified
    # return a flag denoting if there are still open boundaries unfixed

    # manager = trimesh.collision.CollisionManager()
    # manager.add_object('main', mesh)

    # watertight mesh doesn't need to be stitched
    if mesh.is_watertight:
        return False

    # planar mesh cannot be stitched
    if is_single_layer_plane(mesh, eps):
        return True

    # the following is modified from trimesh.repair.stitch
    # fan_faces = trimesh.repair.stitch(mesh)

    nonwatertight = False

    from trimesh.path.exchange.misc import faces_to_path
    
    faces = np.arange(len(mesh.faces))

    # get a sequence of vertex indices representing the
    # boundary of the specified faces
    # will be referencing the same indexes of `mesh.vertices`
    boundaries = [
        e.points
        for e in faces_to_path(mesh, faces)["entities"]
        if len(e.points) > 3 and e.points[0] == e.points[-1]
    ]

    # get properties to avoid querying in loop
    vertices = mesh.vertices
    normals = mesh.face_normals

    # find which faces are associated with an edge
    edges_face = mesh.edges_face
    tree_edge = mesh.edges_sorted_tree

    # MODIFIED: if any two boundary edges share close vertices, we discard both since they may connect
    mask = np.ones(len(boundaries), dtype=bool)
    for i in range(len(boundaries)):
        for j in range(i + 1, len(boundaries)):
            verts_i = vertices[boundaries[i]]  # [N, 3]
            verts_j = vertices[boundaries[j]]  # [M, 3]
            # check pair-wise distance
            dists = np.linalg.norm(verts_i[:, None, :] - verts_j[None, :, :], axis=-1)  # [N, M]
            num_close = np.sum(dists < 1e-6)
            if num_close >= 4 or (num_close / verts_i.shape[0] >= 0.5) or (num_close / verts_j.shape[0] >= 0.5):
                mask[i] = False
                mask[j] = False
                # print(f'discarding boundary {i} and {j} because of close vertices')
    boundaries = [boundaries[i] for i in range(len(boundaries)) if mask[i]]

    # MODIFIED: we only keep coplanar & convex fans
    fans = []
    for vert_indices in boundaries:

        # the fan should be coplanar and convex
        verts = vertices[vert_indices]
        if not is_coplanar_and_convex(verts):
            nonwatertight = True
            continue

        fan = np.column_stack((np.ones(len(vert_indices) - 3, dtype=int) * vert_indices[0], vert_indices[1:-2], vert_indices[2:-1])) # [N, 3]

        # the fan should not collide with any other faces in the mesh [wrong, fan always collide (contact) at the connected vertices...]
        # vert_indices_isolated = np.arange(len(verts))
        # fan_isolated = np.column_stack((np.zeros(len(vert_indices_isolated) - 3, dtype=int), vert_indices_isolated[1:-2], vert_indices_isolated[2:-1])) # [N, 3]
        # mesh_isolated = trimesh.Trimesh(verts, fan_isolated)
        # is_collide = manager.in_collision_single(mesh_isolated)
        # if is_collide:
        #     print(f'fan {fan} collides with other faces')
        #     nonwatertight = True
        #     continue

        fans.append(fan)

    # now we do a normal check against an adjacent face
    # to see if each region needs to be flipped
    for i, t in zip(range(len(fans)), fans):
        # get the edges from the original mesh
        # for the first `n` new triangles
        e = t[:10, 1:].copy()
        e.sort(axis=1)

        # find which indexes of `mesh.edges` these
        # new edges correspond with by finding edges
        # that exactly correspond with the tree
        query = tree_edge.query_ball_point(e, r=1e-10)
        if len(query) == 0:
            continue
        # stack all the indices that exist
        edge_index = np.concatenate(query)

        # get the normals from the original mesh
        original = normals[edges_face[edge_index]]

        # calculate the normals for a few new faces
        check, valid = trimesh.triangles.normals(vertices[t[:3]])
        if not valid.any():
            continue
        # take the first valid normal from our new faces
        check = check[0]

        # if our new faces are reversed from the original
        # Adjacent face flip them along their axis
        sign = np.dot(original, check)
        if sign.mean() < 0:
            fans[i] = np.fliplr(t)

    if len(fans) > 0:
        fans = np.vstack(fans)
        mesh.faces = np.concatenate([mesh.faces, fans])

    return nonwatertight


def smart_grouping(meshes: dict):
    # meshes: {name: trimesh.Trimesh, ...}

    # use collision manager to find all colliding pairs
    manager = trimesh.collision.CollisionManager()
    for name, mesh in meshes.items():
        manager.add_object(name, mesh)

    is_collide, collide_pairs = manager.in_collision_internal(return_names=True)
    # print(f'[INFO] num_collide = {len(collide_pairs)}, {collide_pairs}')

    if not is_collide:
        return meshes

    # pre-calculate some stat for each mesh
    name_to_stat = {}
    total_volume = 0
    max_extent = 0
    num_submeshes = 0
    num_meshes = len(meshes)
    for name, mesh in meshes.items():
        name_to_stat[name] = {}
        submeshes = mesh.split()
        name_to_stat[name]['volume'] = []
        name_to_stat[name]['extent'] = []
        num_submeshes += len(submeshes)
        for submesh in submeshes:
            if submesh.is_watertight: 
                name_to_stat[name]['volume'].append(submesh.volume)
                total_volume += name_to_stat[name]['volume'][-1]
            name_to_stat[name]['extent'].append(np.max(submesh.extents))
            max_extent = max(max_extent, name_to_stat[name]['extent'][-1])
        name_to_stat[name]['volume'] = np.mean(name_to_stat[name]['volume']) if len(name_to_stat[name]['volume']) > 0 else np.inf
        name_to_stat[name]['extent'] = np.max(name_to_stat[name]['extent']) if len(name_to_stat[name]['extent']) > 0 else np.inf
    
    # use a disjoint set to record grouping
    ds = NamedDisjointSet(list(meshes.keys()))

    # decide the merging thresh adaptively (based on the number of meshes, average volume and extent)
    # very empirical...
    if num_meshes <= 16:
        tol_volume = 0.05 * total_volume
        tol_extent = 0.05 * max_extent
    else:
        tol_volume = 0.1 * total_volume
        tol_extent = 0.1 * max_extent

    # loop each pair, determine if they should be grouped together
    for name1, name2 in collide_pairs:

        if ds.find(name1) == ds.find(name2):
            continue

        # single-layer plane should be merged
        if is_single_layer_plane(meshes[name1]) or is_single_layer_plane(meshes[name2]):
            # print(f'[INFO] merge {name1} and {name2} because of single-layer plane')
            ds.merge(name1, name2)
            continue

        # too small component
        if (name_to_stat[name1]['volume'] < tol_volume and name_to_stat[name1]['extent'] < tol_extent) or \
           (name_to_stat[name2]['volume'] < tol_volume and name_to_stat[name2]['extent'] < tol_extent):
            # print(f'[INFO] merge {name1} and {name2} because of small volume')
            ds.merge(name1, name2)
            continue

        # overlaps a lot should be merged (just use bounding box IoU)
        bounds1 = meshes[name1].bounds # [2, 3]
        bounds2 = meshes[name2].bounds # [2, 3]
        vol_intersect, vol_union = calc_intersection_union(bounds1, bounds2)
        vol_iou = vol_intersect / vol_union
        if vol_iou > 0.5:
            # print(f'[INFO] merge {name1} and {name2} because of large IoU')
            ds.merge(name1, name2)
            continue

        # still nonwatertight meshes should be merged [this is too aggressive, disabled]
        # if name_to_nonwatertight[name1] or name_to_nonwatertight[name2]:
        #     ds.merge(name1, name2)
    
    # merge groups
    groups = ds.get_groups()
    for group in groups.values():
        if len(group) <= 1:
            continue
        # print(f'[INFO] merge group: {group}')
        new_name = '_'.join(group)
        new_mesh = trimesh.util.concatenate(list(meshes[name] for name in group))
        meshes[new_name] = new_mesh
        for name in group:
            del meshes[name]
    
    print(f'[INFO] after grouping, num_meshes = {len(meshes)}')
    return meshes


def run(path):

    mesh = trimesh.load(path)

    if not opt.force_cc and isinstance(mesh, trimesh.Scene) and len(mesh.geometry) > 1:
        print(f'[INFO] scene: {len(mesh.geometry)} meshes')
        scene = mesh
                
    else:
        if isinstance(mesh, trimesh.Scene): 
            mesh = mesh.to_mesh()
        # use meshiki backend
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
    
    ### stitch open boundaries to make each mesh watertight
    for name, mesh in meshes.items():
        stitch_nonwatertight_mesh(mesh)

    ### smart grouping to avoid too many single-layer surface or too small objects
    meshes = smart_grouping(meshes)

    ### coloring
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

    # build an undirected collision graph
    manager = trimesh.collision.CollisionManager()
    for name, mesh in meshes.items():
        manager.add_object(name, mesh)
    
    is_collide, collide_pairs = manager.in_collision_internal(return_names=True)

    graph = {name: [] for name in meshes.keys()}
    for name1, name2 in collide_pairs:
        graph[name1].append(name2)
        graph[name2].append(name1)
    
    # we will start graph coloring from center to border
    name_to_color = {}
    queue = []
    for name, dist in name_with_dist:
        if name not in name_to_color:
            name_to_color[name] = 0
            queue.append(name)
            while len(queue) > 0:
                name = queue.pop(0)
                for neighbor in graph[name]:
                    if neighbor not in name_to_color:
                        name_to_color[neighbor] = 1 - name_to_color[name]
                        queue.append(neighbor)
                    else:
                        if name_to_color[neighbor] == name_to_color[name]:
                            print(f'[WARN] {name} and {neighbor} have the same color, this graph cannot be colored with only two colors')

    # get the two parts
    mesh_color0 = []
    mesh_color1 = []
    for name, color in name_to_color.items():
        if color == 0:
            mesh_color0.append(meshes[name])
        else:
            mesh_color1.append(meshes[name])
    
    ### convert to a single mesh and export as glb
    mesh_color0 = trimesh.util.concatenate(mesh_color0)
    mesh_color1 = trimesh.util.concatenate(mesh_color1)
    name = os.path.splitext(os.path.basename(path))[0]

    # export separately
    mesh_color0.export(f'{opt.workspace}/{name}_color0.obj')
    mesh_color1.export(f'{opt.workspace}/{name}_color1.obj')

    # export together (offsetted)
    mesh_color1.vertices += [0, 0, 1]
    mesh_all = trimesh.util.concatenate([mesh_color0, mesh_color1])
    mesh_all.export(f'{opt.workspace}/{name}.obj')

os.makedirs(opt.workspace, exist_ok=True)

if os.path.isdir(opt.test_path):
    file_paths = glob.glob(os.path.join(opt.test_path, "*"))
    for path in tqdm.tqdm(file_paths):
        run(path)
else:
    run(opt.test_path)