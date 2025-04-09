#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <algorithm>
#include <cmath>

#include <mwm.h>
#include <pcg32.h>
#include <bucket_fps_api.h>

#include <meshiki/utils.h>
#include <meshiki/elements.h>
#include <meshiki/collision.h>

using namespace std;

class Mesh {
public:

    // mesh data
    vector<Vertex*> verts;
    vector<Facet*> faces;

    bool verbose = false;
    
    // Euler characteristic: V - E + F = 2 - 2g - b
    int num_verts = 0;
    int num_edges = 0;
    int num_faces = 0;
    int num_components = 0;

    // indicator for quad quality (quad-only)
    float rect_error = 0;
    int num_quad = 0;

    // bvh
    BVHNode* bvh = NULL;

    // total surface area of all faces
    float total_area = 0;

    // if edge-manifold
    bool is_edge_manifold = true;

    // if watertight
    bool is_watertight = true; // the whole mesh
    map<int, bool> component_is_watertight; // per connected component

    // connected components
    map<int, vector<Facet*>> component_faces;
    map<int, BVHNode*> component_bvhs;

    // boundaries
    map<int, vector<BoundaryLoop>> component_boundaries;

    // faces_input could contain mixed trig and quad. (quad assumes 4 vertices are in order)
    Mesh(vector<vector<float>> verts_input, vector<vector<int>> faces_input, bool clean = false, bool verbose = false) {

        this->verbose = verbose;

        // clean vertices
        if (clean) {
            float thresh = 1e-8; // TODO: don't hard-code
            auto [verts_clean, faces_clean] = merge_close_vertices(verts_input, faces_input, thresh, verbose);
            verts_input = move(verts_clean);
            faces_input = move(faces_clean);
        }

        // verts (assume in [-1, 1], we won't do error handling in cpp!)
        for (size_t i = 0; i < verts_input.size(); i++) {
            Vertex* v = new Vertex(verts_input[i][0], verts_input[i][1], verts_input[i][2], i);
            verts.push_back(v);
        }
        num_verts = verts.size();
        // build face and edge
        map<pair<int, int>, HalfEdge*> edge2halfedge; // to hold twin half edge
        for (size_t i = 0; i < faces_input.size(); i++) {
            vector<int>& f_in = faces_input[i];
            Facet* f = new Facet();
            f->i = i;
            // build half edge and link to verts
            float cur_quad_angle = 0;
            int num_edges = f_in.size();
            for (int j = 0; j < num_edges; j++) {
                HalfEdge* e = new HalfEdge();
                e->t = f;
                e->i = j;
                if (num_edges == 3) {
                    // trig
                    e->v = verts[f_in[j]];
                    e->s = verts[f_in[(j + 1) % 3]];
                    e->e = verts[f_in[(j + 2) % 3]];
                    e->angle = angle_between(Vector3f(*e->v, *e->s), Vector3f(*e->v, *e->e));
                } else if (num_edges == 4) {
                    // quad
                    e->v = verts[f_in[j]];
                    e->s = verts[f_in[(j + 1) % 4]];
                    e->e = verts[f_in[(j + 2) % 4]];
                    e->w = verts[f_in[(j + 3) % 4]];
                    e->angle = angle_between(Vector3f(*e->v, *e->s), Vector3f(*e->v, *e->w));
                    // update quad weight
                    cur_quad_angle += abs(90 - e->angle) / 4;
                } else {
                    // polygon
                    e->v = verts[f_in[j]];
                    e->s = verts[f_in[(j + 1) % num_edges]];
                    e->e = verts[f_in[(j + 2) % num_edges]];
                    // no angle defined for polygon
                }
                // update neighbor vertices
                e->s->neighbors.insert(e->e);
                e->e->neighbors.insert(e->s);
                // update face
                f->vertices.push_back(verts[f_in[j]]);
                f->half_edges.push_back(e);
            }
            if (num_edges == 4) {
                // update quad stat
                rect_error = (rect_error * num_quad + cur_quad_angle) / (num_quad + 1);
                num_quad++;
            }
            // link prev and next half_edge
            // assume each face's vertex ordering is counter-clockwise, so next = right, prev = left
            for (int j = 0; j < int(f->half_edges.size()); j++) {
                f->half_edges[j]->n = f->half_edges[(j + 1) % f->half_edges.size()];
                f->half_edges[j]->p = f->half_edges[(j - 1 + f->half_edges.size()) % f->half_edges.size()];
                if (num_edges == 4) {
                    // x is specially defined for quad
                    f->half_edges[j]->x = f->half_edges[(j + 2) % f->half_edges.size()];
                }
            }
            // link opposite half_edge
            for (int j = 0; j < int(f->half_edges.size()); j++) {
                HalfEdge* e = f->half_edges[j];
                // link opposite half edge
                pair<int, int> key = edge_key(f_in[(j + 1) % num_edges], f_in[(j + 2) % num_edges]);
                if (edge2halfedge.find(key) == edge2halfedge.end()) {
                    edge2halfedge[key] = e;
                } else {
                    // if this key has already matched two half_edges, this mesh is not edge-manifold (an edge is shared by three or more faces)!
                    if (edge2halfedge[key] == NULL) {
                        is_edge_manifold = false;
                        // we can do nothing to fix it... treat it as a border edge
                        continue;
                    }
                    // twin half edge
                    e->o = edge2halfedge[key];
                    edge2halfedge[key]->o = e;
                    // using NULL to mark as already matched
                    edge2halfedge[key] = NULL;
                }
            }
            // compute face center and area
            f->update();
            total_area += f->area;
            faces.push_back(f);
        }

        num_faces = faces.size();
        num_edges = edge2halfedge.size();

        // find connected components and fix face orientation
        for (size_t i = 0; i < faces_input.size(); i++) {
            Facet* f = faces[i];
            if (f->ic == -1) {
                component_is_watertight[num_components] = true;
                component_faces[num_components] = vector<Facet*>();
                // if (verbose) cout << "[MESH] find connected component " << num_components << endl;
                // recursively mark all connected faces
                queue<Facet*> q;
                q.push(f);
                while (!q.empty()) {
                    Facet* f = q.front();
                    q.pop();
                    if (f->ic != -1) continue;
                    f->ic = num_components;
                    for (size_t j = 0; j < f->half_edges.size(); j++) {
                        HalfEdge* e = f->half_edges[j];
                        if (e->o != NULL) {
                            if (e->o->t->ic == -1) {
                                // push to queue
                                q.push(e->o->t);
                                // always fix the face orientation (makes it align with the first face)
                                if (e->s->i != e->o->e->i || e->e->i != e->o->s->i) {
                                    e->o->t->flip();
                                }
                            }
                        } else {
                            component_is_watertight[num_components] = false;
                            is_watertight = false;
                            // find the boundary that contains this edge if it's not visited
                            if (e->m == 0) {
                                BoundaryLoop loop;
                                loop.build(e);
                                // mark all edges in this loop
                                for (size_t j = 0; j < loop.edges.size(); j++) {
                                    loop.edges[j]->m = 1;
                                }
                                component_boundaries[num_components].push_back(loop);
                            }
                        }
                    }
                    component_faces[num_components].push_back(f);
                }
                num_components++;
            }
        }

        if (verbose) {
            cout << "[MESH] Vertices = " << num_verts << ", Edges = " << num_edges << ", Faces = " << num_faces << ", Components = " << num_components << ", Watertight = " << (is_watertight ? "true" : "false") << ", Edge-manifold = " << (is_edge_manifold ? "true" : "false") << endl;
            for (int i = 0; i < num_components; i++) {
                cout << "[MESH] Component " << i << " Faces = " << component_faces[i].size() << ", Watertight = " << (component_is_watertight[i] ? "true" : "false") << ", Boundaries = " << component_boundaries[i].size() << endl;
            }
        }

        // sort faces using connected component and center
        sort(faces.begin(), faces.end(), [](const Facet* f1, const Facet* f2) { return *f1 < *f2; });

        // reset face index
        for (size_t i = 0; i < faces.size(); i++) { faces[i]->i = i; }

        // build bvh for the whole mesh and each component
        bvh = build_bvh(faces);
        for (int i = 0; i < num_components; i++) {
            component_bvhs[i] = build_bvh(component_faces[i]);
        }
    }

    void smart_group_components() {
        // assume NOT merge_close_vertices when loading (clean = false) !!!
        
        // loop each pair of components and use boundary edges to determine if two components are connected
        DisjointSet ds(num_components);
        for (int i = 0; i < num_components; i++) {
            for (int j = i + 1; j < num_components; j++) {
                // merge by connecting boundarys
                bool merged = false;
                for (size_t k = 0; k < component_boundaries[i].size(); k++) {
                    for (size_t l = 0; l < component_boundaries[j].size(); l++) {
                        if (boundary_may_connect(component_boundaries[i][k], component_boundaries[j][l])) {
                            component_boundaries[i][k].num_connected++;
                            component_boundaries[j][l].num_connected++;
                            ds.merge(j, i); // merge j to i (so root always has the smallest index)
                            merged = true;
                            if (verbose) cout << "[MESH] merge component " << j << " to " << i << " due to boundary connection" << endl;
                            break;
                        }
                    }
                    if (merged) break;
                }
                // merge too-small components.
                // if (!merged && intersect_bvh(component_bvhs[i], component_bvhs[j], true)) {
                //     // empirical thresholding using bbox extent
                //     float total = bvh->bbox.size().max_component() + 1e-6;
                //     float ri = component_bvhs[i]->bbox.size().max_component() + 1e-6;
                //     float rj = component_bvhs[j]->bbox.size().max_component() + 1e-6;
                //     float ratio_global = min(ri, rj) / total;  
                //     if (ratio_global < 0.05) {
                //         if (verbose) cout << "[MESH] merge component " << j << " to " << i << " due to small ratio: " << ratio_global << endl;
                //         ds.merge(j, i); // merge j to i (so root always has the smallest index)
                //         merged = true;
                //     }
                // }
            }
        }

        // check still-open (num_connected == 0) boundaries
        // TODO: try to close them by adding a lid, or merge to intersecting component
        for (int i = 0; i < num_components; i++) {
            for (size_t j = 0; j < component_boundaries[i].size(); j++) {
                if (component_boundaries[i][j].num_connected == 0) {
                    if (verbose) cout << "[MESH] found still open boundary: component " << i << " boundary " << j << endl;
                }
            }
        }

        // merge components from back to front
        for (int i = num_components - 1; i >= 0; i--) {
            int root = ds.find(i);
            if (root != i) {
                // merge component i to root
                // if (verbose) cout << "[MESH] smart merge component " << i << " to " << root << endl;
                component_is_watertight[root] = component_is_watertight[root] && component_is_watertight[i];
                component_boundaries[root].insert(component_boundaries[root].end(), component_boundaries[i].begin(), component_boundaries[i].end());
                component_faces[root].insert(component_faces[root].end(), component_faces[i].begin(), component_faces[i].end());
                for (size_t j = 0; j < component_faces[i].size(); j++) {
                    component_faces[i][j]->ic = root;
                }
                component_is_watertight.erase(i);
                component_boundaries.erase(i);
                component_faces.erase(i);
            }
        }
        // reindex component
        vector<int> roots;
        for (auto& [root, tmp] : component_is_watertight) {
            roots.push_back(root);
        }
        num_components = roots.size();
        map<int, bool> new_component_is_watertight;
        map<int, vector<Facet*>> new_component_faces;
        map<int, vector<BoundaryLoop>> new_component_boundaries;
        for (int i = 0; i < num_components; i++) {
            if (verbose) cout << "[MESH] reindex component " << roots[i] << " to " << i << endl;
            new_component_is_watertight[i] = component_is_watertight[roots[i]];
            new_component_faces[i] = move(component_faces[roots[i]]);
            new_component_boundaries[i] = move(component_boundaries[roots[i]]);
        }
        component_is_watertight = move(new_component_is_watertight);
        component_faces = move(new_component_faces);
        component_boundaries = move(new_component_boundaries);
        for (int i = 0; i < num_components; i++) {
            // reindex faces
            for (size_t j = 0; j < component_faces[i].size(); j++) {
                component_faces[i][j]->ic = i;
            }
        }
        // rebuild bvh
        for (int i = 0; i < num_components; i++) {
            delete component_bvhs[i];
            component_bvhs[i] = build_bvh(component_faces[i]);
        }
        if (verbose) {
            cout << "[MESH] After smart merge:" << endl;
            for (int i = 0; i < num_components; i++) {
                cout << "[MESH] Component " << i << " Faces = " << component_faces[i].size() << ", Watertight = " << (component_is_watertight[i] ? "true" : "false") << ", Boundaries = " << component_boundaries[i].size() << endl;
            }
        }
    }

    // explode to separate connected components
    void explode(float delta) {
        // smart merge components first
        // smart_group_components();

        // decide which component is the center 
        int center_cid = 0;
        Vector3f center = bvh->bbox.center();
        float min_dist = (component_bvhs[0]->bbox.center() - center).norm();
        for (int i = 1; i < num_components; i++) {
            float dist = (component_bvhs[i]->bbox.center() - center).norm();
            if (dist < min_dist) {
                min_dist = dist;
                center_cid = i;
            }
        }

        // sort components by distance to the center object
        center = component_bvhs[center_cid]->bbox.center();
        vector<pair<int, float>> component_sorter;
        for (int i = 0; i < num_components; i++) {
            component_sorter.push_back({i, (component_bvhs[i]->bbox.center() - center).norm()});
        }
        sort(component_sorter.begin(), component_sorter.end(), [](const pair<int, float>& a, const pair<int, float>& b) {
            return a.second < b.second;
        });
        vector<int> component_order;
        for (auto& [cid, extent] : component_sorter) {
            component_order.push_back(cid);
        }

        // loop components from big to small (skip the first one)
        for (int i = 1; i < component_order.size(); i++) {
            int cid = component_order[i];

            // get the component vertices (unique)
            set<Vertex*> component_verts;
            for (Facet* f : component_faces[cid]) {
                for (Vertex* v : f->vertices) {
                    component_verts.insert(v);
                }
            }
            // get the pushing direction
            Vector3f component_center = component_bvhs[cid]->bbox.center();
            Vector3f direction = (component_center - center);
            float dist = direction.norm();
            if (dist < 1e-6) continue;
            else direction = direction / dist;

            // decide the scale to avoid overlap with other components (always move along the direction at an interval)
            for (int j = 0; j < i; j++) {
                int cid_other = component_order[j];
                while (intersect_bvh(component_bvhs[cid], component_bvhs[cid_other])) {
                    // push away this component a little bit
                    component_bvhs[cid]->translate(direction * delta);
                    for (Vertex* v : component_verts) {
                        v->x += direction.x * delta;
                        v->y += direction.y * delta;
                        v->z += direction.z * delta;
                    }
                    cout << "DEBUG: push away component " << cid << " from " << cid_other << " with direction " << direction << ", center is " << component_bvhs[cid]->bbox.center() << endl;
                }
                // push once more to have some interval
                component_bvhs[cid]->translate(direction * delta * 0.1);
                for (Vertex* v : component_verts) {
                    v->x += direction.x * delta * 0.1;
                    v->y += direction.y * delta * 0.1;
                    v->z += direction.z * delta * 0.1;
                }
            }
        }
    }

    // empirically repair face orientation
    void repair_face_orientation() {
        // for each component, we choose the furthest face from its center
        for (int i = 0; i < num_components; i++) {
            float max_dist = -1;
            Facet* furthest_face = NULL;
            Vector3f component_center = component_bvhs[i]->bbox.center();
            for (size_t j = 0; j < component_faces[i].size(); j++) {
                Facet* f = component_faces[i][j];
                float dist = (f->center - component_center).norm();
                if (dist > max_dist) {
                    max_dist = dist;
                    furthest_face = f;
                }
            }
            // see if this face's orientation is correct
            Vector3f normal = furthest_face->normal;
            Vector3f dir = (furthest_face->center - component_center).normalize();
            float dot = normal.dot(dir);
            if (dot < 0) {
                if (verbose) cout << "[MESH] flip face for component " << i << endl;
                for (size_t j = 0; j < component_faces[i].size(); j++) {
                    component_faces[i][j]->flip();
                }
            }
        }
    }

    // trig-to-quad conversion (in-place)
    void quadrangulate(float thresh_bihedral, float thresh_convex) {

        vector<Facet*> faces_old = faces;
        int num_faces_ori = faces.size();
        num_quad = 0;
        rect_error = 0;

        // reset face mask
        for (size_t i = 0; i < faces.size(); i++) { faces[i]->m = 0; }

        auto merge_func = [&](HalfEdge* e) -> Facet* {
            // we will merge e->t and e->o->t into a quad
            // if (verbose) cout << "[MESH] quadrangulate " << e->t->i << " and " << e->o->t->i << endl;
            Facet* q = new Facet();
            HalfEdge* e1 = new HalfEdge();
            HalfEdge* e2 = new HalfEdge();
            HalfEdge* e3 = new HalfEdge();
            HalfEdge* e4 = new HalfEdge();

            // default triangulation ("fixed" mode in blender) will always connect between the first and the third third vertex.
            // ref: https://docs.blender.org/manual/en/latest/modeling/modifiers/generate/triangulate.html
            q->vertices.push_back(e->s);
            q->vertices.push_back(e->o->v);
            q->vertices.push_back(e->e);
            q->vertices.push_back(e->v);
            q->half_edges.push_back(e2);
            q->half_edges.push_back(e3);
            q->half_edges.push_back(e4);
            q->half_edges.push_back(e1);
            q->center = Vector3f(*e->v + *e->s + *e->o->v + *e->e) / 4.0;
            q->ic = e->t->ic;

            // build half_edges
            e1->v = e->v; e1->s = e->s; e1->e = e->o->v; e1->w = e->e; 
            e2->v = e->s; e2->s = e->o->v; e2->e = e->e; e2->w = e->v;
            e3->v = e->o->v; e3->s = e->e; e3->e = e->v; e3->w = e->s;
            e4->v = e->e; e4->s = e->v; e4->e = e->s; e4->w = e->o->v;
            e1->angle = angle_between(Vector3f(*e1->v, *e1->s), Vector3f(*e1->v, *e1->w));
            e2->angle = angle_between(Vector3f(*e2->v, *e2->s), Vector3f(*e2->v, *e2->w));
            e3->angle = angle_between(Vector3f(*e3->v, *e3->s), Vector3f(*e3->v, *e3->w));
            e4->angle = angle_between(Vector3f(*e4->v, *e4->s), Vector3f(*e4->v, *e4->w));
            e1->t = q; e2->t = q; e3->t = q; e4->t = q;
            e1->i = 0; e2->i = 1; e3->i = 2; e4->i = 3;
            e1->n = e2; e2->n = e3; e3->n = e4; e4->n = e1;
            e1->p = e4; e2->p = e1; e3->p = e2; e4->p = e3;
            e1->x = e3; e2->x = e4; e3->x = e1; e4->x = e2;
            // opposite half_edge (mutually)
            e1->o = e->o->n->o; if (e1->o != NULL) e1->o->o = e1;
            e2->o = e->o->p->o; if (e2->o != NULL) e2->o->o = e2;
            e3->o = e->n->o; if (e3->o != NULL) e3->o->o = e3;
            e4->o = e->p->o; if (e4->o != NULL) e4->o->o = e4;
            // we don't delete isolated faces here, but mark them
            e->t->m = 2;
            e->o->t->m = 2;

            // update vertex    
            e->s->neighbors.erase(e->e);
            e->e->neighbors.erase(e->s);

            // update quad stat
            float cur_quad_angle = (abs(90 - e1->angle) + abs(90 - e2->angle) + abs(90 - e3->angle) + abs(90 - e4->angle)) / 4;
            rect_error = (rect_error * num_quad + cur_quad_angle) / (num_quad + 1);
            num_quad++;

            // delete old isolated halfedges
            delete e->o->n; delete e->o->p; delete e->o;
            delete e->n; delete e->p; delete e;

            // push new faces
            faces_old.push_back(q);
            return q;
        };

        using MWM = MaximumWeightedMatching<int>;
        using MWMEdge = MWM::InputEdge;

        vector<int> ou(num_faces_ori + 2), ov(num_faces_ori + 2);
        vector<MWMEdge> mwm_edges;

        // build graph
        map<pair<int, int>, HalfEdge*> M;
        for (int i = 0; i < num_faces_ori; i++) {
            Facet* f = faces[i];
            if (f->m) continue; // already isolated or visited face
            if (f->half_edges.size() > 3) continue; // already quad

            // detect if this face can compose a quad with a neighbor face
            HalfEdge* e;
            for (int j = 0; j < 3; j++) {
                e = f->half_edges[j];
                if (e->o == NULL) continue; // boundary edge
                if (e->t->i == e->o->t->i) continue; // duplicate faces (rare...)
                if (e->o->t->half_edges.size() > 3) continue; // quad opposite face
                if (angle_between(e->t->normal, e->o->t->normal) > thresh_bihedral) continue; // quad should be (almost) planar
                if (e->n->angle + e->o->p->angle >= thresh_convex || e->p->angle + e->o->n->angle >= thresh_convex) continue; // quad should be convex

                // edge weight, larger values are more likely to be matched
                // it's better to form rectangular quads (angles are close to 90 degree)
                // weight should be offseted to make sure it's positive
                int weight = 1000 -(abs(e->angle - 90) + abs(e->o->angle - 90) + abs(e->n->angle + e->o->p->angle - 90) + abs(e->p->angle + e->o->n->angle - 90));
                
                // add edge
                auto key = edge_key(e->t->i + 1, e->o->t->i + 1);
                if (M.find(key) != M.end()) continue; // edge is undirected, only add once
                M[key] = e; // to retreive the halfedge from matching
                mwm_edges.push_back({e->t->i + 1, e->o->t->i + 1, weight}); // MWM is 1-indexed internally
                ou[e->t->i + 2] += 1; ov[e->o->t->i + 2] += 1;
            }
        }
        // build mwm_edges
        int num_edges = mwm_edges.size();
        mwm_edges.resize(num_edges * 2);
        for (int i = 1; i <= num_faces_ori + 1; ++i) ov[i] += ov[i - 1];
        for (int i = 0; i < num_edges; ++i) mwm_edges[num_edges + (ov[mwm_edges[i].to]++)] = mwm_edges[i];
        for (int i = 1; i <= num_faces_ori + 1; ++i) ou[i] += ou[i - 1];
        for (int i = 0; i < num_edges; ++i) mwm_edges[ou[mwm_edges[i + num_edges].from]++] = mwm_edges[i + num_edges];
        mwm_edges.resize(num_edges);

        // call matching
        auto ans = MWM(num_faces_ori, mwm_edges).maximum_weighted_matching();
        vector<int> match = ans.second;

        if (verbose) cout << "[MESH] quadrangulate matching total weight: " << ans.first << endl;

        // merge those matchings
        for (int i = 1; i <= num_faces_ori; i++) { // match is also 1-indexed
            if (match[i] == 0) continue; // 0 means unmatched
            // if (verbose) cout << "[MESH] merge face: " << i << " and " << match[i] << endl;
            HalfEdge* e = M[edge_key(i, match[i])];
            merge_func(e);
            match[match[i]] = 0; // mark opposite to avoid merge twice
        }

        // delete isolated faces and compact vector
        faces.clear();
        for (size_t i = 0; i < faces_old.size(); i++) { // we appended new faces to the end
            // delete isolated faces
            if (faces_old[i]->m == 2) {
                delete faces_old[i];
                continue; 
            }
            faces.push_back(faces_old[i]);
        }

        sort(faces.begin(), faces.end(), [](const Facet* f1, const Facet* f2) { return *f1 < *f2; });

        // reset face index
        for (size_t i = 0; i < faces.size(); i++) { faces[i]->i = i; }

        if (verbose) cout << "[MESH] quadrangulate from " << num_faces_ori << " to " << faces.size() << " faces (with " << num_quad << " quads, rect angle error = " << rect_error << ")." << endl;

        num_faces = faces.size();
    }

    // merge as many as possible faces to convex polygons (in-place)
    void polygonize(float thresh_bihedral, float thresh_convex, int max_round) {
        int round = 0;
        
        // reset deleted mask
        for (size_t i = 0; i < verts.size(); i++) { verts[i]->m = 0; }
        for (size_t i = 0; i < faces.size(); i++) { faces[i]->m = 0; }

        while (polygonize_once(thresh_bihedral, thresh_convex)) {
            round++;
            if (verbose) cout << "[MESH] polygonize round " << round << " done." << endl;
            if (round >= max_round) {
                if (verbose) cout << "[MESH] polygonize: reached max round, stop." << endl;
                break;
            }
        }

        // delete isolated verts/faces and compact vector
        vector<Vertex*> verts_new;
        vector<Facet*> faces_new;

        for (size_t i = 0; i < verts.size(); i++) {
            if (verts[i]->m == 2) {
                delete verts[i];
                continue;
            }
            verts_new.push_back(verts[i]);
        }

        for (size_t i = 0; i < faces.size(); i++) {
            // delete isolated faces
            if (faces[i]->m == 2) {
                delete faces[i];
                continue; 
            }
            faces_new.push_back(faces[i]);
        }

        verts = verts_new;
        faces = faces_new;

        // reset vertex index
        for (size_t i = 0; i < verts.size(); i++) { verts[i]->i = i; }

        // reset face index
        sort(faces.begin(), faces.end(), [](const Facet* f1, const Facet* f2) { return *f1 < *f2; });
        for (size_t i = 0; i < faces.size(); i++) { faces[i]->i = i; }

        if (verbose) cout << "[MESH] polygonize faces " << num_faces << " --> " << faces.size() << ", verts " << num_verts << " --> " << verts.size() << "." << endl;

        num_verts = verts.size();
        num_faces = faces.size();
    }

    // merge convex polygon once
    bool polygonize_once(float thresh_bihedral, float thresh_convex) {

        auto merge_func = [&](HalfEdge* e) -> Facet* {
            // we will merge e->t and e->o->t into a new face
            // cout << "DEBUG: polygonize merge " << e->t->i << " and " << e->o->t->i << endl;
            Facet* q = new Facet();
            
            // update index
            q->i = faces.size();
            q->ic = e->t->ic;

            /* find the longest chain that contains e, we need to delete all the intermediate vertices (A, ...)
            \                    /
             \  el    e     er  /
              S -- A ---...--- E
             /                  \
            /                    \
            */
            HalfEdge* el = e;
            while (el->s->neighbors.size() == 2) {
                el->s->m = 2; // mark vert to delete
                el = el->p;
            }
            HalfEdge* er = e;
            while (er->e->neighbors.size() == 2) {
                er->e->m = 2; // mark vert to delete
                er = er->n;
            }
            if (el != e) {
                e->s = el->s;
                e->p = el->p; el->p->n = e;
                e->o->e = el->s;
                e->o->n = el->o->n; el->o->n->p = e->o;
                e->s->neighbors.erase(el->e);
            } else {
                e->s->neighbors.erase(e->e);
            }
            if (er != e) {
                e->e = er->e;
                e->n = er->n; er->n->p = e;
                e->o->s = er->e;
                e->o->p = er->o->p; er->o->p->n = e->o;
                e->e->neighbors.erase(er->s);
            } else {
                e->e->neighbors.erase(e->s);
            }
            
            // eliminate vertices if they are in a chain
            if (e->s->neighbors.size() == 2 &&
                angle_between(Vector3f(*e->s, *e->p->s), Vector3f(*e->s, *e->o->n->e)) >= 175
            ) {
                // cout << "DEBUG: delete vertex " << e->s->i << endl;
                // delete e->s
                e->s->m = 2;
                // only keep one of the two set of halfedges after vertex collapsing
                e->p->e = e->o->n->e;
                e->p->n = e->o->n->n;
                e->o->n->n->p = e->p;
                e->p->s->neighbors.erase(e->s); e->p->s->neighbors.insert(e->o->n->e);
                e->o->n->e->neighbors.erase(e->s); e->o->n->e->neighbors.insert(e->p->s);
                if (e->p->o != NULL) {
                    e->p->o->s = e->o->n->e;
                    e->p->o->p = e->o->n->o->p;
                    e->o->n->o->p->n = e->p->o;
                    // also need to fix face e->p->o->t
                    Facet* f = e->p->o->t;
                    f->vertices.erase(remove(f->vertices.begin(), f->vertices.end(), e->s), f->vertices.end());
                    f->half_edges.erase(remove(f->half_edges.begin(), f->half_edges.end(), e->o->n->o), f->half_edges.end());
                }
                delete e->o->n; 
                if (e->o->n->o != NULL) {
                    delete e->o->n->o;
                }
            } else {
                e->p->n = e->o->n;
                e->o->n->p = e->p;
            }

            if (e->e->neighbors.size() == 2 &&
                angle_between(Vector3f(*e->e, *e->n->e), Vector3f(*e->e, *e->o->p->s)) >= 175
            ) {
                // cout << "DEBUG: delete vertex " << e->e->i << endl;
                // delete e->e
                e->e->m = 2;
                // only keep one of the two set of halfedges after vertex collapsing
                e->n->s = e->o->p->s;
                e->n->p = e->o->p->p;
                e->o->p->p->n = e->n;
                e->n->e->neighbors.erase(e->e); e->n->e->neighbors.insert(e->o->p->s);
                e->o->p->s->neighbors.erase(e->e); e->o->p->s->neighbors.insert(e->n->e);
                if (e->n->o != NULL) {
                    e->n->o->e = e->o->p->s;
                    e->n->o->n = e->o->p->o->n;
                    e->o->p->o->n->p = e->n->o;
                    // also need to fix face e->n->o->t
                    Facet* f = e->n->o->t;
                    f->vertices.erase(remove(f->vertices.begin(), f->vertices.end(), e->e), f->vertices.end());
                    f->half_edges.erase(remove(f->half_edges.begin(), f->half_edges.end(), e->o->p->o), f->half_edges.end());
                }
                delete e->o->p;
                if (e->o->p->o != NULL) {
                    delete e->o->p->o; 
                }
            } else {
                e->n->p = e->o->p;
                e->o->p->n = e->n;
            }

            // append vertices and halfedges to the new face
            // now that we have fixed halfedges, just one loop is enough
            HalfEdge* cur = e->n;
            while (true) {
                q->vertices.push_back(cur->s);
                q->half_edges.push_back(cur);
                cur->t = q; // update face pointer in halfedge
                // cout << "DEBUG: append half edge " << *cur << endl;
                cur = cur->n;
                if (cur == e->n) break;
            }
            
            // we don't update q->angle, as it's undefined in polygon
            q->update();

            // we don't delete isolated faces here, but mark them
            e->t->m = 2;
            e->o->t->m = 2;

            // delete isolated halfedges
            delete e->o; 
            delete e; 

            // push new faces
            faces.push_back(q);
            // cout << "DEBUG: merged into new face " << q->i << endl;
            return q;
        };

        using MWM = MaximumWeightedMatching<int>;
        using MWMEdge = MWM::InputEdge;

        num_faces = faces.size();
        vector<int> ou(num_faces + 2), ov(num_faces + 2);
        vector<MWMEdge> mwm_edges;

        // build graph
        map<pair<int, int>, HalfEdge*> M;
        for (int i = 0; i < num_faces; i++) {
            Facet* f = faces[i];
            if (f->m) continue; // already isolated or visited face

            // detect if this face can compose a convex polygon with a neighbor face
            HalfEdge* e;
            for (size_t j = 0; j < f->half_edges.size(); j++) {

                e = f->half_edges[j];
                if (e->o == NULL) continue; // boundary edge
                if (e->t->i == e->o->t->i) continue; // duplicate faces (rare...)

                // polygon should be (almost) planar
                float coplane_error = angle_between(e->t->normal, e->o->t->normal);
                if (coplane_error > thresh_bihedral) continue; 

                // polygon should be convex (in polygon face, we don't define f->angle, so we have to calculate them here)
                float angle_ep_eon = angle_between(Vector3f(*e->s, *e->p->s), Vector3f(*e->s, *e->e)) + angle_between(Vector3f(*e->s, *e->e), Vector3f(*e->s, *e->o->n->e));
                float angle_eop_en = angle_between(Vector3f(*e->e, *e->n->e), Vector3f(*e->e, *e->s)) + angle_between(Vector3f(*e->e, *e->s), Vector3f(*e->e, *e->o->p->s));
                if (angle_ep_eon >= thresh_convex || angle_eop_en >= thresh_convex) continue;

                // edge weight, larger values are more likely to be matched
                // weight should be offseted to make sure it's positive
                // we perfer planar polygons.
                int weight = 1800 - int(coplane_error * 10);
                
                // add edge
                auto key = edge_key(e->t->i + 1, e->o->t->i + 1);
                if (M.find(key) != M.end()) continue; // edge is undirected, only add once
                M[key] = e; // to retreive the halfedge from matching
                mwm_edges.push_back({e->t->i + 1, e->o->t->i + 1, weight}); // MWM is 1-indexed internally
                ou[e->t->i + 2] += 1; ov[e->o->t->i + 2] += 1;
                if (verbose) cout << "[MESH] polygonize add face " << e->t->i << " and " << e->o->t->i << " with weight " << weight << " " << coplane_error << endl;

            }
        }

        // build mwm_edges
        int num_edges = mwm_edges.size();

        if (num_edges == 0) {
            if (verbose) cout << "[MESH] polygonize: no more merges found, stop." << endl;
            return false;
        }

        mwm_edges.resize(num_edges * 2);
        for (int i = 1; i <= num_faces + 1; ++i) ov[i] += ov[i - 1];
        for (int i = 0; i < num_edges; ++i) mwm_edges[num_edges + (ov[mwm_edges[i].to]++)] = mwm_edges[i];
        for (int i = 1; i <= num_faces + 1; ++i) ou[i] += ou[i - 1];
        for (int i = 0; i < num_edges; ++i) mwm_edges[ou[mwm_edges[i + num_edges].from]++] = mwm_edges[i + num_edges];
        mwm_edges.resize(num_edges);

        // call matching
        auto ans = MWM(num_faces, mwm_edges).maximum_weighted_matching();
        vector<int> match = ans.second;

        if (verbose) cout << "[MESH] polygonize matching total weight: " << ans.first << endl;

        // merge those matchings
        for (int i = 1; i <= num_faces; i++) { // match is also 1-indexed
            if (match[i] == 0) continue; // 0 means unmatched
            HalfEdge* e = M[edge_key(i, match[i])];
            merge_func(e);
            match[match[i]] = 0; // mark opposite to avoid merge twice
        }
        
        return true;
    }

    // salient point sampling
    vector<vector<float>> salient_point_sample(int num_samples, float thresh_angle) {
        vector<vector<float>> samples;
        // loop half_edges and calculate the dihedral angle
        set<pair<int, int>> visited_edges;
        vector<tuple<int, int, float>> salient_edges; // start vert index, end vert index, length
        float total_edge_length = 0;
        for (Facet* f : faces) {
            for (HalfEdge* e : f->half_edges) {
                if (e->o == NULL) continue; // boundary edge
                if (visited_edges.find(edge_key(e->s->i, e->e->i)) != visited_edges.end()) continue;
                visited_edges.insert(edge_key(e->s->i, e->e->i));
                float coplane_error = angle_between(e->t->normal, e->o->t->normal); // 180 - dihedral angle
                if (coplane_error > thresh_angle) {
                    float length = Vector3f(*e->s, *e->e).norm();
                    total_edge_length += length;
                    salient_edges.push_back({e->s->i, e->e->i, length});
                }
            }
        }

        // push the edge vertices
        for (auto& edge : salient_edges) {
            // push the start vertex
            Vertex* v1 = verts[get<0>(edge)];
            samples.push_back({v1->x, v1->y, v1->z});
            // push the end vertex
            Vertex* v2 = verts[get<1>(edge)];
            samples.push_back({v2->x, v2->y, v2->z});
        }

        if (samples.size() == num_samples) {
            return samples;
        } else if (samples.size() > num_samples) {
            // the number of salient edges is enough, just FPS subsample
            if (verbose) cout << "[MESH] salient edges are enough, FPS subsample " << num_samples << " samples." << endl;
            // return fps(samples, num_samples);
            return bucket_fps_kdline(samples, num_samples, 0, 5); // height should choose from 3/5/7
        } else if (samples.size() == 0) {
            // no salient edge, return empty set
            if (verbose) cout << "[MESH] no salient edge found, return empty set." << endl;
            return samples;
        } else {
            // not enough, add more samples along the salient edges
            int num_extra = num_samples - samples.size();
            if (verbose) cout << "[MESH] salient edges are not enough, add " << num_extra << " extra samples along the salient edges." << endl;
            for (size_t i = 0; i < salient_edges.size(); i++) {
                auto& edge = salient_edges[i];
                Vertex* v1 = verts[get<0>(edge)];
                Vertex* v2 = verts[get<1>(edge)];
                Vector3f dir(*v1, *v2);
                float edge_length = get<2>(edge);
                int extra_this_edge = ceil(num_extra * edge_length / total_edge_length); 
                for (int j = 0; j <  extra_this_edge; j++) {
                    float t = (j + 1) / ( extra_this_edge + 1.0);
                    samples.push_back({v1->x + dir.x * t, v1->y + dir.y * t, v1->z + dir.z * t});
                }
            }
            // the above loop may over-sample, so we need to subsample again
            if (samples.size() > num_samples) {
                if (verbose) cout << "[MESH] over-sampled, subsample " << num_samples << " samples." << endl;
                // return bucket_fps_kdline(samples, num_samples, 0, 5); // height should choose from 3/5/7
                vector<int> indices = random_subsample(samples.size(), num_samples);
                vector<vector<float>> samples_out;
                for (int i : indices) {
                    samples_out.push_back(samples[i]);
                }
                return samples_out;
            } else {
                return samples;
            }
        }
    }

    // uniform point sampling (assume the mesh is pure-trig)
    vector<vector<float>> uniform_point_sample(int num_samples) {
        vector<vector<float>> samples;
        pcg32 rng;
        for (size_t i = 0; i < faces.size(); i++) {
            Facet* f = faces[i];
            int samples_this_face = ceil(num_samples * f->area / total_area);
            if (samples_this_face == 0) samples_this_face = 1; // at least one sample per face
            for (int j = 0; j < samples_this_face; j++) {
                // ref: https://mathworld.wolfram.com/TrianglePointPicking.html
                float u = rng.nextFloat();
                float v = rng.nextFloat();
                if (u + v > 1) {
                    u = 1 - u;
                    v = 1 - v;
                }
                Vector3f p = Vector3f(*f->vertices[0]) + Vector3f(*f->vertices[0], *f->vertices[1]) * u + Vector3f(*f->vertices[0], *f->vertices[2]) * v;
                samples.push_back({p.x, p.y, p.z});
            }
        }
        // may over-sample, subsample
        if (samples.size() > num_samples) {
            if (verbose) cout << "[MESH] over-sampled, subsample " << num_samples << " samples." << endl;
            // FPS is too expensive for uniformly sampled points
            // return bucket_fps_kdline(samples, num_samples, 0, 7); // height should choose from 3/5/7
            vector<int> indices = random_subsample(samples.size(), num_samples);
            vector<vector<float>> samples_out;
            for (int i : indices) {
                samples_out.push_back(samples[i]);
            }
            return samples_out;
        } else {
            return samples;
        }
    }

    // export mesh to python (support quad)
    tuple<vector<vector<float>>, vector<vector<int>>> export_mesh() {
        vector<vector<float>> verts_out;
        vector<vector<int>> faces_out;
        for (Vertex* v : verts) {
            verts_out.push_back({v->x, v->y, v->z});
        }
        for (Facet* f : faces) {
            vector<int> face;
            for (Vertex* v : f->vertices) {
                face.push_back(v->i);
            }
            faces_out.push_back(face);
        }
        return make_tuple(verts_out, faces_out);
    }

    vector<tuple<vector<vector<float>>, vector<vector<int>>>> export_components() {
        vector<tuple<vector<vector<float>>, vector<vector<int>>>> components;
        // export each component as a separate mesh
        for (int i = 0; i < num_components; i++) {
            // we need loop twice to map global vertice indices to submesh indices
            map<int, int> vert_map;
            int vert_idx = 0;
            for (int j = 0; j < component_faces[i].size(); j++) {
                Facet* f = component_faces[i][j];
                for (Vertex* v : f->vertices) {
                    if (vert_map.find(v->i) == vert_map.end()) {
                        vert_map[v->i] = vert_idx++;
                    }
                }
            }
            // inverted map
            map<int, int> vert_map_inv;
            for (auto& [idx, sub_idx] : vert_map) {
                vert_map_inv[sub_idx] = idx;
            }
            // build the submesh
            vector<vector<float>> verts_out;
            vector<vector<int>> faces_out;
            for (int j = 0; j < vert_map_inv.size(); j++) {
                verts_out.push_back({verts[vert_map_inv[j]]->x, verts[vert_map_inv[j]]->y, verts[vert_map_inv[j]]->z});
            }
            for (int j = 0; j < component_faces[i].size(); j++) {
                Facet* f = component_faces[i][j];
                vector<int> face;
                for (Vertex* v : f->vertices) {
                    face.push_back(vert_map[v->i]);
                }
                faces_out.push_back(face);
            }
            components.push_back(make_tuple(verts_out, faces_out));
        }
        return components;
    }

    ~Mesh() {
        if (bvh) delete bvh;
        for (auto& [cid, bvh] : component_bvhs) { delete bvh; }
        for (Vertex* v : verts) { delete v; }
        for (Facet* f : faces) {
            for (HalfEdge* e : f->half_edges) { delete e; }
            delete f;
        }
    }
};
