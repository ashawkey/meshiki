#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <algorithm>
#include <cmath>

#include <mwm.h>
#include <meshiki/utils.h>
#include <pcg32.h>
#include <bucket_fps_api.h>

using namespace std;

static float INF = 1e8;

// pre-declare
struct Facet;
struct HalfEdge;

struct Vertex {
    float x, y, z; // float coordinates
    int i = -1; // index
    int m = 0; // visited mark

    // neighbor vertices
    set<Vertex*> neighbors;
    
    Vertex() {}
    Vertex(float x, float y, float z, int i=-1) : x(x), y(y), z(z), i(i) {}

    // operators
    Vertex operator+(const Vertex& v) const {
        return Vertex(x + v.x, y + v.y, z + v.z);
    }
    Vertex operator-(const Vertex& v) const {
        return Vertex(x - v.x, y - v.y, z - v.z);
    }
    bool operator==(const Vertex& v) const {
        return x == v.x && y == v.y && z == v.z;
    }
    bool operator<(const Vertex& v) const {
        // y-z-x order
        return y < v.y || (y == v.y && z < v.z) || (y == v.y && z == v.z && x < v.x);
    }
    friend ostream& operator<<(ostream &os, const Vertex &v) {
        os << "Vertex " << v.i << " :(" << v.x << ", " << v.y << ", " << v.z << ")";
        return os;
    }
};

struct Vector3f {
    float x, y, z; // float coordinates
    Vector3f() {}
    Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}
    Vector3f(const Vertex& v) : x(v.x), y(v.y), z(v.z) {}
    Vector3f(const Vertex& v1, const Vertex& v2) : x(v2.x - v1.x), y(v2.y - v1.y), z(v2.z - v1.z) {} // v1 --> v2
    Vector3f operator+(const Vector3f& v) const {
        return Vector3f(x + v.x, y + v.y, z + v.z);
    }
    Vector3f& operator+=(const Vector3f& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
    Vector3f operator-(const Vector3f& v) const {
        return Vector3f(x - v.x, y - v.y, z - v.z);
    }
    Vector3f& operator-=(const Vector3f& v) {
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }
    Vector3f operator*(float s) const {
        return Vector3f(x * s, y * s, z * s);
    }
    Vector3f& operator*=(float s) {
        x *= s; y *= s; z *= s;
        return *this;
    }
    Vector3f operator/(float s) const {
        return Vector3f(x / s, y / s, z / s);
    }
    Vector3f& operator/=(float s) {
        x /= s; y /= s; z /= s;
        return *this;
    }
    bool operator==(const Vector3f& v) const {
        return x == v.x && y == v.y && z == v.z;
    }
    bool operator<(const Vector3f& v) const {
        // y-z-x order
        return y < v.y || (y == v.y && z < v.z) || (y == v.y && z == v.z && x < v.x);
    }
    Vector3f cross(const Vector3f& v) const {
        return Vector3f(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
    float dot(const Vector3f& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    float norm() const {
        return sqrt(x * x + y * y + z * z);
    }
    Vector3f normalize() const {
        float n = norm() + 1e-8;
        return Vector3f(x / n, y / n, z / n);
    }
    friend ostream& operator<<(ostream &os, const Vector3f &v) {
        os << "Vector3f :(" << v.x << ", " << v.y << ", " << v.z << ")";
        return os;
    }
};

inline float angle_between(Vector3f a, Vector3f b) {
    // Normalize vectors and compute dot product
    float dot = a.dot(b) / (a.norm() * b.norm() + 1e-8);
    // Clamp dot product to [-1, 1] to avoid domain errors with acos
    dot = max(-1.0f, min(1.0f, dot));
    float radian = acos(dot);
    return radian * 180 / M_PI;
}

inline float get_trig_area(const Vertex& v1, const Vertex& v2, const Vertex& v3) {
    Vector3f e1(v1, v2);
    Vector3f e2(v1, v3);
    return 0.5 * (e1.cross(e2)).norm();
}

/* HalfEdge structure for arbitrary polygonal face

Triangle case (c is the halfedge):
              v
              /\
             /  \ 
            /    \
           /angle \
          /        \
         /          \
        /            \
       / p          n \
      /                \
     /         t        \
    /                    \
   /                      \
  /         -- c -->       \
 /__________________________\
 s         <-- o --        e
 
Quad case (c is the halfedge):
   v                         w
   ---------------------------
   | angle  <-- x --         |
   |                         |
   |                         |
   |                         |
   | p          t          n |
   |                         |
   |                         |
   |                         |
   |         -- c -->        |
   ---------------------------
   s        <-- o --         e

If the face has more than 4 vertices, we can only access v, s, e, t, n, p, o.

   v 
   ---------...
   | 
   |                           ...
   |                           /
   | p          t          n  /                   
   |                         /
   |                        /
   |       -- c -->        /
   _______________________/
   s      <-- o --        e
*/
struct HalfEdge {
    Vertex* s = NULL; // start vertex
    Vertex* e = NULL; // end vertex
    Vertex* v = NULL; // opposite vertex (trig-or-quad-only)
    Vertex* w = NULL; // next opposite vertex (quad-only)
    Facet* t = NULL; // face
    HalfEdge* n = NULL; // next half edge
    HalfEdge* p = NULL; // previous half edge
    HalfEdge* o = NULL; // opposite half edge (NULL if at boundary)
    HalfEdge* x = NULL; // fronting half edge (quad-only)

    float angle = 0; // angle at opposite vertex v (trig-or-quad-only)
    int i = -1; // index inside the face

    bool is_quad() const { return w != NULL; }

    Vector3f mid_point() const {
        return Vector3f(*s + *e) / 2;
    }

    Vector3f lower_point() const {
        return *s < *e ? Vector3f(*s) : Vector3f(*e);
    }

    // comparison operator
    bool operator<(const HalfEdge& e) const {
        // boundary edge first
        if (o == NULL) return true;
        else if (e.o == NULL) return false;
        else return lower_point() < e.lower_point();
    }

    // parallelogram error (trig-only)
    float parallelogram_error() {
        if (o == NULL) return INF;
        else return Vector3f(*e + *s - *v, *o->v).norm();
    }
};

struct Facet {
    vector<Vertex*> vertices;
    vector<HalfEdge*> half_edges;

    int i = -1; // index
    int ic = -1; // connected component index
    int m = 0; // visited mark

    Vector3f center; // mass center
    float area; // face area
    Vector3f normal; // face normal

    // update status
    void update() {
        // update center
        center = Vector3f(0, 0, 0);
        for (size_t i = 0; i < vertices.size(); i++) {
            center = center + Vector3f(*vertices[i]);
        }
        center = center / vertices.size();
        // update area by summing up the area of all triangles (assume vertices are ordered)
        area = 0;
        for (size_t i = 0; i < half_edges.size(); i++) {
            area += get_trig_area(*vertices[0], *half_edges[i]->s, *half_edges[i]->e);
        }
        // update face normal (we always assume the face is planar and use the first 3 vertices to estimate the normal)
        Vector3f e1(*vertices[0], *vertices[1]);
        Vector3f e2(*vertices[0], *vertices[2]);
        normal = e1.cross(e2).normalize();
    }

    // flip the face orientation
    void flip() {
        // flip half edge directions
        for (size_t i = 0; i < half_edges.size(); i++) {
            swap(half_edges[i]->s, half_edges[i]->e);
            swap(half_edges[i]->n, half_edges[i]->p);
            if (half_edges[i]->w != NULL) swap(half_edges[i]->v, half_edges[i]->w);
        }
        // reverse vertices
        for (size_t i = 0; i < vertices.size() / 2; i++) {
            swap(vertices[i], vertices[vertices.size() - i - 1]);
        }
    }

    // comparison operator
    bool operator<(const Facet& f) const {
        // first by connected component, then by center
        if (ic != f.ic) return ic < f.ic;
        else return center < f.center;
    }
};

// ostream for HalfEdge, since we also use definition of Facet, it must be defined after both classes...
ostream& operator<<(ostream &os, const HalfEdge &ee) {
    os << "HalfEdge f " << ee.t->i << " : v " << ee.s->i << " --> v " << ee.e->i ;
    return os;
}

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

    // total surface area of all faces
    float total_area = 0;

    // if watertight (edge-manifold and vertex-manifold, but we don't consider vertex-manifold, as isolated vertices should be cleaned in preprocessing)
    map<int, bool> component_is_watertight; // per connected component
    bool is_watertight = true; // the whole mesh

    // component centers
    map<int, Vector3f> component_centers;
    map<int, vector<Facet*>> component_faces;

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

        // discretize verts (assume in [-1, 1], we won't do error handling in cpp!)
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
                    // if this key has already matched two half_edges, this mesh is not edge-manifold!
                    if (edge2halfedge[key] == NULL) {
                        is_watertight = false;
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
                component_centers[num_components] = Vector3f(0, 0, 0);
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
                        }
                    }
                    component_centers[num_components] += f->center;
                    component_faces[num_components].push_back(f);
                }
                component_centers[num_components] /= component_faces[num_components].size();
                // cout << "[MESH] component " << num_components << " has " << component_faces[num_components].size() << " faces, center is " << component_centers[num_components] << endl;
                num_components++;
            }
        }

        if (verbose) cout << "[MESH] V = " << num_verts << ", E = " << num_edges << ", F = " << num_faces << ", C = " << num_components << endl;

        // sort faces using connected component and center
        sort(faces.begin(), faces.end(), [](const Facet* f1, const Facet* f2) { return *f1 < *f2; });

        // reset face index
        for (size_t i = 0; i < faces.size(); i++) { faces[i]->i = i; }
    }

    // empirically repair face orientation
    void repair_face_orientation() {
        // for each component, we choose the furthest face from its center
        for (int i = 0; i < num_components; i++) {
            float max_dist = -1;
            Facet* furthest_face = NULL;
            for (size_t j = 0; j < component_faces[i].size(); j++) {
                Facet* f = component_faces[i][j];
                float dist = (f->center - component_centers[i]).norm();
                if (dist > max_dist) {
                    max_dist = dist;
                    furthest_face = f;
                }
            }
            // see if this face's orientation is correct
            Vector3f normal = furthest_face->normal;
            Vector3f dir = (furthest_face->center - component_centers[i]).normalize();
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
            cout << "DEBUG: polygonize merge " << e->t->i << " and " << e->o->t->i << endl;
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
                cout << "DEBUG: delete vertex " << e->s->i << endl;
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
                cout << "DEBUG: delete vertex " << e->e->i << endl;
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
                cout << "DEBUG: append half edge " << *cur << endl;
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
            cout << "DEBUG: merged into new face " << q->i << endl;
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

    ~Mesh() {
        for (Vertex* v : verts) { delete v; }
        for (Facet* f : faces) {
            for (HalfEdge* e : f->half_edges) { delete e; }
            delete f;
        }
    }
};
