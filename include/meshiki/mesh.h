#pragma once

#include <iostream>
#include <vector>
#include <map>
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
    vector<HalfEdge*> edges; // incident edges
    
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
    Vector3f(const Vertex& v1, const Vertex& v2) : x(v2.x - v1.x), y(v2.y - v1.y), z(v2.z - v1.z) {}
    Vector3f operator+(const Vector3f& v) const {
        return Vector3f(x + v.x, y + v.y, z + v.z);
    }
    Vector3f operator-(const Vector3f& v) const {
        return Vector3f(x - v.x, y - v.y, z - v.z);
    }
    Vector3f operator*(float s) const {
        return Vector3f(x * s, y * s, z * s);
    }
    Vector3f operator/(float s) const {
        return Vector3f(x / s, y / s, z / s);
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
        float n = norm();
        return Vector3f(x / n, y / n, z / n);
    }
    friend ostream& operator<<(ostream &os, const Vector3f &v) {
        os << "Vector3f :(" << v.x << ", " << v.y << ", " << v.z << ")";
        return os;
    }
};

inline float angle_between(Vector3f a, Vector3f b) {
    float radian = acos(a.dot(b) / (a.norm() * b.norm() + 1e-8));
    return radian * 180 / M_PI;
}

inline float get_trig_area(const Vertex& v1, const Vertex& v2, const Vertex& v3) {
    Vector3f e1(v1, v2);
    Vector3f e2(v1, v3);
    return 0.5 * (e1.cross(e2)).norm();
}

struct HalfEdge {
    Vertex* v = NULL; // opposite vertex
    Vertex* s = NULL; // start vertex
    Vertex* e = NULL; // end vertex
    Vertex* w = NULL; // next opposite vertex (quad-only, NULL for trig)
    Facet* t = NULL; // face
    HalfEdge* n = NULL; // next half edge
    HalfEdge* x = NULL; // fronting half edge (quad-only, NULL for trig)
    HalfEdge* p = NULL; // previous half edge
    HalfEdge* o = NULL; // opposite half edge (NULL if at boundary)
    float angle = 0; // angle at vertex v
    int i = -1; // index

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

    // coplane error (trig-only), = 180 - dihedral angle
    float coplane_error() {
        if (o == NULL) return INF;
        Vector3f n1 = Vector3f(*v, *s).cross(Vector3f(*v, *e)).normalize();
        Vector3f n2 = Vector3f(*o->v, *o->s).cross(Vector3f(*o->v, *o->e)).normalize();
        float radian = acos(n1.dot(n2));
        return radian * 180 / M_PI; // degree in [0, 180]
    }
};

struct Facet {
    vector<Vertex*> vertices;
    vector<HalfEdge*> half_edges;

    int i = -1; // index
    int ic = -1; // connected component index
    int m = 0; // visited mark

    Vector3f center; // mass center
    float area; // area

    // flip the face orientation (only flip half edges)
    void flip() {
        for (size_t i = 0; i < half_edges.size(); i++) {
            swap(half_edges[i]->s, half_edges[i]->e);
            swap(half_edges[i]->n, half_edges[i]->p);
            if (half_edges[i]->w != NULL) swap(half_edges[i]->v, half_edges[i]->w);
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
    if (ee.is_quad()) {
        os << "HalfEdge quad " << ee.t->i << " : (" << ee.v->i << ", " << ee.s->i << ", " << ee.e->i << ", " << ee.w->i << ")";
    } else {
        os << "HalfEdge trig " << ee.t->i << " : (" << ee.v->i << ", " << ee.s->i << ", " << ee.e->i << ")";
    }
    return os;
}


class Mesh {
public:

    // mesh data
    vector<Vertex*> verts;
    vector<Facet*> faces;

    bool verbose = false;
    
    // Euler characteristic: V - E + F = 2 - 2g - b
    int num_vertices = 0;
    int num_edges = 0;
    int num_faces = 0;
    int num_components = 0;
    bool non_manifold = false;

    // indicator for quad quality (will be set during initialization)
    float rect_error = 0;
    int num_quad = 0;

    // total surface area of all faces
    float total_area = 0;

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
        num_vertices = verts.size();
       
        // build face and edge
        map<pair<int, int>, HalfEdge*> edge2halfedge; // to hold twin half edge
        for (size_t i = 0; i < faces_input.size(); i++) {
            vector<int>& f_in = faces_input[i];
            Facet* f = new Facet();
            f->i = i;
            bool is_quad = f_in.size() == 4;
            // build half edge and link to verts
            float cur_quad_angle = 0;
            for (int j = 0; j < int(f_in.size()); j++) {
                HalfEdge* e = new HalfEdge();
                e->t = f;
                e->i = j;
                // we only handle trig and quad cases
                if (!is_quad) {
                    e->v = verts[f_in[j]];
                    e->s = verts[f_in[(j + 1) % 3]]; verts[f_in[(j + 1) % 3]]->edges.push_back(e);
                    e->e = verts[f_in[(j + 2) % 3]]; verts[f_in[(j + 2) % 3]]->edges.push_back(e);
                    e->angle = angle_between(Vector3f(*e->v, *e->s), Vector3f(*e->v, *e->e));
                } else {
                    e->v = verts[f_in[j]];
                    e->s = verts[f_in[(j + 1) % 4]]; verts[f_in[(j + 1) % 4]]->edges.push_back(e);
                    e->e = verts[f_in[(j + 2) % 4]]; verts[f_in[(j + 2) % 4]]->edges.push_back(e);
                    e->w = verts[f_in[(j + 3) % 4]];
                    e->angle = angle_between(Vector3f(*e->v, *e->s), Vector3f(*e->v, *e->w));
                    // update quad weight
                    cur_quad_angle += abs(90 - e->angle) / 4;
                }
                f->vertices.push_back(verts[f_in[j]]);
                f->half_edges.push_back(e);
            }
            if (is_quad) {
                // update quad stat
                rect_error = (rect_error * num_quad + cur_quad_angle) / (num_quad + 1);
                num_quad++;
            }
            // link prev and next half edges
            // assume each face's vertex ordering is counter-clockwise, so next = right, prev = left
            for (int j = 0; j < int(f->half_edges.size()); j++) {
                f->half_edges[j]->n = f->half_edges[(j + 1) % f->half_edges.size()];
                f->half_edges[j]->p = f->half_edges[(j - 1 + f->half_edges.size()) % f->half_edges.size()];
                f->half_edges[j]->x = f->half_edges[(j + 2) % f->half_edges.size()];
            }
            // link opposite half edges
            for (int j = 0; j < int(f->half_edges.size()); j++) {
                HalfEdge* e = f->half_edges[j];
                // link opposite half edge
                pair<int, int> key = edge_key(f_in[(j + 1) % f_in.size()], f_in[(j + 2) % f_in.size()]);
                if (edge2halfedge.find(key) == edge2halfedge.end()) {
                    edge2halfedge[key] = e;
                } else {
                    // if this key has already matched two half edges, this mesh is not edge-manifold!
                    if (edge2halfedge[key] == NULL) {
                        non_manifold = true;
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
            Vector3f center(*f->vertices[0]);
            for (int j = 1; j < int(f->vertices.size()); j++) {
                center = center + Vector3f(*f->vertices[j]);
            }
            f->center = center / f->vertices.size();
            float area = 0;
            for (int j = 1; j < int(f->vertices.size()) - 1; j++) {
                area += get_trig_area(*f->vertices[0], *f->vertices[j], *f->vertices[j + 1]); // naive fan-cut
            }
            f->area = area;
            total_area += area;
            faces.push_back(f);
        }

        num_faces = faces.size();
        num_edges = edge2halfedge.size();

        for (size_t i = 0; i < faces_input.size(); i++) {
            Facet* f = faces[i];
            for (int j = 0; j < int(f->half_edges.size()); j++) {
                // boundary edges have no opposite half edge, and their start and end vertices are boundary vertices
                if (f->half_edges[j]->o == NULL) {
                    f->half_edges[j]->s->m = 1;
                    f->half_edges[j]->e->m = 1;
                    // if (verbose) cout << "[MESH] Mark boundary vertex for face " << f->i << " : " << f->half_edges[j]->s->i << " -- " << f->half_edges[j]->e->i << endl;
                }
            }
            // sort half edges (this will disturb vertex-half-edge order, but we don't use index anyway)
            sort(f->half_edges.begin(), f->half_edges.end(), [](const HalfEdge* e1, const HalfEdge* e2) { return *e1 < *e2; });
        }

        // sort faces using center
        sort(faces.begin(), faces.end(), [](const Facet* f1, const Facet* f2) { return *f1 < *f2; });

        // find connected components
        for (size_t i = 0; i < faces_input.size(); i++) {
            Facet* f = faces[i];
            if (f->ic == -1) {
                num_components++;
                // if (verbose) cout << "[MESH] find connected component " << num_components << endl;
                // recursively mark all connected faces
                queue<Facet*> q;
                q.push(f);
                while (!q.empty()) {
                    Facet* f = q.front();
                    q.pop();
                    if (f->ic != -1) continue;
                    f->ic = num_components;
                    for (int j = 0; j < int(f->half_edges.size()); j++) {
                        HalfEdge* e = f->half_edges[j];
                        if (e->o != NULL && e->o->t->ic == -1) {
                            q.push(e->o->t);
                        }
                    }
                }
            }
        }

        if (verbose) cout << "[MESH] V = " << num_vertices << ", E = " << num_edges << ", F = " << num_faces << ", C = " << num_components << ", manifold = " << (non_manifold ? "False" : "True") << endl;

        // sort faces again using connected component and center
        sort(faces.begin(), faces.end(), [](const Facet* f1, const Facet* f2) { return *f1 < *f2; });

        // // reset face index
        // for (size_t i = 0; i < faces.size(); i++) { faces[i]->i = i; }
    }

    // trig-to-quad conversion (in-place)
    void quadrangulate(float thresh_bihedral, float thresh_convex) {

        vector<Facet*> faces_old = faces;
        int num_faces_ori = faces.size();
        num_quad = 0;
        rect_error = 0;

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

            // build half edges
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
            // opposite half edges (mutually)
            e1->o = e->o->n->o; if (e1->o != NULL) e1->o->o = e1;
            e2->o = e->o->p->o; if (e2->o != NULL) e2->o->o = e2;
            e3->o = e->n->o; if (e3->o != NULL) e3->o->o = e3;
            e4->o = e->p->o; if (e4->o != NULL) e4->o->o = e4;
            // we don't delete isolated faces here, but mark them
            e->t->m = 2;
            e->o->t->m = 2;

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
                if (e->v->i == e->o->v->i) continue; // duplicate faces (rare...)
                if (e->o->t->half_edges.size() > 3) continue; // quad opposite face
                if (e->n->angle + e->o->p->angle >= thresh_convex || e->p->angle + e->o->n->angle >= thresh_convex) continue; // quad must be convex
                if (e->coplane_error() > thresh_bihedral) continue; // coplane error

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

    // salient point sampling
    vector<vector<float>> salient_point_sample(int num_samples, float thresh_angle) {
        vector<vector<float>> samples;
        // loop edges and calculate the dihedral angle
        set<pair<int, int>> visited_edges;
        vector<tuple<int, int, float>> salient_edges; // start vert index, end vert index, length
        float total_edge_length = 0;
        for (Facet* f : faces) {
            for (HalfEdge* e : f->half_edges) {
                if (e->o == NULL) continue; // boundary edge
                if (visited_edges.find(edge_key(e->s->i, e->e->i)) != visited_edges.end()) continue;
                visited_edges.insert(edge_key(e->s->i, e->e->i));
                float angle = e->coplane_error(); // 180 - dihedral angle
                if (angle > thresh_angle) {
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
            for (int i = 0; i < salient_edges.size(); i++) {
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
        for (int i = 0; i < faces.size(); i++) {
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
