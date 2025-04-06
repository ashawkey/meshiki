#pragma once

#include <iostream>
#include <set>
#include <vector>

#include <meshiki/utils.h>

using namespace std;

// forward declarations
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

    // extra constructor for Vector3f from Vertex
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
    float operator[](int i) const {
        return i == 0 ? x : (i == 1 ? y : z);
    }
    float& operator[](int i) {
        return i == 0 ? x : (i == 1 ? y : z);
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
    float max_component() const {
        return max(x, max(y, z));
    }
    float min_component() const {
        return min(x, min(y, z));
    }
    Vector3f normalize() const {
        float n = norm() + 1e-8;
        return Vector3f(x / n, y / n, z / n);
    }
    friend ostream& operator<<(ostream &os, const Vector3f &v) {
        os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return os;
    }
};

Vector3f min(const Vector3f& a, const Vector3f& b) {
    return Vector3f(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

Vector3f max(const Vector3f& a, const Vector3f& b) {
    return Vector3f(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

float angle_between(Vector3f a, Vector3f b) {
    // Normalize vectors and compute dot product
    float dot = a.dot(b) / (a.norm() * b.norm() + 1e-8);
    // Clamp dot product to [-1, 1] to avoid domain errors with acos
    dot = max(-1.0f, min(1.0f, dot));
    float radian = acos(dot);
    return radian * 180 / M_PI;
}


float get_trig_area(const Vertex& v1, const Vertex& v2, const Vertex& v3) {
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
    int m = 0; // visited mark

    bool is_quad() const { return w != NULL; }

    Vector3f mid_point() const {
        return Vector3f(*s + *e) / 2;
    }

    Vector3f lower_point() const {
        return *s < *e ? Vector3f(*s) : Vector3f(*e);
    }

    Vector3f upper_point() const {
        return *s < *e ? Vector3f(*e) : Vector3f(*s);
    }

    // comparison operator
    bool operator<(const HalfEdge& e) const {
        // boundary edge first, otherwise by lower point
        if (o == NULL && e.o == NULL) return lower_point() < e.lower_point();
        else if (o == NULL) return true;
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
    os << "HalfEdge <f " << ee.t->i << " : v " << ee.s->i << " --> v " << ee.e->i << ">";
    return os;
}

// a BoundaryLoop is a set of half edges without opposite half edges
// if a mesh is not watertight, it must have one or more boundary loops
struct BoundaryLoop {
    
    // the edges and points of the boundary loop (in counter-clockwise order to form a loop/polygon)
    vector<HalfEdge*> edges;
    vector<Vector3f> points;

    // how many other boundary loops have been connected to this one
    // not necessarily means this loop is closed.
    int num_connected = 0;

    // find out a whole boundary loop given an edge on it
    void build(HalfEdge* e) {
        edges.clear();
        points.clear();
        num_connected = 0;

        edges.push_back(e);
        points.push_back(Vector3f(*e->s));
        HalfEdge* cur = e;
        while (true) {
            HalfEdge* next = cur->n;
            while (next->o != NULL) {
                next = next->o->n;
            }
            if (next == e) break;
            edges.push_back(next);
            points.push_back(Vector3f(*next->s));
            cur = next;
        }
    }

    // earcut the boundary loop into new triangles
    // vector<Facet*> earcut() {
    //     // only involves existing Vertex, but need to create new HalfEdge and Facet
    //     // 
    // }
};


// detect if two boundary loops may connect (share some common edges)
bool boundary_may_connect(const BoundaryLoop& a, const BoundaryLoop& b, float thresh = 1e-8) {
    // count how many points are shared
    int count = 0;
    for (size_t i = 0; i < a.points.size(); i++) {
        for (size_t j = 0; j < b.points.size(); j++) {
            if ((a.points[i] - b.points[j]).norm() < thresh) count++;
        }
    }
    float ratio = max(float(count) / a.points.size(), float(count) / b.points.size());
    return count >= 4 || ratio >= 0.5; // very empirical!
}