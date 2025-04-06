#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <algorithm>
#include <cmath>

#include <meshiki/utils.h>
#include <meshiki/elements.h>

using namespace std;


struct BoundingBox {
    Vector3f mn = Vector3f(INF, INF, INF);
    Vector3f mx = Vector3f(-INF, -INF, -INF);

    Vector3f size() const {
        return mx - mn;
    }

    float extent() const {
        return (mx - mn).norm();
    }

    Vector3f center() const {
        return (mn + mx) / 2;
    }

    void translate(const Vector3f& v) {
        mn = mn + v;
        mx = mx + v;
    }

    float volume() const {
        Vector3f size = mx - mn;
        return size.x * size.y * size.z;
    }

    void expand(const Vector3f& v) {
        mn = min(mn, v);
        mx = max(mx, v);
    }

    void expand(const Vertex* v) {
        expand(Vector3f(*v));
    }

    void expand(const Facet* f) {
        for (size_t i = 0; i < f->vertices.size(); i++) {
            expand(f->vertices[i]);
        }
    }

    void expand(const BoundingBox& other) {
        mn = min(mn, other.mn);
        mx = max(mx, other.mx);
    }

    bool overlap(const BoundingBox& other, float thresh = 0.01) const {
        // thresh can adjust the overlap tolerance, a positive thresh can be used for coarse contact detection
        return (mn.x - other.mx.x) <= thresh && (other.mn.x - mx.x) <= thresh &&
               (mn.y - other.mx.y) <= thresh && (other.mn.y - mx.y) <= thresh &&
               (mn.z - other.mx.z) <= thresh && (other.mn.z - mx.z) <= thresh;
    }
};

// simple BVH implementation
struct BVHNode {
    BoundingBox bbox;
    BVHNode* left = NULL;
    BVHNode* right = NULL;
    vector<Facet*> faces; // only at BVH leaf node (trig-only)

    // move the whole BVH tree's bbox
    void translate(const Vector3f& v) {
        bbox.translate(v);
        if (left) left->translate(v);
        if (right) right->translate(v);
    }

    ~BVHNode() {
        if (left) delete left;
        if (right) delete right;
    }
};

BVHNode* build_bvh(vector<Facet*>& faces, int depth = 0, int max_depth = 16, int min_leaf_size = 4) {
    
    if (faces.empty()) return NULL;

    BVHNode* node = new BVHNode();
    for (size_t i = 0; i < faces.size(); i++) {
        node->bbox.expand(faces[i]);
    }

    if (faces.size() <= min_leaf_size || depth >= max_depth) {
        node->faces = faces; // copy
        return node;
    }

    // find longest axis
    Vector3f size = node->bbox.size();
    int longest_axis = 0;
    if (size.y > size.x) longest_axis = 1;
    if (size.z > size.y) longest_axis = 2;

    // split the faces into two groups
    sort(faces.begin(), faces.end(), [&](const Facet* a, const Facet* b) {
        return a->center[longest_axis] < b->center[longest_axis];
    });

    // find the median
    int median = faces.size() / 2;
    vector<Facet*> left_faces(faces.begin(), faces.begin() + median);
    vector<Facet*> right_faces(faces.begin() + median, faces.end());

    // recursively build the BVH
    node->left = build_bvh(left_faces, depth + 1, max_depth, min_leaf_size);
    node->right = build_bvh(right_faces, depth + 1, max_depth, min_leaf_size);

    return node;
}

// triangle-triangle intersection test
bool intersect_face(const Facet* a, const Facet* b) {
    // Make sure both are triangles
    if (a->vertices.size() != 3 || b->vertices.size() != 3) {
        return false;
    }
    
    // Get vertices
    const Vertex* a0 = a->vertices[0];
    const Vertex* a1 = a->vertices[1];
    const Vertex* a2 = a->vertices[2];
    
    const Vertex* b0 = b->vertices[0];
    const Vertex* b1 = b->vertices[1];
    const Vertex* b2 = b->vertices[2];
    
    // Compute plane equation for triangle A: n·(x - a0) = 0
    Vector3f ab = Vector3f(*a0, *a1);
    Vector3f ac = Vector3f(*a0, *a2);
    Vector3f n_a = ab.cross(ac).normalize();
    
    // Compute signed distances from triangle B vertices to plane A
    float dist_b0 = n_a.dot(Vector3f(*b0) - Vector3f(*a0));
    float dist_b1 = n_a.dot(Vector3f(*b1) - Vector3f(*a0));
    float dist_b2 = n_a.dot(Vector3f(*b2) - Vector3f(*a0));
    
    // Check if triangle B is completely on one side of plane A
    if ((dist_b0 > 0 && dist_b1 > 0 && dist_b2 > 0) || 
        (dist_b0 < 0 && dist_b1 < 0 && dist_b2 < 0)) {
        return false;
    }
    
    // Compute plane equation for triangle B: n·(x - b0) = 0
    Vector3f ba = Vector3f(*b0, *b1);
    Vector3f bc = Vector3f(*b0, *b2);
    Vector3f n_b = ba.cross(bc).normalize();
    
    // Compute signed distances from triangle A vertices to plane B
    float dist_a0 = n_b.dot(Vector3f(*a0) - Vector3f(*b0));
    float dist_a1 = n_b.dot(Vector3f(*a1) - Vector3f(*b0));
    float dist_a2 = n_b.dot(Vector3f(*a2) - Vector3f(*b0));
    
    // Check if triangle A is completely on one side of plane B
    if ((dist_a0 > 0 && dist_a1 > 0 && dist_a2 > 0) || 
        (dist_a0 < 0 && dist_a1 < 0 && dist_a2 < 0)) {
        return false;
    }
    
    // Compute the direction of the line of intersection between planes A and B
    Vector3f line_dir = n_a.cross(n_b);
    
    // Handle coplanar case
    if (line_dir.norm() < 1e-6) {
        // Triangles are coplanar, check for 2D overlap
        // Project triangles onto the dominant plane
        
        // Find the dominant axis of the normal (largest component)
        int axis = 0;
        float max_component = std::abs(n_a.x);
        if (std::abs(n_a.y) > max_component) {
            axis = 1;
            max_component = std::abs(n_a.y);
        }
        if (std::abs(n_a.z) > max_component) {
            axis = 2;
        }
        
        // Project onto the plane perpendicular to the dominant axis
        int u = (axis + 1) % 3;
        int v = (axis + 2) % 3;
        
        // Perform 2D triangle-triangle intersection test
        // For simplicity, use the separating axis theorem
        Vector3f axes[6] = {
            Vector3f(a1->x - a0->x, a1->y - a0->y, a1->z - a0->z),
            Vector3f(a2->x - a1->x, a2->y - a1->y, a2->z - a1->z),
            Vector3f(a0->x - a2->x, a0->y - a2->y, a0->z - a2->z),
            Vector3f(b1->x - b0->x, b1->y - b0->y, b1->z - b0->z),
            Vector3f(b2->x - b1->x, b2->y - b1->y, b2->z - b1->z),
            Vector3f(b0->x - b2->x, b0->y - b2->y, b0->z - b2->z)
        };
        
        for (int i = 0; i < 6; i++) {
            Vector3f normal(0, 0, 0);
            normal[u] = -axes[i][v];
            normal[v] = axes[i][u];
            
            float min_a = INF, max_a = -INF;
            float min_b = INF, max_b = -INF;
            
            // Project vertices of triangle A
            float proj_a0 = normal.dot(Vector3f(*a0));
            float proj_a1 = normal.dot(Vector3f(*a1));
            float proj_a2 = normal.dot(Vector3f(*a2));
            min_a = min(min(proj_a0, proj_a1), proj_a2);
            max_a = max(max(proj_a0, proj_a1), proj_a2);
            
            // Project vertices of triangle B
            float proj_b0 = normal.dot(Vector3f(*b0));
            float proj_b1 = normal.dot(Vector3f(*b1));
            float proj_b2 = normal.dot(Vector3f(*b2));
            min_b = min(min(proj_b0, proj_b1), proj_b2);
            max_b = max(max(proj_b0, proj_b1), proj_b2);
            
            // Check for separation
            if (min_a > max_b || min_b > max_a) {
                return false;
            }
        }
        
        return true;
    }
    
    // Compute interval for triangle A
    float t_a[2] = {0, 0};
    int count_a = 0;
    
    // Check edge a0a1
    if ((dist_a0 * dist_a1) <= 0 && dist_a0 != dist_a1) {
        t_a[count_a++] = dist_a0 / (dist_a0 - dist_a1);
    }
    
    // Check edge a1a2
    if ((dist_a1 * dist_a2) <= 0 && dist_a1 != dist_a2) {
        t_a[count_a++] = dist_a1 / (dist_a1 - dist_a2);
    }
    
    // Check edge a2a0
    if (count_a < 2 && (dist_a2 * dist_a0) <= 0 && dist_a2 != dist_a0) {
        t_a[count_a++] = dist_a2 / (dist_a2 - dist_a0);
    }
    
    if (count_a < 2) return false; // Triangle A doesn't intersect plane B properly
    
    // Compute interval for triangle B
    float t_b[2] = {0, 0};
    int count_b = 0;
    
    // Check edge b0b1
    if ((dist_b0 * dist_b1) <= 0 && dist_b0 != dist_b1) {
        t_b[count_b++] = dist_b0 / (dist_b0 - dist_b1);
    }
    
    // Check edge b1b2
    if ((dist_b1 * dist_b2) <= 0 && dist_b1 != dist_b2) {
        t_b[count_b++] = dist_b1 / (dist_b1 - dist_b2);
    }
    
    // Check edge b2b0
    if (count_b < 2 && (dist_b2 * dist_b0) <= 0 && dist_b2 != dist_b0) {
        t_b[count_b++] = dist_b2 / (dist_b2 - dist_b0);
    }
    
    if (count_b < 2) return false; // Triangle B doesn't intersect plane A properly
    
    // Make sure t_a and t_b are ordered
    if (t_a[0] > t_a[1]) std::swap(t_a[0], t_a[1]);
    if (t_b[0] > t_b[1]) std::swap(t_b[0], t_b[1]);
    
    // Compute intersection points on triangle A
    Vector3f p_a1 = Vector3f(*a0) + (Vector3f(*a1) - Vector3f(*a0)) * t_a[0];
    Vector3f p_a2 = Vector3f(*a0) + (Vector3f(*a2) - Vector3f(*a0)) * t_a[1];
    
    // Compute intersection points on triangle B
    Vector3f p_b1 = Vector3f(*b0) + (Vector3f(*b1) - Vector3f(*b0)) * t_b[0];
    Vector3f p_b2 = Vector3f(*b0) + (Vector3f(*b2) - Vector3f(*b0)) * t_b[1];
    
    // Compute parameters for points on line of intersection
    float param_a1 = line_dir.dot(p_a1);
    float param_a2 = line_dir.dot(p_a2);
    float param_b1 = line_dir.dot(p_b1);
    float param_b2 = line_dir.dot(p_b2);
    
    // Order parameters
    if (param_a1 > param_a2) std::swap(param_a1, param_a2);
    if (param_b1 > param_b2) std::swap(param_b1, param_b2);
    
    // Check if intervals overlap
    float max_min = max(param_a1, param_b1);
    float min_max = min(param_a2, param_b2);
    
    return max_min <= min_max;
}

bool intersect_bvh(const BVHNode* a, const BVHNode* b, bool bbox_only = false) {
    if (!a || !b || !a->bbox.overlap(b->bbox)) return false;

    if (!a->left && !a->right && !b->left && !b->right) {
        // both are leaf nodes
        if (bbox_only) {
            // simply perform bbox overlap test
            return a->bbox.overlap(b->bbox);
        } else {
            // perform triangle-triangle intersection test
            for (const auto& fa : a->faces) {
                for (const auto& fb : b->faces) {
                    if (intersect_face(fa, fb)) return true;
                }
            }
            return false;
        }
    }

    // recursively test children
    if (a->left && intersect_bvh(a->left, b)) return true;
    if (a->right && intersect_bvh(a->right, b)) return true;
    if (b->left && intersect_bvh(a, b->left)) return true;
    if (b->right && intersect_bvh(a, b->right)) return true;

    return false;
}