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

    bool overlap(const BoundingBox& other, float thresh = 0) const {
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


// Helper function to compute intervals for the line-triangle intersection
bool compute_intervals(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2, 
                      float d0, float d1, float d2, 
                      const Vector3f& line_dir, float intervals[2]) {
    int count = 0;
    
    // Check each edge for intersection with the plane
    if ((d0 * d1) <= 0) {
        // Edge v0-v1 intersects the plane
        float t = d0 / (d0 - d1);
        Vector3f intersection = v0 + (v1 - v0) * t;
        // Project onto the line direction to get the parameter
        intervals[count++] = line_dir.dot(intersection);
    }
    
    if ((d0 * d2) <= 0) {
        // Edge v0-v2 intersects the plane
        float t = d0 / (d0 - d2);
        Vector3f intersection = v0 + (v2 - v0) * t;
        intervals[count++] = line_dir.dot(intersection);
    }
    
    if ((d1 * d2) <= 0 && count < 2) {
        // Edge v1-v2 intersects the plane
        float t = d1 / (d1 - d2);
        Vector3f intersection = v1 + (v2 - v1) * t;
        intervals[count++] = line_dir.dot(intersection);
    }
    
    // Sort the intervals
    if (count == 2 && intervals[0] > intervals[1]) {
        swap(intervals[0], intervals[1]);
    }
    
    return count == 2;
}

// Helper function to check if an edge separates triangle points
bool edge_separation_test(const pair<float, float>& e1, const pair<float, float>& e2, 
                         const pair<float, float>& p1, const pair<float, float>& p2, const pair<float, float>& p3) {
    // Edge normal (perpendicular to the edge, pointing outward)
    float nx = -(e2.second - e1.second);
    float ny = e2.first - e1.first;
    
    // Check which side of the edge each point is on
    float d1 = nx * (p1.first - e1.first) + ny * (p1.second - e1.second);
    float d2 = nx * (p2.first - e1.first) + ny * (p2.second - e1.second);
    float d3 = nx * (p3.first - e1.first) + ny * (p3.second - e1.second);
    
    // If all points are on the same side, and it's the opposite side from the triangle containing the edge,
    // then this edge separates the triangles
    if ((d1 >= 0 && d2 >= 0 && d3 >= 0) || (d1 <= 0 && d2 <= 0 && d3 <= 0)) {
        return false; // Edge separates
    }
    
    return true; // Edge does not separate
}

// Helper function for the 2D separating axis test
bool edge_separates(const pair<float, float>& a0, const pair<float, float>& a1, const pair<float, float>& a2,
                   const pair<float, float>& b0, const pair<float, float>& b1, const pair<float, float>& b2) {
    // Check each edge of triangle A
    if (!edge_separation_test(a0, a1, b0, b1, b2)) return false;
    if (!edge_separation_test(a1, a2, b0, b1, b2)) return false;
    if (!edge_separation_test(a2, a0, b0, b1, b2)) return false;
    
    return true;
}

// Helper function for 2D triangle overlap test (for coplanar triangles)
bool triangle_overlap_2d(float ax0, float ay0, float ax1, float ay1, float ax2, float ay2,
                        float bx0, float by0, float bx1, float by1, float bx2, float by2) {
    // Check if any edge of triangle A separates triangle B
    if (!edge_separates({ax0, ay0}, {ax1, ay1}, {ax2, ay2}, {bx0, by0}, {bx1, by1}, {bx2, by2})) {
        return false;
    }
    
    // Check if any edge of triangle B separates triangle A
    if (!edge_separates({bx0, by0}, {bx1, by1}, {bx2, by2}, {ax0, ay0}, {ax1, ay1}, {ax2, ay2})) {
        return false;
    }
    
    // No separating edge found, triangles overlap
    return true;
}

bool intersect_face(const Facet* a, const Facet* b) {
    // For efficiency, quickly check if bounding boxes overlap
    Vector3f min_a(INF, INF, INF), max_a(-INF, -INF, -INF);
    Vector3f min_b(INF, INF, INF), max_b(-INF, -INF, -INF);
    
    // Compute bounding boxes
    for (const Vertex* v : a->vertices) {
        min_a = min(min_a, Vector3f(*v));
        max_a = max(max_a, Vector3f(*v));
    }
    
    for (const Vertex* v : b->vertices) {
        min_b = min(min_b, Vector3f(*v));
        max_b = max(max_b, Vector3f(*v));
    }
    
    // Check if bounding boxes don't overlap
    if (max_a.x < min_b.x || max_b.x < min_a.x ||
        max_a.y < min_b.y || max_b.y < min_a.y ||
        max_a.z < min_b.z || max_b.z < min_a.z) {
        return false;
    }
    
    // We only handle triangles for the intersection test
    if (a->vertices.size() != 3 || b->vertices.size() != 3) {
        // For non-triangular faces, we would need to triangulate them first
        // or use a different approach. For now, return false or implement triangulation.
        return false;
    }
    
    // Extract triangle vertices
    const Vector3f a0(*a->vertices[0]);
    const Vector3f a1(*a->vertices[1]);
    const Vector3f a2(*a->vertices[2]);
    
    const Vector3f b0(*b->vertices[0]);
    const Vector3f b1(*b->vertices[1]);
    const Vector3f b2(*b->vertices[2]);
    
    // Compute triangle A's normal
    Vector3f edge1_a(a1.x - a0.x, a1.y - a0.y, a1.z - a0.z);
    Vector3f edge2_a(a2.x - a0.x, a2.y - a0.y, a2.z - a0.z);
    Vector3f normal_a = edge1_a.cross(edge2_a).normalize();
    
    // Compute signed distances from triangle B's vertices to triangle A's plane
    float dist_b0 = normal_a.dot(b0 - a0);
    float dist_b1 = normal_a.dot(b1 - a0);
    float dist_b2 = normal_a.dot(b2 - a0);
    
    // If all vertices of B are on the same side of A's plane, no intersection
    if ((dist_b0 > 0 && dist_b1 > 0 && dist_b2 > 0) || 
        (dist_b0 < 0 && dist_b1 < 0 && dist_b2 < 0)) {
        return false;
    }
    
    // Compute triangle B's normal
    Vector3f edge1_b(b1.x - b0.x, b1.y - b0.y, b1.z - b0.z);
    Vector3f edge2_b(b2.x - b0.x, b2.y - b0.y, b2.z - b0.z);
    Vector3f normal_b = edge1_b.cross(edge2_b).normalize();
    
    // Compute signed distances from triangle A's vertices to triangle B's plane
    float dist_a0 = normal_b.dot(a0 - b0);
    float dist_a1 = normal_b.dot(a1 - b0);
    float dist_a2 = normal_b.dot(a2 - b0);
    
    // If all vertices of A are on the same side of B's plane, no intersection
    if ((dist_a0 > 0 && dist_a1 > 0 && dist_a2 > 0) || 
        (dist_a0 < 0 && dist_a1 < 0 && dist_a2 < 0)) {
        return false;
    }
    
    // Compute the direction of the intersection line
    Vector3f intersection_line = normal_a.cross(normal_b).normalize();
    
    // Check if intersection line is valid (normals aren't parallel)
    if (intersection_line.norm() < 1e-6) {
        // Triangles are coplanar - need special handling
        // For coplanar triangles, we check if projected triangles in 2D overlap
        // We can pick the axis with the largest normal component to project onto
        int axis = 0;
        float max_component = fabs(normal_a.x);
        if (fabs(normal_a.y) > max_component) {
            axis = 1;
            max_component = fabs(normal_a.y);
        }
        if (fabs(normal_a.z) > max_component) {
            axis = 2;
        }
        
        // Project onto the plane defined by the axis
        int axis1 = (axis + 1) % 3;
        int axis2 = (axis + 2) % 3;
        
        // Use 2D triangle-triangle overlap test
        return triangle_overlap_2d(
            a0[axis1], a0[axis2], a1[axis1], a1[axis2], a2[axis1], a2[axis2],
            b0[axis1], b0[axis2], b1[axis1], b1[axis2], b2[axis1], b2[axis2]
        );
    }
    
    // Compute the intervals where the planes intersect the triangles
    float t_a[2], t_b[2];
    if (!compute_intervals(a0, a1, a2, dist_a0, dist_a1, dist_a2, intersection_line, t_a) ||
        !compute_intervals(b0, b1, b2, dist_b0, dist_b1, dist_b2, intersection_line, t_b)) {
        return false;
    }
    
    // Check if the intervals overlap
    if (t_a[0] > t_b[1] || t_b[0] > t_a[1]) {
        return false;
    }
    
    // Intervals overlap, triangles intersect
    return true;
}



bool intersect_bvh(const BVHNode* a, const BVHNode* b) {
    if (!a || !b || !a->bbox.overlap(b->bbox)) return false;

    if (!a->left && !a->right && !b->left && !b->right) {
        // both are leaf nodes
        for (const auto& fa : a->faces) {
            for (const auto& fb : b->faces) {
                if (intersect_face(fa, fb)) return true;
            }
        }
        return false;
    }

    // recursively test children
    if (a->left && intersect_bvh(a->left, b)) return true;
    if (a->right && intersect_bvh(a->right, b)) return true;
    if (b->left && intersect_bvh(a, b->left)) return true;
    if (b->right && intersect_bvh(a, b->right)) return true;

    return false;
}