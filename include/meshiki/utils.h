#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <set>
#include <algorithm>
#include <cmath>

#include <nanoflann.h>
#include <pcg32.h>

using namespace std;

static float INF = 1e8;

// helper class for nanoflann
template <typename T>
struct PointCloud {
    struct Point {
        T x, y, z;
    };

    using coord_t = T;  //!< The type of each coordinate

    std::vector<Point> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const {
        return false;
    }
};

// unique edge key
inline pair<int, int> edge_key(int a, int b) {
    return a < b ? make_pair(a, b) : make_pair(b, a);
}

// simple quad-triangulation ("fixed" mode in blender, always cut v0-v2 edge)
vector<vector<int>> triangulate(const vector<vector<int>> faces_input) {
    vector<vector<int>> faces;
    for (size_t i = 0; i < faces_input.size(); i++) {
        const vector<int>& f_in = faces_input[i];
        if (f_in.size() == 3) {
            faces.push_back(f_in);
        } else { // assert 4, not supporting polygon >= 5
            faces.push_back({f_in[0], f_in[1], f_in[2]});
            faces.push_back({f_in[0], f_in[2], f_in[3]});
        }
    }
    return faces;
}

// pad triangles to fake-quad (degenerated-quad by repeating last vertex)
vector<vector<int>> pad_to_quad(const vector<vector<int>> faces_input) {
    vector<vector<int>> faces;
    for (size_t i = 0; i < faces_input.size(); i++) {
        const vector<int>& f_in = faces_input[i];
        if (f_in.size() == 3) {
            faces.push_back({f_in[0], f_in[1], f_in[2], f_in[2]});
        } else { // assert 4, not supporting polygon >= 5
            faces.push_back(f_in);
        }
    }
    return faces;
}

// deduplicate / merge close vertices and reindex faces
tuple<vector<vector<float>>, vector<vector<int>>> merge_close_vertices(const vector<vector<float>> verts_input, const vector<vector<int>> faces_input, const float thresh=1e-5, bool verbose=false) {
    vector<vector<float>> verts;
    vector<vector<int>> faces;
    map<int, int> vmap;

    if (verbose) {
        cout << "[INFO] merge_close_vertices: before " << verts_input.size() << " vertices, " << faces_input.size() << " faces." << endl;
    }

    // deduplicate using octree radius search
    PointCloud<float> pointcloud;
    using Octree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Adaptor<float, PointCloud<float>>, PointCloud<float>, 3>;
    
    pointcloud.pts.resize(verts_input.size());
    for (size_t i = 0; i < verts_input.size(); i++) {
        pointcloud.pts[i].x = verts_input[i][0];
        pointcloud.pts[i].y = verts_input[i][1];
        pointcloud.pts[i].z = verts_input[i][2];
    }
    Octree octree(3, pointcloud, {10});

    for (size_t i = 0; i < verts_input.size(); i++) {

        if (vmap.find(i) != vmap.end()) continue; // already merged
        
        vector<float> v = verts_input[i];
        int idx = verts.size();
        vmap[i] = idx;
        verts.push_back(v);

        // radius search
        std::vector<nanoflann::ResultItem<uint32_t, float>> ret_matches;
        const size_t nMatches = octree.radiusSearch(&v[0], thresh, ret_matches); // thresh is the radius, excluding self

        for (size_t j = 0; j < nMatches; j++) {
            vmap[ret_matches[j].first] = idx;
        }
    }

    auto get_trig_area = [&](int a, int b, int c) {
        vector<float>& v1 = verts[a];
        vector<float>& v2 = verts[b];
        vector<float>& v3 = verts[c];
        vector<float> e1 = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
        vector<float> e2 = {v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]};
        return 0.5 * sqrt(pow(e1[1] * e2[2] - e1[2] * e2[1], 2) + pow(e1[2] * e2[0] - e1[0] * e2[2], 2) + pow(e1[0] * e2[1] - e1[1] * e2[0], 2));
    };

    auto get_poly_area = [&](const vector<int>& f) {
        float area = 0;
        for (int i = 1; i < int(f.size()) - 1; i++) {
            area += get_trig_area(f[0], f[i], f[i + 1]); // naive fan-cut
        }
        return area;
    };

    // reindex faces and remove collapsed faces
    for (size_t i = 0; i < faces_input.size(); i++) {
        vector<int> face;
        for (int j = 0; j < int(faces_input[i].size()); j++) {
            face.push_back(vmap[faces_input[i][j]]);
        }
        // collapse detection by face area
        if (get_poly_area(face) < thresh) continue;
        faces.push_back(face);
    }

    if (verbose) {
        cout << "[INFO] merge_close_vertices: after " << verts.size() << " vertices, " << faces.size() << " faces." << endl;
    }

    return make_tuple(verts, faces);
}

// naive furthest point sampling
vector<vector<float>> fps(const vector<vector<float>> verts_input, int num_samples, int start_idx) {
    vector<vector<float>> samples;
    set<int> indices;
    vector<float> dists;
    int num_verts = verts_input.size();
    int num_features = verts_input[0].size();

    if (num_samples >= num_verts) return verts_input; // no need to sample

    // find the first sample
    samples.push_back(verts_input[start_idx]);
    indices.insert(start_idx);
    // compute distance to the first sample
    for (int i = 0; i < num_verts; i++) {
        if (i == start_idx) {
            dists.push_back(0);
        } else {
            float dist = 0;
            for (int j = 0; j < num_features; j++) {
                dist += pow(verts_input[i][j] - verts_input[start_idx][j], 2);
            }
            dists.push_back(dist);
        }
    }

    // find the rest samples
    for (int i = 1; i < num_samples; i++) {
        float max_dist = 0;
        int max_idx = -1;
        for (int j = 0; j < num_verts; j++) {
            if (indices.find(j) != indices.end()) continue; // already sampled
            if (dists[j] > max_dist) {
                max_dist = dists[j];
                max_idx = j;
            }
        }
        samples.push_back(verts_input[max_idx]);
        indices.insert(max_idx);
        for (int j = 0; j < num_verts; j++) {
            if (indices.find(j) != indices.end()) continue; // already sampled
            float dist = 0;
            for (int k = 0; k < num_features; k++) {
                dist += pow(verts_input[j][k] - verts_input[max_idx][k], 2);
            }
            dists[j] = min(dists[j], dist);
        }
    }

    return samples;
}

// efficient random subsample (return indices)
vector<int> random_subsample(int total, int num_samples, bool repeat=false) {
    vector<int> samples;
    pcg32 rng;
    
    assert(total >= num_samples);

    if (repeat) {
        for (int i = 0; i < num_samples; i++) {
            int idx = rng.nextUInt(total);
            samples.push_back(idx);
        }
    } else {
        // permute and take the first num_samples
        vector<int> perm(total);
        for (int i = 0; i < total; i++) {
            perm[i] = i;
        }
        rng.shuffle(perm.begin(), perm.end());
        for (int i = 0; i < num_samples; i++) {
            samples.push_back(perm[i]);
        }
    }

    return samples;
}


struct DisjointSet {
    vector<int> parent;
    
    DisjointSet(int n) {
        clear(n);
    }

    void clear(int n) {
        parent.resize(n);
        for (int i = 0; i < n; i++) { parent[i] = i; }
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    void merge(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        parent[rootX] = rootY;
    }
};