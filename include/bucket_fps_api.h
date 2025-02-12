#pragma once

#include <bucket_fps/KDLineTree.h>
#include <bucket_fps/KDTree.h>
#include <array>
#include <memory>
#include <utility>
#include <vector>

#ifndef BUCKET_FPS_MAX_DIM
#define BUCKET_FPS_MAX_DIM 8
#endif
constexpr size_t max_dim = BUCKET_FPS_MAX_DIM;

using quickfps::KDLineTree;
using quickfps::KDTree;
using quickfps::Point;

template <typename T, size_t DIM, typename S>
std::vector<Point<T, DIM, S>> raw_data_to_points(const float *raw_data,
                                                 size_t n_points, size_t dim) {
    std::vector<Point<T, DIM, S>> points;
    points.reserve(n_points);
    for (size_t i = 0; i < n_points; i++) {
        const float *ptr = raw_data + i * dim;
        points.push_back(Point<T, DIM, S>(ptr, i));
    }
    return points;
}

template <typename T, size_t DIM, typename S = T>
void kdtree_sample(const float *raw_data, size_t n_points, size_t dim,
                   size_t n_samples, size_t start_idx,
                   size_t *sampled_point_indices) {
    auto points = raw_data_to_points<T, DIM, S>(raw_data, n_points, dim);
    std::unique_ptr<Point<T, DIM, S>[]> sampled_points(
        new Point<T, DIM, S>[n_samples]);
    KDTree<T, DIM, S> tree(points.data(), n_points, sampled_points.get());
    tree.buildKDtree();
    tree.init(points[start_idx]);
    tree.sample(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        sampled_point_indices[i] = sampled_points[i].id;
    }
}

template <typename T, size_t DIM, typename S = T>
void kdline_sample(const float *raw_data, size_t n_points, size_t dim,
                   size_t n_samples, size_t start_idx, size_t height,
                   size_t *sampled_point_indices) {
    auto points = raw_data_to_points<T, DIM, S>(raw_data, n_points, dim);
    std::unique_ptr<Point<T, DIM, S>[]> sampled_points(
        new Point<T, DIM, S>[n_samples]);
    KDLineTree<T, DIM, S> tree(points.data(), n_points, height,
                               sampled_points.get());
    tree.buildKDtree();
    tree.init(points[start_idx]);
    tree.sample(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
        sampled_point_indices[i] = sampled_points[i].id;
    }
}

////////////////////////////////////////
//                                    //
//    Compile Time Function Helper    //
//                                    //
////////////////////////////////////////
using KDTreeFuncType = void (*)(const float *, size_t, size_t, size_t, size_t,
                                size_t *);
using KDLineFuncType = void (*)(const float *, size_t, size_t, size_t, size_t,
                                size_t, size_t *);

template <typename T, size_t Count, typename M, size_t... I>
constexpr std::array<T, Count> mapIndices(M &&m, std::index_sequence<I...>) {
    std::array<T, Count> result{m.template operator()<I + 1>()...};
    return result;
}

template <typename T, size_t Count, typename M>
constexpr std::array<T, Count> map_helper(M m) {
    return mapIndices<T, Count>(m, std::make_index_sequence<Count>());
}

template <typename T, typename S = T> struct kdtree_func_helper {
    template <size_t DIM> KDTreeFuncType operator()() {
        return &kdtree_sample<T, DIM, S>;
    }
};

template <typename T, typename S = T> struct kdline_func_helper {
    template <size_t DIM> KDLineFuncType operator()() {
        return &kdline_sample<T, DIM, S>;
    }
};

/////////////////
//             //
//    C API    //
//             //
/////////////////

extern "C" {
int c_bucket_fps_kdtree(const float *raw_data, size_t n_points, size_t dim,
                      size_t n_samples, size_t start_idx,
                      size_t *sampled_point_indices) {
    if (dim == 0 || dim > max_dim) {
        // only support 1 to MAX_DIM dimensions
        return 1;
    } else if (start_idx >= n_points) {
        // start_idx should be smaller than n_samples
        return 2;
    }
    auto func_arr = map_helper<KDTreeFuncType, max_dim>(kdtree_func_helper<float>{});
    func_arr[dim - 1](raw_data, n_points, dim, n_samples, start_idx,
                      sampled_point_indices);
    return 0;
}

int c_bucket_fps_kdline(const float *raw_data, size_t n_points, size_t dim,
                      size_t n_samples, size_t start_idx, size_t height,
                      size_t *sampled_point_indices) {
    if (dim == 0 || dim > max_dim) {
        // only support 1 to MAX_DIM dimensions
        return 1;
    } else if (start_idx >= n_points) {
        // start_idx should be smaller than n_samples
        return 2;
    }
    auto func_arr = map_helper<KDLineFuncType, max_dim>(kdline_func_helper<float>{});
    func_arr[dim - 1](raw_data, n_points, dim, n_samples, start_idx, height,
                      sampled_point_indices);
    return 0;
}
}


////////////////////////
//                    //
//      C++ API       //
//                    //
////////////////////////

std::vector<std::vector<float>> bucket_fps_kdtree(const std::vector<std::vector<float>> verts_input, size_t num_samples, size_t start_idx) {
    size_t n_points = verts_input.size();
    size_t dim = verts_input[0].size();
    std::vector<float> raw_data;
    raw_data.reserve(n_points * dim);
    for (const auto &v : verts_input) {
        raw_data.insert(raw_data.end(), v.begin(), v.end());
    }
    std::vector<size_t> sampled_point_indices(num_samples);
    c_bucket_fps_kdtree(raw_data.data(), n_points, dim, num_samples, start_idx, sampled_point_indices.data());
    std::vector<std::vector<float>> sampled_points(num_samples);
    for (size_t i = 0; i < num_samples; i++) {
        sampled_points[i] = verts_input[sampled_point_indices[i]];
    }
    return sampled_points;
}

std::vector<std::vector<float>> bucket_fps_kdline(const std::vector<std::vector<float>> verts_input, size_t num_samples, size_t start_idx, size_t height) {
    size_t n_points = verts_input.size();
    size_t dim = verts_input[0].size();
    std::vector<float> raw_data;
    raw_data.reserve(n_points * dim);
    for (const auto &v : verts_input) {
        raw_data.insert(raw_data.end(), v.begin(), v.end());
    }
    std::vector<size_t> sampled_point_indices(num_samples);
    c_bucket_fps_kdline(raw_data.data(), n_points, dim, num_samples, start_idx, height, sampled_point_indices.data());
    std::vector<std::vector<float>> sampled_points(num_samples);
    for (size_t i = 0; i < num_samples; i++) {
        sampled_points[i] = verts_input[sampled_point_indices[i]];
    }
    return sampled_points;
}