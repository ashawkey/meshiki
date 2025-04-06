#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <meshiki/mesh.h>

namespace py = pybind11;

PYBIND11_MODULE(_meshiki, m) {
    // Mesh
    py::class_<Mesh>(m, "Mesh")
        .def(py::init<std::vector<std::vector<float>>, std::vector<std::vector<int>>, bool, bool>())
        .def_readwrite("num_verts", &Mesh::num_verts)
        .def_readwrite("num_edges", &Mesh::num_edges)
        .def_readwrite("num_faces", &Mesh::num_faces)
        .def_readwrite("num_quad", &Mesh::num_quad)
        .def_readwrite("num_components", &Mesh::num_components)
        .def_readwrite("rect_error", &Mesh::rect_error)
        .def("quadrangulate", &Mesh::quadrangulate, py::arg("thresh_bihedral"), py::arg("thresh_convex"))
        .def("polygonize", &Mesh::polygonize, py::arg("thresh_bihedral"), py::arg("thresh_convex"), py::arg("max_round"))
        .def("salient_point_sample", &Mesh::salient_point_sample, py::arg("num_samples"), py::arg("thresh_angle"))
        .def("uniform_point_sample", &Mesh::uniform_point_sample, py::arg("num_samples"))
        .def("repair_face_orientation", &Mesh::repair_face_orientation)
        .def("smart_group_components", &Mesh::smart_group_components)
        .def("explode", &Mesh::explode, py::arg("delta"))
        .def("export_components", &Mesh::export_components)
        .def("export_mesh", &Mesh::export_mesh);
    
    // triangulate
    m.def("triangulate", &triangulate);

    // pad_to_quad
    m.def("pad_to_quad", &pad_to_quad);

    // merge close vertices
    m.def("merge_close_vertices", &merge_close_vertices);

    // furthest point sampling
    m.def("fps", &fps);
    m.def("bucket_fps_kdtree", &bucket_fps_kdtree);
    m.def("bucket_fps_kdline", &bucket_fps_kdline);
}