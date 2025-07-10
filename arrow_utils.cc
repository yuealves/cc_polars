#include <pybind11/pybind11.h>
#include <arrow/python/pyarrow.h>
#include <arrow/api.h>
#include <arrow/builder.h>

#include <iostream>

namespace py = pybind11;
py::object extract_depth_feature(py::object py_batch, py::list py_depth_list) {
    // Import pyarrow C++ API.
    if (arrow::py::import_pyarrow() < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Could not import pyarrow.");
        throw py::error_already_set();
    }

    // Unwrap RecordBatch
    auto result = arrow::py::unwrap_batch(py_batch.ptr());
    if (!result.ok()) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to unwrap RecordBatch.");
        throw py::error_already_set();
    }
    std::shared_ptr<arrow::RecordBatch> batch = result.ValueOrDie();

    int col_cnt = batch->num_columns();
    if (col_cnt % 2) {
        throw std::runtime_error("Input RecordBatch must have an even number of columns.");
    }

    int max_depth_level = col_cnt / 2;

    if (static_cast<int64_t>(py_depth_list.size()) < 1) {
        throw std::runtime_error("py_depth_list can't be empty");
    }

    std::vector<double> result_vec;
    result_vec.resize(batch->num_rows());
  
    // Convert py::list to std::vector<double>
    std::vector<double> depth_list;
    depth_list.reserve(py_depth_list.size());
    for (auto item : py_depth_list) {
        depth_list.push_back(item.cast<double>());
    }


    std::vector<std::shared_ptr<arrow::DoubleArray>> double_columns;
    for (int i = 0; i < col_cnt; ++i) {
        double_columns.push_back(std::static_pointer_cast<arrow::DoubleArray>(batch->column(i)));
    }

    // calculate the depth feature
    for (int i = 0; i < batch->num_rows(); ++i) {
      double depth = depth_list[0];
      int depth_level = max_depth_level - 1;
      for (int j = 0; j < max_depth_level; j++) {
        depth -= (double_columns[j]->Value(i) * double_columns[j+max_depth_level]->Value(i));
        if (depth < 0) {
          depth_level = j;
          break;
        }
      }
      result_vec[i] = double_columns[depth_level]->Value(i);
    }

    arrow::DoubleBuilder result_builder;
    arrow::Status st = result_builder.AppendValues(result_vec);
    if (!st.ok()) {
        throw std::runtime_error("Failed to append result_vec values: " + st.ToString());
    }

    std::shared_ptr<arrow::Array> result_array;
    st = result_builder.Finish(&result_array);
    if (!st.ok()) {
        throw std::runtime_error("Failed to finalize builder: " + st.ToString());
    }

    // Wrap the arrow::Array back to a pyarrow.Array
    PyObject* py_result_array = arrow::py::wrap_array(result_array);
    if (py_result_array == NULL) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(py_result_array);
}

PYBIND11_MODULE(arrow_utils, m) {
    m.doc() = "A module for processing Arrow RecordBatches";
    m.def("extract_depth_feature", &extract_depth_feature, "Extract depth feature from data in a pyarrow.RecordBatch object",
          py::arg("py_batch"), py::arg("depth_list"));
}
