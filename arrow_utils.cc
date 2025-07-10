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

    if (py_depth_list.empty()) {
        throw std::runtime_error("py_depth_list can't be empty");
    }

    // Convert py::list to std::vector<double>
    std::vector<double> depth_values;
    depth_values.reserve(py_depth_list.size());
    for (auto item : py_depth_list) {
        depth_values.push_back(item.cast<double>());
    }

    // Validate that depth_values are sorted in ascending order
    for (size_t i = 1; i < depth_values.size(); ++i) {
        if (depth_values[i] < depth_values[i - 1]) {
            throw std::runtime_error("depth_list must be sorted in ascending order");
        }
    }

    // 1. Create intermediate storage: a vector of vectors for each depth feature
    std::vector<std::vector<double>> results(depth_values.size());
    for (auto& vec : results) {
        vec.resize(batch->num_rows());
    }

    std::vector<std::shared_ptr<arrow::DoubleArray>> double_columns;
    double_columns.reserve(col_cnt);
    for (int i = 0; i < col_cnt; ++i) {
        double_columns.push_back(std::static_pointer_cast<arrow::DoubleArray>(batch->column(i)));
    }

    // 2. Optimized calculation loop, storing results in vectors
    for (int64_t i = 0; i < batch->num_rows(); ++i) {
        size_t depth_idx = 0;
        double cumulative_subtrahend = 0.0;

        for (int j = 0; j < max_depth_level; ++j) {
            cumulative_subtrahend += (double_columns[j]->Value(i) * double_columns[j + max_depth_level]->Value(i));
            
            // Since depth_values is sorted, check which depths have now been resolved
            while (depth_idx < depth_values.size() && depth_values[depth_idx] < cumulative_subtrahend) {
                results[depth_idx][i] = std::abs(double_columns[j]->Value(i) - double_columns[0]->Value(0)) / double_columns[0]->Value(0);
                depth_idx++;
            }
            if (depth_idx == depth_values.size()) {
                break;
            }
        }

        // Any remaining depth values (those larger than the total subtrahend)
        // get the feature value from the last depth level
        double last_level_value = double_columns[max_depth_level - 1]->Value(i);
        while (depth_idx < depth_values.size()) {
            results[depth_idx][i] = last_level_value;
            depth_idx++;
        }
    }

    // 3. Build all arrays from the intermediate vectors
    std::vector<std::shared_ptr<arrow::Array>> result_arrays;
    std::vector<std::shared_ptr<arrow::Field>> result_fields;
    result_arrays.reserve(depth_values.size());
    result_fields.reserve(depth_values.size());

    for (size_t i = 0; i < results.size(); ++i) {
        arrow::DoubleBuilder builder;
        arrow::Status st = builder.AppendValues(results[i]);
        if (!st.ok()) {
            throw std::runtime_error("Failed to append values for feature " + std::to_string(i));
        }
        
        std::shared_ptr<arrow::Array> array;
        st = builder.Finish(&array);
        if (!st.ok()) {
            throw std::runtime_error("Failed to finalize builder for feature " + std::to_string(i));
        }
        result_arrays.push_back(array);
        result_fields.push_back(
            arrow::field("feature_depth_" + std::to_string(i), arrow::float64()));
    }

    auto new_schema = arrow::schema(result_fields);

    // 4. Create and return the RecordBatch
    auto new_batch = arrow::RecordBatch::Make(new_schema, batch->num_rows(), result_arrays);

    PyObject* py_new_batch = arrow::py::wrap_batch(new_batch);
    if (py_new_batch == NULL) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(py_new_batch);
}

PYBIND11_MODULE(arrow_utils, m) {
    m.doc() = "A module for processing Arrow RecordBatches";
    m.def("extract_depth_feature", &extract_depth_feature, "Extract depth feature from data in a pyarrow.RecordBatch object",
          py::arg("py_batch"), py::arg("depth_list"));
}
