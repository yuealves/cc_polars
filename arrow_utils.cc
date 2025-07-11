#include <pybind11/pybind11.h>
#include <arrow/python/pyarrow.h>
#include <arrow/api.h>
#include <arrow/builder.h>

#include <iostream>
#include <thread>
#include <future>
#include <vector>
#include <algorithm>

namespace py = pybind11;

// Helper function to extract depth features from a single RecordBatch
std::shared_ptr<arrow::RecordBatch> extract_depth_feature_from_batch(
    std::shared_ptr<arrow::RecordBatch> batch, 
    const std::vector<double>& depth_values) {
    
    int col_cnt = batch->num_columns();
    if (col_cnt % 2) {
        throw std::runtime_error("Input RecordBatch must have an even number of columns.");
    }

    int max_depth_level = col_cnt / 2;

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
    return arrow::RecordBatch::Make(new_schema, batch->num_rows(), result_arrays);
}

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

    // Call the helper function
    auto feature_batch = extract_depth_feature_from_batch(batch, depth_values);

    // Wrap and return
    PyObject* py_new_batch = arrow::py::wrap_batch(feature_batch);
    if (py_new_batch == NULL) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(py_new_batch);
}

py::object extract_depth_feature_from_arrow_table(py::object py_table, py::list py_depth_list, py::object py_max_threads = py::none()) {
    // Import pyarrow C++ API.
    if (arrow::py::import_pyarrow() < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Could not import pyarrow.");
        throw py::error_already_set();
    }

    // Unwrap Table
    auto result = arrow::py::unwrap_table(py_table.ptr());
    if (!result.ok()) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to unwrap Table.");
        throw py::error_already_set();
    }
    std::shared_ptr<arrow::Table> table = result.ValueOrDie();

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

    // First, collect all batches from the table
    arrow::TableBatchReader reader(*table);
    std::vector<std::shared_ptr<arrow::RecordBatch>> input_batches;
    std::shared_ptr<arrow::RecordBatch> batch;
    
    while (true) {
        auto status = reader.ReadNext(&batch);
        if (!status.ok()) {
            throw std::runtime_error("Failed to read batch: " + status.ToString());
        }
        if (!batch) {
            break; // End of batches
        }
        input_batches.push_back(batch);
    }

    if (input_batches.empty()) {
        throw std::runtime_error("Table contains no RecordBatches");
    }

    // Determine the number of threads to use
    int max_threads;
    if (py_max_threads.is_none()) {
        // Default: min(cpu_cores, num_batches)
        int cpu_cores = std::thread::hardware_concurrency();
        max_threads = std::min(cpu_cores, static_cast<int>(input_batches.size()));
    } else {
        max_threads = py_max_threads.cast<int>();
        if (max_threads <= 0) {
            throw std::runtime_error("max_threads must be positive");
        }
        // Limit to actual number of batches
        max_threads = std::min(max_threads, static_cast<int>(input_batches.size()));
    }

    // Process batches in parallel
    std::vector<std::shared_ptr<arrow::RecordBatch>> feature_batches(input_batches.size());
    std::vector<std::future<void>> futures;
    futures.reserve(max_threads);

    // Function to process a range of batches
    auto process_batch_range = [&](int start_idx, int end_idx) {
        for (int i = start_idx; i < end_idx; ++i) {
            try {
                feature_batches[i] = extract_depth_feature_from_batch(input_batches[i], depth_values);
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to process batch " + std::to_string(i) + ": " + e.what());
            }
        }
    };

    // Calculate batch ranges for each thread
    int batches_per_thread = input_batches.size() / max_threads;
    int remaining_batches = input_batches.size() % max_threads;

    int current_start = 0;
    for (int thread_id = 0; thread_id < max_threads; ++thread_id) {
        int current_end = current_start + batches_per_thread;
        if (thread_id < remaining_batches) {
            current_end++; // Distribute remaining batches to first few threads
        }
        
        if (current_start < current_end) {
            futures.push_back(std::async(std::launch::async, process_batch_range, current_start, current_end));
        }
        current_start = current_end;
    }

    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get(); // This will propagate any exceptions
    }

    // Combine all feature batches into a Table
    auto feature_table_result = arrow::Table::FromRecordBatches(feature_batches);
    if (!feature_table_result.ok()) {
        throw std::runtime_error("Failed to create Table from RecordBatches: " + feature_table_result.status().ToString());
    }
    std::shared_ptr<arrow::Table> feature_table = feature_table_result.ValueOrDie();

    // Wrap the Table back to a pyarrow.Table
    PyObject* py_feature_table = arrow::py::wrap_table(feature_table);
    if (py_feature_table == NULL) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(py_feature_table);
}

PYBIND11_MODULE(arrow_utils, m) {
    m.doc() = "A module for processing Arrow RecordBatches and Tables";
    m.def("extract_depth_feature", &extract_depth_feature, "Extract depth feature from data in a pyarrow.RecordBatch object",
          py::arg("py_batch"), py::arg("depth_list"));
    m.def("extract_depth_feature_from_arrow_table", &extract_depth_feature_from_arrow_table, 
          "Extract depth feature from data in a pyarrow.Table object",
          py::arg("py_table"), py::arg("depth_list"), py::arg("max_threads") = py::none());
}
