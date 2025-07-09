#include <pybind11/pybind11.h>
#include <arrow/python/pyarrow.h>
#include <arrow/api.h>
#include <arrow/builder.h>

#include <iostream>

namespace py = pybind11;

// This function takes a pyarrow.RecordBatch, counts positive odd numbers
// in the first 4 columns for each row, and returns a new RecordBatch
// with the counts in a new 5th column.
py::object process_record_batch(py::object py_batch) {
    // Import pyarrow C++ API. Required for all pyarrow C++ functions.
    if (arrow::py::import_pyarrow() < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Could not import pyarrow.");
        throw py::error_already_set();
    }

    // Convert pyarrow.RecordBatch to arrow::RecordBatch
    auto result = arrow::py::unwrap_batch(py_batch.ptr());
    if (!result.ok()) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to unwrap RecordBatch.");
        throw py::error_already_set();
    }
    std::shared_ptr<arrow::RecordBatch> batch = result.ValueOrDie();

    if (batch->num_columns() != 4) {
        throw std::runtime_error("Input RecordBatch must have exactly 4 columns.");
    }

    arrow::Int64Builder count_builder;
    arrow::Status st = count_builder.Reserve(batch->num_rows());
    if (!st.ok()) {
        throw std::runtime_error("Failed to reserve memory for builder: " + st.ToString());
    }

    std::vector<std::shared_ptr<arrow::Array>> original_columns;
    for (int i = 0; i < 4; ++i) {
        original_columns.push_back(batch->column(i));
    }

    for (int64_t i = 0; i < batch->num_rows(); ++i) {
        int64_t odd_count = 0;
        for (int j = 0; j < 4; ++j) {
            // Assuming columns are Int64Array. Add checks/casting for other types if needed.
            auto col = std::static_pointer_cast<arrow::Int64Array>(original_columns[j]);
            if (!col->IsNull(i)) {
                int64_t value = col->Value(i);
                if (value > 0 && value % 2 != 0) {
                    odd_count++;
                }
            }
        }
        st = count_builder.Append(odd_count);
        if (!st.ok()) {
            throw std::runtime_error("Failed to append data to builder: " + st.ToString());
        }
    }

    std::shared_ptr<arrow::Array> count_array;
    st = count_builder.Finish(&count_array);
    if (!st.ok()) {
        throw std::runtime_error("Failed to finalize builder: " + st.ToString());
    }

    // The output is a single array (which will be a chunk in the ChunkedArray).
    auto chunked_array = std::make_shared<arrow::ChunkedArray>(count_array);

    // Wrap the new ChunkedArray back to a pyarrow.ChunkedArray
    PyObject* py_new_array = arrow::py::wrap_chunked_array(chunked_array);
    if (py_new_array == NULL) {
        // The error is already set by wrap_chunked_array in case of failure
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(py_new_array);
}

PYBIND11_MODULE(arrow_utils, m) {
    m.doc() = "A module for processing Arrow RecordBatches";
    m.def("process_record_batch", &process_record_batch, "Processes a 4-column RecordBatch and returns a ChunkedArray with counts of positive odd numbers.");
}
