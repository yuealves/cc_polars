#include <iostream>
#include <vector>
#include <memory>

#include <arrow/api.h>
#include <arrow/builder.h>

// Function to create a simple RecordBatch
arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreateRecordBatch() {
    // Create a builder for an Int64 array
    arrow::Int64Builder builder;
    std::vector<int64_t> values = {1, 2, 3, 4, 5};
    ARROW_RETURN_NOT_OK(builder.AppendValues(values));

    // Finish the array
    std::shared_ptr<arrow::Array> array;
    ARROW_ASSIGN_OR_RAISE(array, builder.Finish());

    // Define the schema for the RecordBatch
    std::shared_ptr<arrow::Field> field = arrow::field("my_field", arrow::int64());
    std::shared_ptr<arrow::Schema> schema = arrow::schema({field});

    // Create the RecordBatch
    return arrow::RecordBatch::Make(schema, array->length(), {array});
}

int main() {
    // Create a RecordBatch
    auto result = CreateRecordBatch();
    if (!result.ok()) {
        std::cerr << "Failed to create RecordBatch: " << result.status().ToString() << std::endl;
        return 1;
    }
    std::shared_ptr<arrow::RecordBatch> record_batch = *result;

    std::cout << "Successfully created a RecordBatch!" << std::endl;
    std::cout << "Number of columns: " << record_batch->num_columns() << std::endl;
    std::cout << "Number of rows: " << record_batch->num_rows() << std::endl;
    std::cout << "Schema:" << std::endl << record_batch->schema()->ToString() << std::endl;

    return 0;
}
