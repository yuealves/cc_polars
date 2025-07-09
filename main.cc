#include <iostream>
#include <vector>
#include <memory>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <arrow/table.h>
#include <arrow/chunked_array.h>

// 1. Create Arrays - basic building blocks
arrow::Result<std::shared_ptr<arrow::Array>> CreateInt64Array() {
    arrow::Int64Builder builder;
    std::vector<int64_t> values = {1, 2, 3, 4, 5};
    ARROW_RETURN_NOT_OK(builder.AppendValues(values));
    
    std::shared_ptr<arrow::Array> array;
    ARROW_ASSIGN_OR_RAISE(array, builder.Finish());
    return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> CreateStringArray() {
    arrow::StringBuilder builder;
    std::vector<std::string> values = {"Alice", "Bob", "Charlie", "David", "Eve"};
    ARROW_RETURN_NOT_OK(builder.AppendValues(values));
    
    std::shared_ptr<arrow::Array> array;
    ARROW_ASSIGN_OR_RAISE(array, builder.Finish());
    return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> CreateDoubleArray() {
    arrow::DoubleBuilder builder;
    std::vector<double> values = {1.1, 2.2, 3.3, 4.4, 5.5};
    ARROW_RETURN_NOT_OK(builder.AppendValues(values));
    
    std::shared_ptr<arrow::Array> array;
    ARROW_ASSIGN_OR_RAISE(array, builder.Finish());
    return array;
}

// 2. Create ChunkedArrays - collections of arrays with same type
arrow::Result<std::shared_ptr<arrow::ChunkedArray>> CreateChunkedArray() {
    // Create multiple chunks of int64 arrays
    arrow::Int64Builder builder1, builder2;
    
    // First chunk
    std::vector<int64_t> chunk1_values = {10, 20, 30};
    ARROW_RETURN_NOT_OK(builder1.AppendValues(chunk1_values));
    std::shared_ptr<arrow::Array> chunk1;
    ARROW_ASSIGN_OR_RAISE(chunk1, builder1.Finish());
    
    // Second chunk
    std::vector<int64_t> chunk2_values = {40, 50, 60, 70};
    ARROW_RETURN_NOT_OK(builder2.AppendValues(chunk2_values));
    std::shared_ptr<arrow::Array> chunk2;
    ARROW_ASSIGN_OR_RAISE(chunk2, builder2.Finish());
    
    // Create ChunkedArray from multiple arrays
    arrow::ArrayVector chunks = {chunk1, chunk2};
    return arrow::ChunkedArray::Make(chunks);
}

// 3. Create RecordBatch - columnar data with schema
arrow::Result<std::shared_ptr<arrow::RecordBatch>> CreateRecordBatch() {
    // Create arrays for different columns
    ARROW_ASSIGN_OR_RAISE(auto id_array, CreateInt64Array());
    ARROW_ASSIGN_OR_RAISE(auto name_array, CreateStringArray());
    ARROW_ASSIGN_OR_RAISE(auto score_array, CreateDoubleArray());
    
    // Define schema
    std::vector<std::shared_ptr<arrow::Field>> fields = {
        arrow::field("id", arrow::int64()),
        arrow::field("name", arrow::utf8()),
        arrow::field("score", arrow::float64())
    };
    std::shared_ptr<arrow::Schema> schema = arrow::schema(fields);
    
    // Create RecordBatch
    std::vector<std::shared_ptr<arrow::Array>> arrays = {id_array, name_array, score_array};
    return arrow::RecordBatch::Make(schema, id_array->length(), arrays);
}

// 4. Create Table - collection of RecordBatches with same schema
arrow::Result<std::shared_ptr<arrow::Table>> CreateTable() {
    // Create multiple RecordBatches
    ARROW_ASSIGN_OR_RAISE(auto batch1, CreateRecordBatch());
    
    // Create second batch with different data
    arrow::Int64Builder id_builder;
    arrow::StringBuilder name_builder;
    arrow::DoubleBuilder score_builder;
    
    std::vector<int64_t> ids = {6, 7, 8};
    std::vector<std::string> names = {"Frank", "Grace", "Henry"};
    std::vector<double> scores = {6.6, 7.7, 8.8};
    
    ARROW_RETURN_NOT_OK(id_builder.AppendValues(ids));
    ARROW_RETURN_NOT_OK(name_builder.AppendValues(names));
    ARROW_RETURN_NOT_OK(score_builder.AppendValues(scores));
    
    std::shared_ptr<arrow::Array> id_array, name_array, score_array;
    ARROW_ASSIGN_OR_RAISE(id_array, id_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(name_array, name_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(score_array, score_builder.Finish());
    
    std::vector<std::shared_ptr<arrow::Array>> arrays = {id_array, name_array, score_array};
    auto batch2 = arrow::RecordBatch::Make(batch1->schema(), id_array->length(), arrays);
    
    // Create Table from RecordBatches
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches = {batch1, batch2};
    return arrow::Table::FromRecordBatches(batches);
}

void DemonstrateArrays() {
    std::cout << "\n=== 1. Arrays Demo ===" << std::endl;
    
    auto int_array_result = CreateInt64Array();
    if (!int_array_result.ok()) {
        std::cerr << "Failed to create int64 array" << std::endl;
        return;
    }
    auto int_array = *int_array_result;
    
    std::cout << "Int64 Array:" << std::endl;
    std::cout << "  Length: " << int_array->length() << std::endl;
    std::cout << "  Type: " << int_array->type()->ToString() << std::endl;
    std::cout << "  Values: ";
    for (int64_t i = 0; i < int_array->length(); ++i) {
        auto value = std::static_pointer_cast<arrow::Int64Array>(int_array)->Value(i);
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

void DemonstrateChunkedArrays() {
    std::cout << "\n=== 2. ChunkedArrays Demo ===" << std::endl;
    
    auto chunked_array_result = CreateChunkedArray();
    if (!chunked_array_result.ok()) {
        std::cerr << "Failed to create chunked array" << std::endl;
        return;
    }
    auto chunked_array = *chunked_array_result;
    
    std::cout << "ChunkedArray:" << std::endl;
    std::cout << "  Total length: " << chunked_array->length() << std::endl;
    std::cout << "  Number of chunks: " << chunked_array->num_chunks() << std::endl;
    std::cout << "  Type: " << chunked_array->type()->ToString() << std::endl;
    
    for (int i = 0; i < chunked_array->num_chunks(); ++i) {
        auto chunk = chunked_array->chunk(i);
        std::cout << "  Chunk " << i << " length: " << chunk->length() << std::endl;
    }
}

void DemonstrateRecordBatch() {
    std::cout << "\n=== 3. RecordBatch Demo ===" << std::endl;
    
    auto batch_result = CreateRecordBatch();
    if (!batch_result.ok()) {
        std::cerr << "Failed to create RecordBatch" << std::endl;
        return;
    }
    auto batch = *batch_result;
    
    std::cout << "RecordBatch:" << std::endl;
    std::cout << "  Number of rows: " << batch->num_rows() << std::endl;
    std::cout << "  Number of columns: " << batch->num_columns() << std::endl;
    std::cout << "  Schema: " << batch->schema()->ToString() << std::endl;
    
    // Access individual columns
    for (int i = 0; i < batch->num_columns(); ++i) {
        auto column = batch->column(i);
        auto field = batch->schema()->field(i);
        std::cout << "  Column '" << field->name() << "': " << column->length() << " values" << std::endl;
    }
}

void DemonstrateTable() {
    std::cout << "\n=== 4. Table Demo ===" << std::endl;
    
    auto table_result = CreateTable();
    if (!table_result.ok()) {
        std::cerr << "Failed to create Table" << std::endl;
        return;
    }
    auto table = *table_result;
    
    std::cout << "Table:" << std::endl;
    std::cout << "  Number of rows: " << table->num_rows() << std::endl;
    std::cout << "  Number of columns: " << table->num_columns() << std::endl;
    std::cout << "  Schema: " << table->schema()->ToString() << std::endl;
    
    // Access columns as ChunkedArrays
    for (int i = 0; i < table->num_columns(); ++i) {
        auto column = table->column(i);
        auto field = table->schema()->field(i);
        std::cout << "  Column '" << field->name() << "':" << std::endl;
        std::cout << "    Total length: " << column->length() << std::endl;
        std::cout << "    Number of chunks: " << column->num_chunks() << std::endl;
    }
    
    std::cout << "  Table demonstrates combining multiple RecordBatches" << std::endl;
}

int main() {
    std::cout << "Arrow C++ Data Structures Demo" << std::endl;
    std::cout << "==============================" << std::endl;
    
    try {
        DemonstrateArrays();
        DemonstrateChunkedArrays();
        DemonstrateRecordBatch();
        DemonstrateTable();
        
        std::cout << "\n=== Demo completed successfully! ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
