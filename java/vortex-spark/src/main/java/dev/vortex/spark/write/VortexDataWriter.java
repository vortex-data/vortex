// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.api.VortexWriter;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.*;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Schema;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.WriterCommitMessage;
import org.apache.spark.sql.execution.arrow.ArrowWriter;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.apache.spark.sql.vectorized.ArrowColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * Writes Spark InternalRow data to a Vortex file.
 * 
 * This writer converts Spark's internal row format to Arrow vectors,
 * then writes them to a Vortex file using the Vortex writer API.
 */
public final class VortexDataWriter implements DataWriter<InternalRow> {
    
    private static final int DEFAULT_BATCH_SIZE = 4096;
    
    private final String filePath;
    private final StructType schema;
    private final CaseInsensitiveStringMap options;
    private final int batchSize;
    
    private VortexWriter writer;
    private BufferAllocator allocator;
    private VectorSchemaRoot vectorSchemaRoot;
    private ArrowWriter arrowWriter;
    private long recordCount = 0;
    private long bytesWritten = 0;
    private boolean closed = false;
    
    /**
     * Creates a new VortexDataWriter.
     *
     * @param filePath the path where the Vortex file will be written
     * @param schema the schema of the data to write
     * @param options additional write options
     */
    public VortexDataWriter(
            String filePath,
            StructType schema,
            CaseInsensitiveStringMap options) {
        this.filePath = filePath;
        this.schema = schema;
        this.options = options;
        this.batchSize = options.getInt("batch.size", DEFAULT_BATCH_SIZE);
    }
    
    /**
     * Initializes the writer resources.
     * 
     * This creates the Arrow allocator, schema, and Vortex writer.
     */
    private void initialize() throws IOException {
        // Ensure parent directory exists
        Path path = Paths.get(filePath);
        Files.createDirectories(path.getParent());
        
        // Create Arrow allocator and schema
        allocator = new RootAllocator(Long.MAX_VALUE);
        Schema arrowSchema = SparkToArrowSchema.convert(schema);
        vectorSchemaRoot = VectorSchemaRoot.create(arrowSchema, allocator);
        
        // Create Arrow writer for converting InternalRows
        arrowWriter = ArrowWriter.create(vectorSchemaRoot);
        
        // Create Vortex writer
        writer = VortexWriter.create(filePath, arrowSchema, options.asCaseSensitiveMap());
    }
    
    /**
     * Writes a single row to the Vortex file.
     * 
     * Rows are buffered in Arrow vectors and written in batches
     * for efficiency.
     *
     * @param row the row to write
     * @throws IOException if writing fails
     */
    @Override
    public void write(InternalRow row) throws IOException {
        if (writer == null) {
            initialize();
        }
        
        // Write row to Arrow vectors
        arrowWriter.write(row);
        recordCount++;
        
        // If batch is full, flush to Vortex
        if (arrowWriter.getBatchSize() >= batchSize) {
            flush();
        }
    }
    
    /**
     * Flushes buffered data to the Vortex file.
     */
    private void flush() throws IOException {
        if (arrowWriter.getBatchSize() > 0) {
            arrowWriter.finish();
            
            // Write the batch to Vortex
            writer.writeBatch(vectorSchemaRoot);
            bytesWritten += vectorSchemaRoot.getRowCount() * estimateRowSize();
            
            // Reset for next batch
            arrowWriter.reset();
        }
    }
    
    /**
     * Estimates the size of a row in bytes.
     */
    private long estimateRowSize() {
        // Simple estimation based on schema
        // In production, this should be more sophisticated
        return schema.fields().length * 8;
    }
    
    /**
     * Commits the write operation and returns a commit message.
     * 
     * This flushes any remaining data and closes the writer.
     *
     * @return a commit message with file information
     * @throws IOException if commit fails
     */
    @Override
    public WriterCommitMessage commit() throws IOException {
        if (!closed) {
            if (writer != null) {
                flush();
                writer.close();
            }
            cleanup();
            closed = true;
        }
        
        return new VortexWriterCommitMessage(filePath, recordCount, bytesWritten);
    }
    
    /**
     * Aborts the write operation and cleans up resources.
     * 
     * This deletes any partially written file.
     *
     * @throws IOException if abort fails
     */
    @Override
    public void abort() throws IOException {
        if (!closed) {
            cleanup();
            
            // Delete the partial file if it exists
            try {
                Files.deleteIfExists(Paths.get(filePath));
            } catch (IOException e) {
                // Log but don't throw - we're already aborting
                System.err.println("Failed to delete partial file: " + filePath);
            }
            
            closed = true;
        }
    }
    
    /**
     * Closes the writer and releases resources.
     */
    @Override
    public void close() throws IOException {
        if (!closed) {
            commit();
        }
    }
    
    /**
     * Cleans up Arrow resources.
     */
    private void cleanup() {
        if (vectorSchemaRoot != null) {
            vectorSchemaRoot.close();
            vectorSchemaRoot = null;
        }
        if (allocator != null) {
            allocator.close();
            allocator = null;
        }
        arrowWriter = null;
        writer = null;
    }
}