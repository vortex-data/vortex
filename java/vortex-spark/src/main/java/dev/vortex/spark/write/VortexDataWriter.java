// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.api.VortexWriter;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.ipc.ArrowStreamWriter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.WriterCommitMessage;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Writes Spark InternalRow data to a Vortex file.
 * 
 * This writer converts Spark's internal row format to Arrow vectors
 * and writes them to a Vortex file using the Vortex writer API.
 */
public final class VortexDataWriter implements DataWriter<InternalRow> {
    
    private static final int DEFAULT_BATCH_SIZE = 4096;
    
    private final String filePath;
    private final StructType schema;
    private final CaseInsensitiveStringMap options;
    private final int batchSize;
    
    private VortexWriter vortexWriter;
    private BufferAllocator allocator;
    private VectorSchemaRoot vectorSchemaRoot;
    private final List<InternalRow> batchRows = new ArrayList<>();
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
        
        try {
            // Initialize Arrow components
            this.allocator = new RootAllocator();
            
            // Convert Spark schema to Arrow schema
            var arrowSchema = SparkToArrowSchema.convert(schema);
            String schemaJson = arrowSchema.toJson();
            
            // Create Vortex writer
            Map<String, String> writerOptions = new HashMap<>();
            this.vortexWriter = VortexWriter.create(filePath, schemaJson, writerOptions);
            
            // Create VectorSchemaRoot for batching rows
            this.vectorSchemaRoot = VectorSchemaRoot.create(arrowSchema, allocator);
            
        } catch (IOException e) {
            throw new RuntimeException("Failed to initialize VortexDataWriter", e);
        }
    }
    
    
    /**
     * Writes a single row to the Vortex file.
     * 
     * Rows are batched and converted to Arrow format before writing.
     *
     * @param row the row to write
     * @throws IOException if writing fails
     */
    @Override
    public void write(InternalRow row) throws IOException {
        // Add row to current batch
        batchRows.add(row.copy());
        recordCount++;
        
        // Write batch if it's full
        if (batchRows.size() >= batchSize) {
            writeBatch();
        }
    }
    
    /**
     * Writes the current batch of rows to the Vortex file.
     */
    private void writeBatch() throws IOException {
        if (batchRows.isEmpty()) {
            return;
        }
        
        // For now, we'll write a simplified placeholder
        // TODO: Implement actual InternalRow to Arrow conversion
        // This would involve:
        // 1. Allocating vectors in VectorSchemaRoot
        // 2. Copying data from InternalRows to vectors
        // 3. Serializing to Arrow IPC format
        // 4. Writing to Vortex
        
        // Placeholder: Create minimal Arrow batch
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            // Write empty Arrow batch for now
            vectorSchemaRoot.allocateNew();
            vectorSchemaRoot.setRowCount(batchRows.size());
            
            try (ArrowStreamWriter writer = new ArrowStreamWriter(
                    vectorSchemaRoot, null, Channels.newChannel(baos))) {
                writer.start();
                writer.writeBatch();
            }
            
            byte[] arrowData = baos.toByteArray();
            vortexWriter.writeBatch(arrowData);
            bytesWritten += arrowData.length;
            
            vectorSchemaRoot.clear();
            batchRows.clear();
        }
    }
    
    /**
     * Commits the write operation and returns a commit message.
     * 
     * This flushes any remaining rows and closes the Vortex writer.
     *
     * @return a commit message with file information
     * @throws IOException if commit fails
     */
    @Override
    public WriterCommitMessage commit() throws IOException {
        if (!closed) {
            // Write any remaining rows
            if (!batchRows.isEmpty()) {
                writeBatch();
            }
            
            // Close the Vortex writer to finalize the file
            if (vortexWriter != null) {
                vortexWriter.close();
                vortexWriter = null;
            }
            
            // Clean up Arrow resources
            if (vectorSchemaRoot != null) {
                vectorSchemaRoot.close();
                vectorSchemaRoot = null;
            }
            
            if (allocator != null) {
                allocator.close();
                allocator = null;
            }
            
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
            // Close resources
            try {
                if (vortexWriter != null) {
                    vortexWriter.close();
                }
            } catch (Exception e) {
                // Log but don't throw
                System.err.println("Error closing writer during abort: " + e.getMessage());
            }
            
            if (vectorSchemaRoot != null) {
                vectorSchemaRoot.close();
            }
            
            if (allocator != null) {
                allocator.close();
            }
            
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
}