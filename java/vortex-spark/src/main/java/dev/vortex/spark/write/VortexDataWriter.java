// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.WriterCommitMessage;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Writes Spark InternalRow data to a Vortex file.
 * 
 * This is a simplified implementation that creates placeholder Vortex files.
 * In production, this would convert Spark's internal row format to Arrow vectors
 * and write them to a Vortex file using the Vortex writer API.
 */
public final class VortexDataWriter implements DataWriter<InternalRow> {
    
    private static final int DEFAULT_BATCH_SIZE = 4096;
    
    private final String filePath;
    private final StructType schema;
    private final CaseInsensitiveStringMap options;
    private final int batchSize;
    
    private final List<InternalRow> rows = new ArrayList<>();
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
     * Writes a single row to the Vortex file.
     * 
     * In this simplified implementation, rows are collected in memory.
     * In production, they would be converted to Arrow format and written in batches.
     *
     * @param row the row to write
     * @throws IOException if writing fails
     */
    @Override
    public void write(InternalRow row) throws IOException {
        // For now, just collect rows
        // In production, this would convert to Arrow and write batches
        rows.add(row.copy());
        recordCount++;
        bytesWritten += estimateRowSize();
    }
    
    /**
     * Estimates the size of a row in bytes.
     */
    private long estimateRowSize() {
        // Simple estimation based on schema
        return schema.fields().length * 8;
    }
    
    /**
     * Commits the write operation and returns a commit message.
     * 
     * This creates a placeholder Vortex file for now.
     * In production, this would flush Arrow batches and close the Vortex writer.
     *
     * @return a commit message with file information
     * @throws IOException if commit fails
     */
    @Override
    public WriterCommitMessage commit() throws IOException {
        if (!closed) {
            // Ensure parent directory exists
            Path path = Paths.get(filePath);
            Files.createDirectories(path.getParent());
            
            // Create a placeholder Vortex file
            // In production, this would write actual Vortex format data
            Files.write(path, new byte[0]);
            
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