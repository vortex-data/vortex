// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import java.io.Serializable;
import org.apache.spark.sql.connector.write.WriterCommitMessage;

/**
 * Commit message containing information about a successfully written Vortex file.
 *
 * <p>This message is passed from executors back to the driver to coordinate the commit phase of the write operation.
 */
public record VortexWriterCommitMessage(String filePath, long recordCount, long bytesWritten)
        implements WriterCommitMessage, Serializable {

    /**
     * Creates a new commit message for a written Vortex file.
     *
     * @param filePath the path to the written file
     * @param recordCount the number of records written
     * @param bytesWritten the number of bytes written
     */
    public VortexWriterCommitMessage {}

    /**
     * Gets the path to the written Vortex file.
     *
     * @return the file path
     */
    @Override
    public String filePath() {
        return filePath;
    }

    /**
     * Gets the number of records written to the file.
     *
     * @return the record count
     */
    @Override
    public long recordCount() {
        return recordCount;
    }

    /**
     * Gets the number of bytes written to the file.
     *
     * @return the byte count
     */
    @Override
    public long bytesWritten() {
        return bytesWritten;
    }
}
