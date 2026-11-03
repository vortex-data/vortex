// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import dev.vortex.spark.io.HadoopWritable;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.spark.sql.execution.datasources.OutputWriter;
import org.apache.spark.sql.execution.datasources.OutputWriterFactory;
import org.apache.spark.sql.types.StructType;

/** Creates Vortex output writers at paths assigned by Spark's file commit protocol. */
public final class VortexOutputWriterFactory extends OutputWriterFactory {
    private static final long serialVersionUID = 1L;

    private final StructType schema;
    private final VortexOptions options;

    public VortexOutputWriterFactory(StructType schema, VortexOptions options) {
        validate(options);
        this.schema = schema;
        this.options = options;
    }

    /**
     * Rejects options every writer would fail on. Spark builds the factory on the driver before it starts a task, so a
     * bad value fails the write there instead of in each task.
     *
     * @throws IllegalArgumentException if a write option is unusable
     */
    public static void validate(VortexOptions options) {
        VortexOutputWriter.configuredBatchSize(options);
        VortexSparkSession.validate(options);
    }

    @Override
    public String getFileExtension(TaskAttemptContext context) {
        return ".vortex";
    }

    @Override
    public OutputWriter newInstance(String path, StructType dataSchema, TaskAttemptContext context) {
        return new VortexOutputWriter(path, schema, options, HadoopWritable.create(context.getConfiguration(), path));
    }
}
