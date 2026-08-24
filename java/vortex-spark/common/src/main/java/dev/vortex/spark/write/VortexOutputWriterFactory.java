// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.spark.VortexOptions;
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
        this.schema = schema;
        this.options = options;
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
