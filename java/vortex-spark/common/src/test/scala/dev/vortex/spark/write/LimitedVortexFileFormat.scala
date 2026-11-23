// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write

import org.apache.hadoop.fs.FileStatus
import org.apache.hadoop.mapreduce.{Job, TaskAttemptContext}

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.{
  FileFormat,
  OutputWriter,
  OutputWriterFactory
}
import org.apache.spark.sql.types.StructType

import dev.vortex.spark.VortexFileFormat

/** Vortex, with every writer handed to [[VortexOutputWriterAbortTest]]. In
  * Scala because FileFormat's Seq differs between Scala 2.12 and 2.13.
  */
final class LimitedVortexFileFormat extends FileFormat {
  private val delegate = new VortexFileFormat

  override def inferSchema(
      spark: SparkSession,
      options: Map[String, String],
      files: Seq[FileStatus]
  ): Option[StructType] = delegate.inferSchema(spark, options, files)

  override def prepareWrite(
      spark: SparkSession,
      job: Job,
      options: Map[String, String],
      dataSchema: StructType
  ): OutputWriterFactory = new LimitedOutputWriterFactory(
    delegate.prepareWrite(spark, job, options, dataSchema)
  )
}

final class LimitedOutputWriterFactory(delegate: OutputWriterFactory)
    extends OutputWriterFactory {
  override def getFileExtension(context: TaskAttemptContext): String =
    delegate.getFileExtension(context)

  override def newInstance(
      path: String,
      dataSchema: StructType,
      context: TaskAttemptContext
  ): OutputWriter = VortexOutputWriterAbortTest.opened(
    delegate
      .newInstance(path, dataSchema, context)
      .asInstanceOf[VortexOutputWriter]
  )
}
