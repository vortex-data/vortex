// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write

import scala.collection.JavaConverters._

import org.apache.hadoop.mapreduce.Job

import org.apache.spark.sql.connector.write.LogicalWriteInfo
import org.apache.spark.sql.execution.datasources.OutputWriterFactory
import org.apache.spark.sql.execution.datasources.v2.FileWrite
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{DataType, StructType}

import dev.vortex.spark.VortexOptions

/** Vortex write support using Spark's file commit protocol. */
final case class VortexWrite(
    paths: Seq[String],
    formatName: String,
    supportsDataType: DataType => Boolean,
    info: LogicalWriteInfo
) extends FileWrite {

  override def prepareWrite(
      sqlConf: SQLConf,
      job: Job,
      options: Map[String, String],
      dataSchema: StructType
  ): OutputWriterFactory =
    new VortexOutputWriterFactory(dataSchema, VortexOptions.of(options.asJava))
}
