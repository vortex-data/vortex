// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark

import scala.collection.JavaConverters._

import org.apache.hadoop.fs.FileStatus

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.connector.write.{
  LogicalWriteInfo,
  Write,
  WriteBuilder
}
import org.apache.spark.sql.execution.datasources.FileFormat
import org.apache.spark.sql.execution.datasources.v2.FileTable
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.sql.util.CaseInsensitiveStringMap

import dev.vortex.spark.VortexOptions
import dev.vortex.spark.io.VortexIo
import dev.vortex.spark.read.{VortexFooterReader, VortexScanBuilder}
import dev.vortex.spark.write.{SparkToArrowSchema, VortexWrite}

/** A Vortex table backed by Spark's file index and commit protocol. */
final class VortexTable(
    tableName: String,
    sparkSession: SparkSession,
    options: CaseInsensitiveStringMap,
    paths: Seq[String],
    userSpecifiedSchema: Option[StructType],
    override val fallbackFileFormat: Class[_ <: FileFormat]
) extends FileTable(sparkSession, options, paths, userSpecifiedSchema) {

  override def name(): String = tableName

  override def newScanBuilder(
      operationOptions: CaseInsensitiveStringMap
  ): VortexScanBuilder =
    new VortexScanBuilder(
      sparkSession,
      fileIndex,
      schema,
      dataSchema,
      VortexOptions.of(mergeOptions(operationOptions).asCaseSensitiveMap)
    )

  override def inferSchema(files: Seq[FileStatus]): Option[StructType] = {
    val vortexOptions = VortexOptions.of(options.asCaseSensitiveMap)
    val hadoopConf = sparkSession.sessionState.newHadoopConfWithOptions(
      vortexOptions.asCaseSensitiveMap.asScala.toMap
    )
    Option(
      VortexFooterReader.inferSchema(
        files.asJava,
        VortexIo.create(vortexOptions, hadoopConf),
        vortexOptions
      )
    )
  }

  override def newWriteBuilder(info: LogicalWriteInfo): WriteBuilder = {
    val mergedInfo = new LogicalWriteInfo {
      override def queryId(): String = info.queryId()
      override def schema(): StructType = info.schema()
      override def options(): CaseInsensitiveStringMap = mergeOptions(
        info.options()
      )
    }
    new WriteBuilder {
      override def build(): Write =
        VortexWrite(paths, formatName, supportsDataType, mergedInfo)
    }
  }

  override def supportsDataType(dataType: DataType): Boolean =
    SparkToArrowSchema.supportsDataType(dataType)

  override def formatName: String = "VORTEX"

  private def mergeOptions(
      operationOptions: CaseInsensitiveStringMap
  ): CaseInsensitiveStringMap = {
    val merged =
      options.asCaseSensitiveMap.asScala ++ operationOptions.asCaseSensitiveMap.asScala
    new CaseInsensitiveStringMap(merged.asJava)
  }
}
