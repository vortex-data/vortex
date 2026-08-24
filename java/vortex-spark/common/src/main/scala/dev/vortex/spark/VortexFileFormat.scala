// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark

import scala.collection.JavaConverters._

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileStatus, Path}
import org.apache.hadoop.mapreduce.Job

import org.apache.spark.TaskContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.vectorized.ConstantColumnVector
import org.apache.spark.sql.execution.datasources.{
  FileFormat,
  OutputWriterFactory,
  PartitionedFile
}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.sources.{DataSourceRegister, Filter}
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch

import dev.vortex.spark.VortexOptions
import dev.vortex.spark.io.VortexIo
import dev.vortex.spark.read.{
  VortexArrowColumnVector,
  VortexFooterReader,
  VortexPartitionReader
}
import dev.vortex.spark.write.{SparkToArrowSchema, VortexOutputWriterFactory}

/** V1 fallback used for catalog tables and when Vortex is listed in
  * useV1SourceList.
  */
final class VortexFileFormat
    extends FileFormat
    with DataSourceRegister
    with Serializable {
  override def shortName(): String = "vortex"

  override def toString: String = "Vortex"

  override def equals(other: Any): Boolean =
    other.isInstanceOf[VortexFileFormat]

  override def hashCode(): Int = getClass.hashCode()

  override def inferSchema(
      spark: SparkSession,
      options: Map[String, String],
      files: Seq[FileStatus]
  ): Option[StructType] = {
    val hadoopConf = spark.sessionState.newHadoopConfWithOptions(options)
    val vortexOptions = VortexOptions.of(options.asJava)
    Option(
      VortexFooterReader.inferSchema(
        files.asJava,
        VortexIo.create(vortexOptions, hadoopConf),
        vortexOptions
      )
    )
  }

  override def isSplitable(
      sparkSession: SparkSession,
      options: Map[String, String],
      path: Path
  ): Boolean = false

  override def supportBatch(
      sparkSession: SparkSession,
      dataSchema: StructType
  ): Boolean = true

  override def vectorTypes(
      requiredSchema: StructType,
      partitionSchema: StructType,
      sqlConf: SQLConf
  ): Option[Seq[String]] =
    Some(
      Seq.fill(requiredSchema.length)(
        classOf[VortexArrowColumnVector].getName
      ) ++
        Seq.fill(partitionSchema.length)(classOf[ConstantColumnVector].getName)
    )

  override def prepareWrite(
      spark: SparkSession,
      job: Job,
      options: Map[String, String],
      dataSchema: StructType
  ): OutputWriterFactory =
    new VortexOutputWriterFactory(dataSchema, VortexOptions.of(options.asJava))

  override def buildReader(
      spark: SparkSession,
      dataSchema: StructType,
      partitionSchema: StructType,
      requiredSchema: StructType,
      filters: Seq[Filter],
      options: Map[String, String],
      hadoopConf: Configuration
  ): PartitionedFile => Iterator[InternalRow] = {
    val optionMap = VortexOptions.of(options.asJava)
    val io = VortexIo.create(optionMap, hadoopConf)
    (file: PartitionedFile) => {
      val reader = new VortexPartitionReader(
        file,
        dataSchema,
        requiredSchema,
        new StructType(),
        io,
        optionMap,
        filters.toArray
      )
      Option(TaskContext.get())
        .foreach(_.addTaskCompletionListener[Unit](_ => reader.close()))
      // Reached whenever Spark asks for rows rather than batches, which `FileSourceScanExec` does when
      // whole-stage codegen is off or the schema has more fields than `spark.sql.codegen.maxFields`.
      new Iterator[InternalRow] {
        private var batch = Option.empty[ColumnarBatch]
        private var rowIndex = 0

        override def hasNext: Boolean = {
          // The reader owns each batch: `next()` releases the one before it, and `close()` releases the last.
          while (batch.forall(rowIndex >= _.numRows()) && reader.next()) {
            batch = Some(reader.get())
            rowIndex = 0
          }
          if (batch.exists(rowIndex < _.numRows())) {
            true
          } else {
            batch = None
            reader.close()
            false
          }
        }

        override def next(): InternalRow = {
          if (!hasNext) {
            throw new NoSuchElementException("end of Vortex file")
          }
          val row = batch.get.getRow(rowIndex)
          rowIndex += 1
          row
        }
      }
    }
  }

  override def buildReaderWithPartitionValues(
      spark: SparkSession,
      dataSchema: StructType,
      partitionSchema: StructType,
      requiredSchema: StructType,
      filters: Seq[Filter],
      options: Map[String, String],
      hadoopConf: Configuration
  ): PartitionedFile => Iterator[InternalRow] = {
    if (
      !options.getOrElse(FileFormat.OPTION_RETURNING_BATCH, "false").toBoolean
    ) {
      return super.buildReaderWithPartitionValues(
        spark,
        dataSchema,
        partitionSchema,
        requiredSchema,
        filters,
        options,
        hadoopConf
      )
    }

    val optionMap = VortexOptions.of(options.asJava)
    val io = VortexIo.create(optionMap, hadoopConf)
    (file: PartitionedFile) => {
      val reader = new VortexPartitionReader(
        file,
        dataSchema,
        requiredSchema,
        partitionSchema,
        io,
        optionMap,
        filters.toArray
      )
      Option(TaskContext.get()).foreach(
        _.addTaskCompletionListener[Unit](_ => reader.close())
      )
      new Iterator[ColumnarBatch] {
        private var loaded = false
        private var exhausted = false

        override def hasNext: Boolean = {
          if (!loaded && !exhausted) {
            loaded = reader.next()
            if (!loaded) {
              exhausted = true
              reader.close()
            }
          }
          loaded
        }

        override def next(): ColumnarBatch = {
          if (!hasNext) {
            throw new NoSuchElementException("end of Vortex file")
          }
          loaded = false
          reader.get()
        }
      }.asInstanceOf[Iterator[InternalRow]]
    }
  }

  override def supportDataType(dataType: DataType): Boolean =
    SparkToArrowSchema.supportsDataType(dataType)
}
