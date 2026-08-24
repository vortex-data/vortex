// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import java.util.OptionalLong

import scala.collection.JavaConverters._

import org.apache.hadoop.fs.Path

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.FileSourceOptions
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.connector.expressions.aggregate.Aggregation
import org.apache.spark.sql.connector.read.{PartitionReaderFactory, Statistics}
import org.apache.spark.sql.execution.datasources.{
  AggregatePushDownUtils,
  PartitioningAwareFileIndex
}
import org.apache.spark.sql.execution.datasources.v2.FileScan
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types.StructType

import dev.vortex.spark.VortexOptions
import dev.vortex.spark.io.{VortexFile, VortexIo}

object VortexScan {

  /** Option turning footer-backed row-count statistics off. */
  val RowCountStatisticsOption = "vortex.stats.rowCount"
}

/** A Vortex scan planned by Spark's file source framework. */
final class VortexScan(
    override val sparkSession: SparkSession,
    override val fileIndex: PartitioningAwareFileIndex,
    override val dataSchema: StructType,
    override val readDataSchema: StructType,
    override val readPartitionSchema: StructType,
    val options: VortexOptions,
    val pushedFilters: Array[Filter],
    val pushedAggregation: Option[Aggregation],
    override val partitionFilters: Seq[Expression],
    override val dataFilters: Seq[Expression]
) extends FileScan {

  private val caseSensitiveOptions = options.asCaseSensitiveMap.asScala.toMap
  private val hadoopConf =
    sparkSession.sessionState.newHadoopConfWithOptions(caseSensitiveOptions)
  private val io = VortexIo.create(options, hadoopConf)

  override def isSplitable(path: Path): Boolean = false

  override def readSchema(): StructType =
    if (pushedAggregation.nonEmpty) readDataSchema else super.readSchema()

  override def createReaderFactory(): PartitionReaderFactory = {
    val fileOptions = new FileSourceOptions(caseSensitiveOptions)
    pushedAggregation match {
      case Some(aggregation) =>
        new VortexAggregateReaderFactory(
          fileOptions,
          io,
          options,
          readDataSchema,
          readPartitionSchema,
          aggregation
        )
      case None =>
        new VortexPartitionReaderFactory(
          fileOptions,
          io,
          options,
          dataSchema,
          readDataSchema,
          readPartitionSchema,
          pushedFilters
        )
    }
  }

  override def estimateStatistics(): Statistics = cachedStatistics

  private lazy val cachedStatistics: Statistics = {
    val base = super.estimateStatistics()
    val rows = pushedAggregation match {
      // The aggregate reader answers COUNT(*) from the same footers, and emits one row per file. Summing them
      // here as well would pay for every footer twice over -- once on the driver, once on the executors -- and
      // would describe rows the scan does not emit.
      case Some(_) => OptionalLong.of(partitions.map(_.files.length.toLong).sum)
      case None
          if options.getBoolean(VortexScan.RowCountStatisticsOption, true) =>
        // Count over the same file set the scan will read, so statistics and execution cannot
        // disagree about what belongs to the dataset.
        val files = partitions
          .flatMap(_.files)
          .map(file => new VortexFile(file.toPath.toString, file.fileSize))
        VortexFooterReader.sumRowCounts(files.asJava, io, options)
      case None => OptionalLong.empty()
    }
    new Statistics {
      override def sizeInBytes(): OptionalLong = base.sizeInBytes()
      override def numRows(): OptionalLong = rows
    }
  }

  override def equals(other: Any): Boolean = other match {
    case scan: VortexScan =>
      val aggregationsEqual =
        (pushedAggregation, scan.pushedAggregation) match {
          case (Some(left), Some(right)) =>
            AggregatePushDownUtils.equivalentAggregations(left, right)
          case (None, None) => true
          case _            => false
        }
      super.equals(
        scan
      ) && dataSchema == scan.dataSchema && options == scan.options &&
      equivalentFilters(pushedFilters, scan.pushedFilters) && aggregationsEqual
    case _ => false
  }

  override def hashCode(): Int = getClass.hashCode()

  override def getMetaData(): Map[String, String] = {
    val aggregation = pushedAggregation
      .map(value => seqToString(value.aggregateExpressions().toSeq))
      .getOrElse("[]")
    val groupBy = pushedAggregation
      .map(value => seqToString(value.groupByExpressions().toSeq))
      .getOrElse("[]")
    super.getMetaData() ++ Map(
      "PushedFilters" -> seqToString(pushedFilters.toSeq),
      "PushedAggregation" -> aggregation,
      "PushedGroupBy" -> groupBy
    )
  }
}
