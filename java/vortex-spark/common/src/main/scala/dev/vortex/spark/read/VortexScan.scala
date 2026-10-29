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

  /** Option turning splitting of large files into row ranges off. */
  val SplitFilesOption = "vortex.splitFiles"

  /** Whether Spark may split `path` into byte ranges, which the reader maps to
    * row ranges.
    */
  def isSplitable(options: VortexOptions, path: Path): Boolean =
    options.getBoolean(SplitFilesOption, true) &&
      VortexFile.hasVortexExtension(path.getName)
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
  private val caseSensitive =
    sparkSession.sessionState.conf.caseSensitiveAnalysis

  // A pushed COUNT(*) reads only the footer, so splitting would just open it once per split.
  override def isSplitable(path: Path): Boolean =
    pushedAggregation.isEmpty && VortexScan.isSplitable(options, path)

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
          pushedFilters,
          caseSensitive
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
      case Some(_) => OptionalLong.of(vortexFiles.length.toLong)
      case None
          if options.getBoolean(VortexScan.RowCountStatisticsOption, true) =>
        // Count over the same file set the scan will read, so statistics and execution cannot
        // disagree about what belongs to the dataset.
        val files = vortexFiles
          .map(file => new VortexFile(file.toPath.toString, file.fileSize))
        VortexFooterReader.sumRowCounts(files.asJava, io, options)
      case None => OptionalLong.empty()
    }
    new Statistics {
      override def sizeInBytes(): OptionalLong = base.sizeInBytes()
      override def numRows(): OptionalLong = rows
    }
  }

  /** The listed files that belong to the dataset. The readers skip the others.
    */
  private def vortexFiles =
    partitions
      .flatMap(_.files)
      // A split file appears once per split, and exactly one of them starts at its first byte.
      .filter(file =>
        file.start == 0 && VortexFile.hasVortexExtension(file.toPath.getName)
      )

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
      caseSensitive == scan.caseSensitive &&
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
