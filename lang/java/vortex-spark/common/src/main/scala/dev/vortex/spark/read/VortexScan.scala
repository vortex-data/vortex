// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import java.util.OptionalLong

import scala.collection.JavaConverters._
import org.apache.hadoop.fs.Path

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.{FileSourceOptions, InternalRow}
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.connector.expressions.aggregate.Aggregation
import org.apache.spark.sql.connector.metric.CustomMetric
import org.apache.spark.sql.connector.read.{PartitionReaderFactory, Scan, Statistics}
import org.apache.spark.sql.execution.datasources.{AggregatePushDownUtils, FilePartition, PartitionedFile, PartitioningAwareFileIndex}
import org.apache.spark.sql.execution.datasources.v2.FileScan
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types.StructType

import dev.vortex.spark.VortexOptions
import dev.vortex.spark.io.{VortexFile, VortexIo}

object VortexScan {

  /** Option turning footer-backed row-count statistics off. */
  val RowCountStatisticsOption = "vortex.stats.rowCount"

  /** Option turning file splitting off, so each file is read by one task. */
  val SplitFilesOption = "vortex.splitFiles"

  /** Whether Spark may split `path` into byte ranges, which the reader maps to
    * row ranges.
    */
  def isSplitable(options: VortexOptions, path: Path): Boolean =
    options.getBoolean(SplitFilesOption, true) &&
      VortexFile.hasVortexExtension(path.getName)

}

/** A Vortex scan planned by Spark’s file source. */
class VortexScan(
    override val sparkSession: SparkSession,
    override val fileIndex: PartitioningAwareFileIndex,
    override val dataSchema: StructType,
    override val readDataSchema: StructType,
    override val readPartitionSchema: StructType,
    val options: VortexOptions,
    val pushedFilters: Array[Filter],
    val pushedAggregation: Option[Aggregation],
    val pushedLimit: Option[Int],
    val pushedTopN: Option[VortexTopN],
    val pushedSample: Option[VortexSample],
    val pushedCatalystPredicates: Seq[Expression],
    val pushedOffset: Option[Int],
    val variantColumns: Map[String, StructType],
    override val partitionFilters: Seq[Expression],
    override val dataFilters: Seq[Expression]
) extends FileScan {
  import VortexScan._

  private val caseSensitiveOptions = options.asCaseSensitiveMap.asScala.toMap
  private val hadoopConf =
    sparkSession.sessionState.newHadoopConfWithOptions(caseSensitiveOptions)
  private val io = VortexIo.create(options, hadoopConf)

  /** The I/O the readers use, whose configuration is broadcast once rather than
    * shipped with every task.
    */
  private lazy val executorIo =
    VortexIo.broadcast(options, hadoopConf, sparkSession.sparkContext)
  private val caseSensitive =
    sparkSession.sessionState.conf.caseSensitiveAnalysis

  override def supportedCustomMetrics(): Array[CustomMetric] =
    VortexScanMetrics.supported()

  // Every Vortex reader is columnar, so Spark need not plan the input
  // partitions to ask each one.
  override def columnarSupportMode(): Scan.ColumnarSupportMode =
    Scan.ColumnarSupportMode.SUPPORTED

  override def isSplitable(path: Path): Boolean =
    // An aggregate row stands for a whole file, so a pushed aggregate reads each file in one piece, and an offset
    // skips rows of whole files.
    pushedAggregation.isEmpty && pushedOffset.isEmpty &&
      VortexScan.isSplitable(options, path)

  override def getFileUnSplittableReason(path: Path): String =
    if (pushedAggregation.nonEmpty) {
      "a pushed aggregate is answered once per file"
    } else if (pushedOffset.nonEmpty) {
      "a pushed offset skips rows of whole files"
    } else if (!VortexFile.hasVortexExtension(path.getName)) {
      "the file has no Vortex extension"
    } else {
      s"the $SplitFilesOption option is false"
    }

  override def readSchema(): StructType =
    if (pushedAggregation.nonEmpty) readDataSchema
    else
      StructType(effectiveReadDataSchema.fields ++ readPartitionSchema.fields)

  /** The data columns the readers return: variant columns with pushed
    * extractions become the struct of the values extracted from them.
    */
  private lazy val effectiveReadDataSchema: StructType =
    if (variantColumns.isEmpty) {
      readDataSchema
    } else {
      StructType(readDataSchema.fields.map { field =>
        variantColumns
          .get(field.name)
          .map(extracted => field.copy(dataType = extracted))
          .getOrElse(field)
      })
    }

  override def createReaderFactory(): PartitionReaderFactory = {
    val fileOptions = new FileSourceOptions(caseSensitiveOptions)
    pushedAggregation match {
      case Some(aggregation) =>
        new VortexAggregateReaderFactory(
          fileOptions,
          executorIo,
          options,
          dataSchema,
          readDataSchema,
          readPartitionSchema,
          aggregation,
          caseSensitive,
          exactSums
        )
      case None =>
        new VortexPartitionReaderFactory(
          fileOptions,
          executorIo,
          options,
          dataSchema,
          effectiveReadDataSchema,
          readPartitionSchema,
          pushedFilters,
          caseSensitive,
          pushedLimit.map(_.toLong).getOrElse(0L),
          seededTopN.orNull,
          pushedSample.orNull,
          pushedCatalystPredicates.toArray,
          offsetPlan.firstRows.asJava,
          false
        )
    }
  }

  /** The pushed top-N, starting from the best bound the footers prove, so
    * readers skip files before reading a row.
    */
  private lazy val seededTopN: Option[VortexTopN] = pushedTopN.map { topN =>
    val files = vortexFiles
      .map(file => new VortexFile(file.toPath.toString, file.fileSize))
      .asJava
    val seed = VortexFooterReader.topNSeed(
      files,
      io,
      options,
      dataSchema(topN.column),
      topN.descending,
      topN.limit,
      caseSensitive
    )
    topN.withSeed(seed.orElse(null))
  }

  /** Which files a pushed offset skips whole, and where it ends in the next. */
  private case class OffsetPlan(
      skippedFiles: Set[String],
      firstRows: Map[String, java.lang.Long]
  )

  /** The offset skips rows in the order the files are planned. Without an ORDER
    * BY, Spark leaves the rows an offset skips unspecified, and footer row
    * counts let whole files go unread.
    */
  private lazy val offsetPlan: OffsetPlan = pushedOffset match {
    case None         => OffsetPlan(Set.empty, Map.empty)
    case Some(offset) =>
      val files = super.partitions
        .flatMap(_.files)
        .filter(file => VortexFile.hasVortexExtension(file.toPath.getName))
      val counts = VortexFooterReader
        .exactRowCounts(
          files
            .map(file => new VortexFile(file.toPath.toString, file.fileSize))
            .asJava,
          io,
          options
        )
        .asScala
      var remaining = offset.toLong
      val skipped = Set.newBuilder[String]
      val firstRows = Map.newBuilder[String, java.lang.Long]
      files.zip(counts).foreach { case (file, count) =>
        val path = file.toPath.toString
        if (remaining >= count) {
          skipped += path
          remaining -= count
        } else if (remaining > 0) {
          firstRows += path -> java.lang.Long.valueOf(remaining)
          remaining = 0
        }
      }
      OffsetPlan(skipped.result(), firstRows.result())
  }

  override protected def partitions: Seq[FilePartition] = {
    val skipped = offsetPlan.skippedFiles
    if (skipped.isEmpty) super.partitions
    else super.partitions.flatMap { partition =>
      val remaining = partition.files.filterNot(file => skipped.contains(file.toPath.toString))
      if (remaining.isEmpty) None else Some(FilePartition(partition.index, remaining))
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

  private[read] def exactInputRows(): OptionalLong = {
    val files = fileIndex.listFiles(partitionFilters, Seq.empty).flatMap(_.files)
      .filter(file => VortexFile.hasVortexExtension(file.getPath.getName))
      .map(file => new VortexFile(file.getPath.toString, file.getLen))
    VortexFooterReader.sumExactRowCounts(files.asJava, io, options)
  }

  /** Whether pushed sums follow ANSI rules, which fail on overflow. */
  private def exactSums: Boolean = sparkSession.sessionState.conf.ansiEnabled

  /** The final rows of the pushed aggregation, when every file's footer states
    * its partial exactly, or None. Nothing is computed from data.
    */
  private[read] def footerAggregateRows(): Option[Array[InternalRow]] =
    pushedAggregation.flatMap { aggregation =>
      val factory = new VortexAggregateReaderFactory(
        new FileSourceOptions(caseSensitiveOptions),
        io,
        options,
        dataSchema,
        readDataSchema,
        readPartitionSchema,
        aggregation,
        caseSensitive,
        exactSums
      )
      // An aggregate scan does not split files, so each file is listed once.
      val files = partitions.flatMap(_.files).asJava
      if (!VortexFooterReader.withinFileBudget(files.size(), options)) return None
      val answers = VortexFooterReader
        .mapInParallel(
          files,
          options,
          (file: PartitionedFile) => factory.footerRows(file)
        )
        .asScala
      if (answers.exists(_.isEmpty)) {
        None
      } else {
        Some(
          CompleteAggregation.merge(
            aggregation,
            readDataSchema,
            answers.flatMap(_.get.asScala).toSeq,
            exactSums
          )
        )
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
      equivalentFilters(pushedFilters, scan.pushedFilters) &&
      aggregationsEqual && pushedLimit == scan.pushedLimit &&
      pushedTopN == scan.pushedTopN && pushedSample == scan.pushedSample &&
      variantColumns == scan.variantColumns &&
      pushedCatalystPredicates.map(_.canonicalized) ==
        scan.pushedCatalystPredicates.map(_.canonicalized) &&
        pushedOffset == scan.pushedOffset
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
      "PushedPredicates" -> seqToString(pushedCatalystPredicates),
      "PushedAggregation" -> aggregation,
      "PushedGroupBy" -> groupBy
    ) ++ pushedLimit.map(limit => "PushedLimit" -> s"LIMIT $limit") ++
      pushedOffset.map(offset => "PushedOffset" -> s"OFFSET $offset") ++
      (if (variantColumns.isEmpty) None
       else
         Some(
           "PushedVariants" -> variantColumns.toSeq
             .sortBy(_._1)
             .map { case (column, extracted) =>
               s"$column: ${extracted.catalogString}"
             }
             .mkString("[", ", ", "]")
         )) ++
      pushedTopN.map { topN =>
        val direction = if (topN.descending) "DESC" else "ASC"
        val nulls = if (topN.nullsFirst) "NULLS FIRST" else "NULLS LAST"
        "PushedTopN" -> s"ORDER BY ${topN.column} $direction $nulls LIMIT ${topN.limit}"
      } ++
      pushedSample.map { sample =>
        val method = if (sample.blocks) "SYSTEM " else ""
        "PushedSample" -> s"SAMPLE $method[${sample.lowerBound}, ${sample.upperBound}) SEED ${sample.seed}"
      }
  }
}
