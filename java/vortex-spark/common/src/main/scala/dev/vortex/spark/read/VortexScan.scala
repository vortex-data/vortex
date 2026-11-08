// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import java.util.OptionalLong

import scala.collection.JavaConverters._
import scala.util.Try

import org.apache.hadoop.fs.Path

import org.apache.spark.network.util.JavaUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.{FileSourceOptions, InternalRow}
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.parser.CatalystSqlParser
import org.apache.spark.sql.connector.expressions.{
  Expressions,
  Literal,
  NamedReference,
  NullOrdering,
  SortDirection,
  SortOrder,
  Expression => V2Expression
}
import org.apache.spark.sql.connector.expressions.aggregate.Aggregation
import org.apache.spark.sql.connector.expressions.filter.Predicate
import org.apache.spark.sql.connector.metric.CustomMetric
import org.apache.spark.sql.connector.read.{
  PartitionReaderFactory,
  Scan,
  Statistics,
  SupportsReportOrdering,
  SupportsReportPartitioning,
  SupportsRuntimeV2Filtering
}
import org.apache.spark.sql.connector.read.partitioning.{
  KeyGroupedPartitioning,
  Partitioning,
  UnknownPartitioning
}
import org.apache.spark.sql.connector.read.colstats.ColumnStatistics
import org.apache.spark.sql.execution.datasources.{
  AggregatePushDownUtils,
  FilePartition,
  PartitionedFile,
  PartitioningAwareFileIndex
}
import org.apache.spark.sql.execution.datasources.v2.FileScan
import org.apache.spark.sql.sources.{
  Filter,
  GreaterThanOrEqual,
  In,
  LessThanOrEqual
}
import org.apache.spark.sql.types._

import dev.vortex.spark.VortexOptions
import dev.vortex.spark.io.{VortexFile, VortexIo}

object VortexScan {

  /** Option turning footer-backed row-count statistics off. */
  val RowCountStatisticsOption = "vortex.stats.rowCount"

  /** Option turning footer-backed column statistics off when the cost-based
    * optimizer is enabled.
    */
  val ColumnStatisticsOption = "vortex.stats.columns"

  /** Option turning file splitting off, so each file is read by one task. */
  val SplitFilesOption = "vortex.splitFiles"

  /** Whether Spark may split `path` into byte ranges, which the reader maps to
    * row ranges.
    */
  def isSplitable(options: VortexOptions, path: Path): Boolean =
    options.getBoolean(SplitFilesOption, true) &&
      VortexFile.hasVortexExtension(path.getName)

  /** Option giving the smallest split, such as `64m`. A file is split only into
    * pieces of at least this size, or of Spark's own split size when that is
    * larger, and each piece holds whole chunks. It does not go past
    * `spark.sql.files.maxPartitionBytes`.
    */
  val MinSplitBytesOption = "vortex.minSplitBytes"

  private val DefaultMinSplitBytes = 16L * 1024 * 1024

  /** Option turning off the pruning of files on the driver by the statistics in
    * their footers.
    */
  val PruneFilesOption = "vortex.pruneFiles"

  /** Option turning the report of Hive partition keys to Spark off. Spark only
    * uses the report under `spark.sql.sources.v2.bucketing.enabled`.
    */
  val ReportPartitioningOption = "vortex.reportPartitioning"

  /** Option turning on the report of the order the files store their rows in,
    * which Spark then need not sort again. Off by default: it reads one file or
    * split per task, in file order.
    */
  val ReportOrderingOption = "vortex.reportOrdering"

  /** Option turning runtime filtering, such as dynamic partition pruning, off.
    */
  val RuntimeFilteringOption = "vortex.runtimeFilter"

  /** Option bounding how many values of a runtime filter reach the reader as a
    * set. A larger set is pushed as the range between its smallest and largest
    * value.
    */
  val RuntimeFilterMaxValuesOption = "vortex.runtimeFilter.maxValues"

  private val DefaultRuntimeFilterMaxValues = 10000

  /** Data columns a runtime filter can narrow: those whose literals Vortex
    * compares as Spark does.
    */
  private def runtimeFilterable(dataType: DataType): Boolean =
    dataType match {
      case BooleanType | ByteType | ShortType | IntegerType | LongType |
          DateType | TimestampType | TimestampNTZType | BinaryType =>
        true
      case _: DecimalType => true
      case other          => other == StringType
    }

  /** Types whose Catalyst values order through `Comparable` and whose extrema
    * the cost-based optimizer reads.
    */
  private def hasComparableExtrema(dataType: DataType): Boolean =
    dataType match {
      case BooleanType | ByteType | ShortType | IntegerType | LongType |
          FloatType | DoubleType | DateType | TimestampType |
          TimestampNTZType =>
        true
      case _: DecimalType => true
      case _              => false
    }

  private def quoted(name: String): String =
    "`" + name.replace("`", "``") + "`"
}

/** A Vortex scan planned by Spark's file source framework.
  *
  * Files are split by byte range at chunk boundaries, and each split reads its
  * share of the file's rows. The driver drops files whose footer statistics
  * rule out every row of the pushed and runtime filters. At run time the scan
  * narrows to the join keys of a dynamic pruning filter: on a partition column
  * it drops files on the driver, and on a data column each reader evaluates the
  * key set natively, skipping every file and chunk whose statistics rule all
  * keys out. It offers data columns only in a session with
  * [[dev.vortex.spark.VortexSparkSessionExtensions]].
  *
  * Builds for later Spark lines extend it with what those lines add.
  */
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
) extends FileScan
    with SupportsReportOrdering
    with SupportsReportPartitioning
    with SupportsRuntimeV2Filtering {
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

  /** Runtime filters on partition columns: the accepted values, by the column's
    * ordinal in `readPartitionSchema`.
    */
  @volatile private var runtimePartitionValues =
    Map.empty[Int, (DataType, Set[Any])]

  /** Runtime filters on data columns, which every reader evaluates. */
  @volatile private var runtimeDataFilters = Array.empty[Filter]

  /** The runtime predicates the scan acted on, across every call to `filter`.
    */
  @volatile private var acceptedRuntimePredicates = Vector.empty[Predicate]

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
          readDataSchema,
          readPartitionSchema,
          aggregation,
          caseSensitive
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
          sortedColumn.nonEmpty
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
      topN.column,
      topN.descending,
      topN.limit,
      caseSensitive
    )
    topN.withSeed(seed.orElse(null))
  }

  override def filterAttributes(): Array[NamedReference] = {
    // Runtime filters drop rows that a pushed offset has already counted.
    if (
      pushedAggregation.nonEmpty || pushedOffset.nonEmpty || !options
        .getBoolean(
          RuntimeFilteringOption,
          true
        )
    ) {
      return Array.empty
    }
    val partitionColumns = readPartitionSchema.fields.toSeq

    // Without the rule, pruning filters on data columns can nest exponentially.
    val dataColumns =
      if (NestedDynamicPruning.installed(sparkSession)) {
        readDataSchema.fields.toSeq.filter(field =>
          runtimeFilterable(field.dataType)
        )
      } else {
        Seq.empty
      }

    (partitionColumns ++ dataColumns)
      .map(field => Expressions.column(quoted(field.name)))
      .toArray
  }

  /** Narrows the scan to runtime `predicates`. Spark may call it more than
    * once, and each call narrows the scan further.
    *
    * An `IN` over a partition column drops files on the driver. Any predicate
    * with a V1 filter form over data columns, such as an `IN` of join keys or a
    * comparison with a scalar subquery's value, reaches every reader. Other
    * predicates are left to Spark, which evaluates every runtime filter anyway.
    */
  override def filter(predicates: Array[Predicate]): Unit = {
    var partitionValues = runtimePartitionValues
    val dataColumnFilters = Array.newBuilder[Filter] ++= runtimeDataFilters
    val accepted = Vector.newBuilder[Predicate] ++= acceptedRuntimePredicates
    predicates.foreach { predicate =>
      inList(predicate) match {
        case Some((name, values))
            if fieldIndex(readPartitionSchema, name) >= 0 =>
          val partitionOrdinal = fieldIndex(readPartitionSchema, name)
          val dataType = readPartitionSchema.fields(partitionOrdinal).dataType
          // Binary values compare by identity, so they cannot prune by set membership.
          if (dataType != BinaryType) {
            val keys = values.filter(_ != null).toSet
            partitionValues = partitionValues.updated(
              partitionOrdinal,
              partitionValues
                .get(partitionOrdinal)
                .map { case (_, earlier) => (dataType, earlier & keys) }
                .getOrElse((dataType, keys))
            )
            accepted += predicate
          }
        case Some((name, values)) if fieldIndex(readDataSchema, name) >= 0 =>
          dataColumnFilters ++= dataFilter(
            readDataSchema.fields(fieldIndex(readDataSchema, name)),
            values
          )
          accepted += predicate
        case _ =>
          dataColumnFilter(predicate).foreach { filter =>
            dataColumnFilters += filter
            accepted += predicate
          }
      }
    }
    runtimePartitionValues = partitionValues
    runtimeDataFilters = dataColumnFilters.result()
    acceptedRuntimePredicates = accepted.result()
  }

  /** The runtime predicates the scan has acted on, across every call to
    * `filter`.
    */
  protected def runtimePredicates: Array[Predicate] =
    acceptedRuntimePredicates.toArray

  /** The V1 form of a runtime predicate over data columns alone, which readers
    * evaluate natively.
    */
  private def dataColumnFilter(predicate: Predicate): Option[Filter] = {
    val converted = CatalystPredicates.toV1(predicate)
    if (!converted.isPresent) {
      None
    } else {
      val filter = converted.get
      val dataColumnsOnly = filter.references.forall(reference =>
        fieldIndex(readPartitionSchema, reference) < 0
      )
      if (
        dataColumnsOnly &&
        SparkFilterToVortexExpression.isPushable(filter, readDataSchema)
      ) Some(filter)
      else None
    }
  }

  /** Whether a later Spark line narrows the files to read by more than this
    * scan's own runtime filters.
    */
  protected def filtersFiles: Boolean = false

  /** Whether `file` survives the runtime filters a later Spark line adds. */
  protected def acceptsFile(file: PartitionedFile): Boolean = true

  /** The column and Catalyst values of a runtime `IN` predicate. */
  private def inList(predicate: Predicate): Option[(String, Seq[Any])] =
    if (predicate.name() != "IN") {
      None
    } else {
      predicate.children().toSeq match {
        case (reference: NamedReference) +: literals
            if reference.fieldNames().length == 1 &&
              literals.forall(_.isInstanceOf[Literal[_]]) =>
          Some(
            reference.fieldNames().head -> literals.map(
              _.asInstanceOf[Literal[_]].value(): Any
            )
          )
        case _ => None
      }
    }

  /** The filter a reader evaluates for the join keys `values` of `field`.
    *
    * The native set answers any number of keys in one pass, but the keys travel
    * with every task, so beyond a bound only their range does.
    */
  private def dataFilter(field: StructField, values: Seq[Any]): Seq[Filter] = {
    val attribute = quoted(field.name)
    val keys = values.filter(_ != null)
    val maxValues = options.getInt(
      RuntimeFilterMaxValuesOption,
      DefaultRuntimeFilterMaxValues
    )
    if (keys.size <= maxValues) {
      Seq(In(attribute, keys.toArray))
    } else {
      keys.head match {
        case _: Comparable[_] =>
          val ordered = keys.asInstanceOf[Seq[Comparable[Any]]]
          val lowest =
            ordered.reduce((a, b) => if (a.compareTo(b) <= 0) a else b)
          val highest =
            ordered.reduce((a, b) => if (a.compareTo(b) >= 0) a else b)
          Seq(
            GreaterThanOrEqual(attribute, lowest),
            LessThanOrEqual(attribute, highest)
          )
        case _ => Seq.empty
      }
    }
  }

  private def fieldIndex(structType: StructType, name: String): Int = {
    val exact = structType.fields.indexWhere(_.name == name)
    if (exact >= 0 || caseSensitive) exact
    else structType.fields.indexWhere(_.name.equalsIgnoreCase(name))
  }

  /** Whether the scan reports its Hive partition keys, so Spark can join and
    * aggregate on them without a shuffle. Spark only honours the report under
    * v2 bucketing, and every partition must then hold a single key, so the scan
    * only plans that way when it does.
    */
  private def reportsPartitioning: Boolean =
    readPartitionSchema.nonEmpty && pushedAggregation.isEmpty &&
      sparkSession.sessionState.conf.v2BucketingEnabled &&
      options.getBoolean(ReportPartitioningOption, true) &&
      // Binary values compare by identity, so they cannot key a partition.
      !readPartitionSchema.fields.exists(_.dataType == BinaryType)

  override def outputPartitioning(): Partitioning =
    if (reportsPartitioning) {
      new KeyGroupedPartitioning(
        readPartitionSchema.fields.map(field =>
          Expressions.identity(quoted(field.name)): V2Expression
        ),
        partitions
          .map(_.files.head.partitionValues.toSeq(readPartitionSchema))
          .distinct
          .length
      )
    } else {
      new UnknownPartitioning(0)
    }

  /** The first read column every footer states sorted ascending without nulls,
    * when the scan reports ordering. A split keeps its file's order, so a scan
    * that reads each file or split in its own task, in file order, returns each
    * task's rows in that order.
    */
  private lazy val sortedColumn: Option[String] =
    if (
      !options.getBoolean(ReportOrderingOption, false) ||
      pushedAggregation.nonEmpty || pushedSample.nonEmpty
    ) {
      None
    } else {
      val candidates = readDataSchema.fields
        .filter(field => TopNThreshold.supports(field.dataType))
        .map(_.name)
        .toList
      val files = super.partitions
        .flatMap(_.files)
        .filter(file => VortexFile.hasVortexExtension(file.toPath.getName))
        .filter(_.start == 0)
        .map(file => new VortexFile(file.toPath.toString, file.fileSize))
      val sorted = VortexFooterReader.sortedColumn(
        files.asJava,
        io,
        options,
        candidates.asJava,
        caseSensitive
      )
      if (sorted.isPresent) Some(sorted.get) else None
    }

  override def outputOrdering(): Array[SortOrder] =
    sortedColumn
      .map(column =>
        Array(
          Expressions.sort(
            Expressions.column(quoted(column)),
            SortDirection.ASCENDING,
            NullOrdering.NULLS_FIRST
          )
        )
      )
      .getOrElse(Array.empty)

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
    val planned = plannedPartitions
    val partitionValues = runtimePartitionValues
    val dataColumnFilters = runtimeDataFilters
    val keyed = reportsPartitioning
    if (
      partitionValues.isEmpty && dataColumnFilters.isEmpty && !keyed &&
      sortedColumn.isEmpty && !filtersFiles
    ) {
      planned
    } else {
      val pieces = planned
        .map(
          _.files.filter(file =>
            acceptsPartitionValues(file, partitionValues) && acceptsFile(file)
          )
        )
        .filter(_.nonEmpty)
      if (sortedColumn.nonEmpty && !keyed) {
        // Rows of two files have no order between them, so each file or split is its own partition.
        pieces.flatten.zipWithIndex.map { case (file, index) =>
          new VortexFilePartition(index, Array(file), dataColumnFilters)
        }
      } else if (keyed) {
        // Spark's bin-packing mixes keys in one partition. Split each by key, and Spark groups the pieces again.
        pieces
          .flatMap(files =>
            files
              .groupBy(file => file.partitionValues.toSeq(readPartitionSchema))
              .values
              // With a reported order, every partition must hold one file or split, keyed or not.
              .flatMap(group =>
                if (sortedColumn.nonEmpty) group.map(Array(_)) else Seq(group)
              )
          )
          .zipWithIndex
          .map { case (files, index) =>
            new VortexKeyedFilePartition(
              index,
              files,
              dataColumnFilters,
              files.head.partitionValues
            )
          }
      } else {
        pieces.zipWithIndex.map { case (files, index) =>
          new VortexFilePartition(index, files, dataColumnFilters)
        }
      }
    }
  }

  /** The files to read, packed into partitions.
    *
    * Spark lists the files and the pushed offset skips some of them. Files that
    * the footer statistics prove hold no matching row are dropped, and the rest
    * are split along their chunks into pieces of at least the split size.
    */
  private def plannedPartitions: Seq[FilePartition] = {
    val skippedFiles = offsetPlan.skippedFiles
    val files =
      listedFiles.filterNot(file => skippedFiles.contains(file.toPath.toString))
    val splitSize = splitBytes(files)
    val pieces = split(pruneByStatistics(files), splitSize)

    FilePartition.getFilePartitions(
      sparkSession,
      pieces.sortBy(-_.length),
      splitSize
    )
  }

  /** Every listed file once, whole, in listing order. Spark lists a split file
    * once per split.
    */
  private def listedFiles: Seq[PartitionedFile] =
    super.partitions
      .flatMap(_.files)
      .filter(_.start == 0)
      .map(file => file.copy(start = 0, length = file.fileSize))

  /** The size Spark splits and packs `files` by, raised to the smallest split
    * but not past `spark.sql.files.maxPartitionBytes`.
    */
  private def splitBytes(files: Seq[PartitionedFile]): Long = {
    val conf = sparkSession.sessionState.conf
    val openCost = conf.filesOpenCostInBytes
    val parallelism = conf.filesMinPartitionNum.getOrElse(
      sparkSession.conf
        .getOption("spark.sql.leafNodeDefaultParallelism")
        .map(_.toInt)
        .getOrElse(sparkSession.sparkContext.defaultParallelism)
    )
    val bytesPerCore = files.map(_.fileSize + openCost).sum / parallelism
    val sparkSplit =
      math.min(conf.filesMaxPartitionBytes, math.max(openCost, bytesPerCore))
    val configured = options.get(MinSplitBytesOption)
    val minSplit =
      if (configured.isPresent) JavaUtils.byteStringAsBytes(configured.get.trim)
      else DefaultMinSplitBytes

    // Spark's largest split caps the smallest, so a session that asks for smaller splits still gets them.
    math.max(sparkSplit, math.min(minSplit, conf.filesMaxPartitionBytes))
  }

  /** Splits each Vortex file larger than `splitSize` at the chunk boundaries
    * that start a piece of at least `splitSize` bytes.
    *
    * A reader maps the bytes of its split to rows in proportion to the file, so
    * each boundary becomes the first byte that maps to its row. A piece then
    * reads whole chunks, and no two tasks decode the same chunk.
    */
  private def split(
      files: Seq[PartitionedFile],
      splitSize: Long
  ): Seq[PartitionedFile] = {
    val large = files.filter(file =>
      file.fileSize > splitSize && isSplitable(file.toPath)
    )
    if (large.isEmpty) {
      return files
    }

    val boundaries = VortexFooterReader
      .splitBoundaries(
        large
          .map(file => new VortexFile(file.toPath.toString, file.fileSize))
          .asJava,
        io,
        options
      )
      .asScala
    val cuts = large
      .zip(boundaries)
      .map { case (file, rows) =>
        file.toPath.toString -> cutOffsets(rows, file.fileSize, splitSize)
      }
      .toMap

    files.flatMap { file =>
      cuts.get(file.toPath.toString) match {
        case Some(offsets) if offsets.nonEmpty =>
          (0L +: offsets).zip(offsets :+ file.fileSize).map {
            case (start, end) => file.copy(start = start, length = end - start)
          }
        case _ => Seq(file)
      }
    }
  }

  /** The byte offsets to cut a file of `size` bytes at, given the row
    * boundaries of its chunks.
    */
  private def cutOffsets(
      boundaries: Array[Long],
      size: Long,
      splitSize: Long
  ): Seq[Long] = {
    val rowCount = if (boundaries.isEmpty) 0L else boundaries.last
    val cuts = Seq.newBuilder[Long]
    var start = 0L
    if (rowCount > 0) {
      boundaries.foreach { row =>
        val offset = VortexPartitionReader.splitOffset(rowCount, row, size)
        if (offset < size && offset - start >= splitSize) {
          cuts += offset
          start = offset
        }
      }
    }
    cuts.result()
  }

  /** Drops the Vortex files whose footer statistics prove that no row satisfies
    * the pushed filters and the runtime filters on data columns.
    */
  private def pruneByStatistics(
      files: Seq[PartitionedFile]
  ): Seq[PartitionedFile] = {
    val filters = pushedFilters ++ runtimeDataFilters
    val columns =
      filters.toSeq
        .flatMap(_.references)
        .flatMap(prunableColumn)
        .distinct
        .sorted
    if (
      columns.isEmpty || pushedOffset.nonEmpty ||
      !options.getBoolean(PruneFilesOption, true)
    ) {
      return files
    }

    val vortex =
      files.filter(file => VortexFile.hasVortexExtension(file.toPath.getName))
    statisticsByFile(vortex, columns) match {
      case None             => files
      case Some(statistics) =>
        val dropped = vortex
          .zip(statistics)
          .collect {
            case (file, footer)
                if !filters.forall(filter =>
                  FileStatisticsPruning.mightMatch(
                    filter,
                    footer.rowCount(),
                    attribute =>
                      prunableColumn(attribute)
                        .flatMap(column => Option(footer.columns().get(column)))
                  )
                ) =>
              file.toPath.toString
          }
          .toSet
        files.filterNot(file => dropped.contains(file.toPath.toString))
    }
  }

  /** The footer statistics of `files` for `columns`, read once per set of
    * columns. Spark plans the partitions again after each runtime filter.
    */
  private def statisticsByFile(
      files: Seq[PartitionedFile],
      columns: Seq[String]
  ): Option[Seq[VortexFooterReader.FileStatistics]] =
    statisticsCache.synchronized {
      statisticsCache.getOrElseUpdate(
        columns, {
          val read = VortexFooterReader.statisticsByFile(
            files
              .map(file => new VortexFile(file.toPath.toString, file.fileSize))
              .asJava,
            io,
            options,
            columns.asJava,
            caseSensitive
          )
          if (read.isPresent) Some(read.get.asScala.toSeq) else None
        }
      )
    }

  private val statisticsCache = scala.collection.mutable
    .Map[Seq[String], Option[Seq[VortexFooterReader.FileStatistics]]]()

  /** The data column a filter attribute names, when its footer extrema order as
    * Spark orders its values.
    */
  private def prunableColumn(attribute: String): Option[String] =
    Try(CatalystSqlParser.parseMultipartIdentifier(attribute)).toOption
      .collect { case Seq(name) => fieldIndex(dataSchema, name) }
      .filter(_ >= 0)
      .map(dataSchema.fields(_))
      .filter(field =>
        hasComparableExtrema(field.dataType) &&
          field.dataType != FloatType && field.dataType != DoubleType
      )
      .map(_.name)

  private def acceptsPartitionValues(
      file: PartitionedFile,
      accepted: Map[Int, (DataType, Set[Any])]
  ): Boolean =
    accepted.forall { case (ordinal, (dataType, values)) =>
      !file.partitionValues.isNullAt(ordinal) &&
      values.contains(file.partitionValues.get(ordinal, dataType))
    }

  override def estimateStatistics(): Statistics = cachedStatistics

  private lazy val cachedStatistics: Statistics = {
    val base = super.estimateStatistics()
    val (rows, columns) = pushedAggregation match {
      // The aggregate reader answers from the same footers, and emits one row per file. Reading them here as well
      // would pay for every footer twice over -- once on the driver, once on the executors -- and would describe
      // rows the scan does not emit.
      case Some(_) =>
        (
          OptionalLong.of(vortexFiles.length.toLong),
          Map.empty[NamedReference, ColumnStatistics]
        )
      case None if options.getBoolean(RowCountStatisticsOption, true) =>
        // Count over the same file set the scan will read, so statistics and execution cannot
        // disagree about what belongs to the dataset.
        val files = vortexFiles
          .map(file => new VortexFile(file.toPath.toString, file.fileSize))
          .asJava
        if (columnStatisticsEnabled) {
          val statistics = VortexFooterReader.statistics(
            files,
            io,
            options,
            readDataSchema.fields
              .filter(field => hasComparableExtrema(field.dataType))
              .map(_.name)
              .toList
              .asJava,
            caseSensitive
          )
          (statistics.rowCount(), columnStatistics(statistics))
        } else {
          (
            VortexFooterReader.sumRowCounts(files, io, options),
            Map.empty[NamedReference, ColumnStatistics]
          )
        }
      case None =>
        (OptionalLong.empty(), Map.empty[NamedReference, ColumnStatistics])
    }
    val javaColumns = columns.asJava
    new Statistics {
      override def sizeInBytes(): OptionalLong = base.sizeInBytes()
      override def numRows(): OptionalLong = rows
      override def columnStats()
          : java.util.Map[NamedReference, ColumnStatistics] =
        javaColumns
    }
  }

  /** Column statistics only serve the cost-based optimizer, and cost more
    * footer reads than a row count, so they follow its switch.
    */
  private def columnStatisticsEnabled: Boolean =
    sparkSession.sessionState.conf.cboEnabled &&
      options.getBoolean(ColumnStatisticsOption, true)

  private def columnStatistics(
      statistics: VortexFooterReader.Statistics
  ): Map[NamedReference, ColumnStatistics] =
    statistics.columns().asScala.toMap.map { case (name, column) =>
      // Spark matches a statistic to an output attribute by the reference's description, its plain name.
      val reference: NamedReference = Expressions.column(quoted(name))
      reference -> new ColumnStatistics {
        override def nullCount(): OptionalLong = column.nullCount()
        override def min(): java.util.Optional[Object] = column.min()
        override def max(): java.util.Optional[Object] = column.max()
      }
    }

  /** The final rows of the pushed aggregation, when every file's footer states
    * its partial exactly, or None. Nothing is computed from data.
    */
  private[read] def footerAggregateRows(): Option[Array[InternalRow]] =
    pushedAggregation.flatMap { aggregation =>
      val factory = new VortexAggregateReaderFactory(
        new FileSourceOptions(caseSensitiveOptions),
        io,
        options,
        readDataSchema,
        readPartitionSchema,
        aggregation,
        caseSensitive
      )
      // An aggregate scan does not split files, so each file is listed once.
      val files = partitions.flatMap(_.files).asJava
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
            answers.flatMap(_.get.asScala).toSeq
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
