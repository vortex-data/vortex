// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.expressions.{
  Expression,
  NamedReference,
  NullOrdering,
  SortDirection,
  SortOrder
}
import org.apache.spark.sql.connector.expressions.aggregate.{
  AggregateFunc,
  Aggregation,
  Count,
  CountStar,
  Max,
  Min,
  Sum
}
import org.apache.spark.sql.connector.read.{
  Scan,
  SupportsPushDownAggregates,
  SupportsPushDownLimit,
  SupportsPushDownOffset,
  SupportsPushDownTableSample,
  SupportsPushDownTopN
}
import org.apache.spark.sql.execution.datasources.{
  AggregatePushDownUtils,
  PartitioningAwareFileIndex
}
import org.apache.spark.sql.execution.datasources.v2.FileScanBuilder
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types._

import dev.vortex.spark.VortexOptions

object VortexScanBuilder {

  /** Option turning aggregate pushdown off. */
  val AggregatePushdownOption = "vortex.aggregatePushdown"

  /** Option turning off answering aggregates in full on the driver from the
    * footers, which leaves Spark to combine one partial row per file.
    */
  val CompleteAggregatePushdownOption = "vortex.completeAggregatePushdown"

  /** Option turning LIMIT pushdown off. */
  val LimitPushdownOption = "vortex.limitPushdown"

  /** Option turning ORDER BY ... LIMIT pushdown off. */
  val TopNPushdownOption = "vortex.topNPushdown"

  /** Option turning OFFSET pushdown off. */
  val OffsetPushdownOption = "vortex.offsetPushdown"

  /** Option turning TABLESAMPLE pushdown off. */
  val SamplePushdownOption = "vortex.samplePushdown"

  /** Option turning off the pushdown of predicates without a V1 form, such as
    * `LIKE` patterns and comparisons between columns.
    */
  val PredicatePushdownOption = "vortex.predicatePushdown"

  /** Whether the reader answers MIN and MAX over a data column of `dataType`.
    *
    * Vortex orders these types as Spark does. A collated string orders by its
    * collation instead, and nested types have no extrema in Vortex.
    */
  private[read] def supportsExtrema(dataType: DataType): Boolean =
    dataType match {
      case BooleanType | ByteType | ShortType | IntegerType | LongType |
          FloatType | DoubleType | DateType | TimestampType | TimestampNTZType |
          BinaryType =>
        true
      case _: DecimalType => true
      case other          => other == StringType
    }
}

/** Builds Vortex file scans with projection, nested column, filter, LIMIT,
  * ORDER BY ... LIMIT, OFFSET, TABLESAMPLE, and aggregate pushdown.
  *
  * Builds for later Spark lines extend it with the pushdowns those lines add.
  */
class VortexScanBuilder(
    sparkSession: SparkSession,
    fileIndex: PartitioningAwareFileIndex,
    dataSchema: StructType,
    options: VortexOptions
) extends FileScanBuilder(sparkSession, fileIndex, dataSchema)
    with SupportsPushDownAggregates
    with SupportsPushDownLimit
    with SupportsPushDownOffset
    with SupportsPushDownTableSample
    with SupportsPushDownTopN {

  private val partitionSchema = fileIndex.partitionSchema
  private val caseSensitive =
    sparkSession.sessionState.conf.caseSensitiveAnalysis

  private var finalSchema = new StructType()
  private var pushedAggregation = Option.empty[Aggregation]

  /** The last aggregation asked about, and its final rows from the footers. */
  private var footerAnswer =
    Option.empty[(Aggregation, Option[Array[InternalRow]])]
  private var pushedLimit = Option.empty[Int]
  private var pushedTopN = Option.empty[VortexTopN]
  private var pushedSample = Option.empty[VortexSample]
  private var pushedOffset = Option.empty[Int]

  /** For each top-level variant column with pushed extractions, the struct of
    * the values extracted from it.
    */
  protected var variantColumns = Map.empty[String, StructType]

  /** Whether an aggregate is pushed, which leaves no rows to extract from. */
  protected def aggregationPushed: Boolean = pushedAggregation.nonEmpty

  override protected def pushDataFilters(
      filters: Array[Filter]
  ): Array[Filter] =
    filters.filter(SparkFilterToVortexExpression.isPushable(_, dataSchema))

  override def pushLimit(limit: Int): Boolean = {
    // Spark only offers a LIMIT that no filter above the scan has to see first, and keeps applying it itself, so
    // each file can stop reading after `limit` rows.
    if (
      limit <= 0 || !options.getBoolean(
        VortexScanBuilder.LimitPushdownOption,
        true
      )
    ) {
      return false
    }
    pushedLimit = Some(limit)
    true
  }

  override def pushOffset(offset: Int): Boolean = {
    // Spark drops its own OFFSET once pushed, so the scan must skip exactly `offset` rows. Footer row counts say
    // which rows those are, unless a sample, an aggregate, or a top-N changes the rows the scan returns.
    if (
      offset <= 0 || pushedSample.nonEmpty || pushedAggregation.nonEmpty ||
      pushedTopN.nonEmpty || !options.getBoolean(
        VortexScanBuilder.OffsetPushdownOption,
        true
      )
    ) {
      return false
    }
    pushedOffset = Some(offset)
    true
  }

  override def pushTableSample(
      lowerBound: Double,
      upperBound: Double,
      withReplacement: Boolean,
      seed: Long
  ): Boolean =
    pushSample(lowerBound, upperBound, withReplacement, seed, blocks = false)

  /** Pushes a sample of rows, or with `blocks` of whole chunks of rows. */
  protected def pushSample(
      lowerBound: Double,
      upperBound: Double,
      withReplacement: Boolean,
      seed: Long,
      blocks: Boolean
  ): Boolean = {
    // A sample with replacement repeats rows, which a row selection cannot express.
    if (
      withReplacement || !options.getBoolean(
        VortexScanBuilder.SamplePushdownOption,
        true
      )
    ) {
      return false
    }
    pushedSample = Some(new VortexSample(lowerBound, upperBound, seed, blocks))
    true
  }

  // Both a LIMIT and a top-N bound each reader's work only, so Spark always applies them again.
  override def isPartiallyPushed(): Boolean = true

  override def pushTopN(orders: Array[SortOrder], limit: Int): Boolean = {
    // Spark keeps sorting and limiting what the scan returns, so readers only drop rows that cannot reach the
    // result. That needs the first sort key to be a data column whose values Vortex orders as Spark does.
    if (
      limit <= 0 || orders.isEmpty || pushedAggregation.nonEmpty ||
      !options.getBoolean(VortexScanBuilder.TopNPushdownOption, true)
    ) {
      return false
    }
    val first = orders.head
    val pushed = columnName(first.expression())
      .filter(findField(partitionSchema, _).isEmpty)
      .flatMap(findField(dataSchema, _))
      .filter(field => TopNThreshold.supports(field.dataType))
      .map { field =>
        new VortexTopN(
          field.name,
          first.direction() == SortDirection.DESCENDING,
          first.nullOrdering() == NullOrdering.NULLS_FIRST,
          orders.length == 1,
          limit,
          null
        )
      }
    pushedTopN = pushed
    pushed.nonEmpty
  }

  override def pushAggregation(aggregation: Aggregation): Boolean = {
    // Footer statistics describe every row, not a sample of them.
    if (!aggregatePushdownEnabled || pushedSample.nonEmpty) {
      return false
    }
    pushedAggregationSchema(aggregation) match {
      case Some(aggregateSchema) =>
        finalSchema = aggregateSchema
        pushedAggregation = Some(aggregation)
        true
      case None => false
    }
  }

  /** Whether the footers answer `aggregation` in full. Then the driver merges
    * the files' partials into the final rows, and the scan runs no task.
    * Otherwise each file's reader emits a partial row that Spark combines.
    */
  override def supportCompletePushDown(aggregation: Aggregation): Boolean =
    footerAggregateRows(aggregation).nonEmpty

  /** The final rows of `aggregation` from the footers, or None when a footer
    * does not state its partial exactly. Spark asks before and after it pushes
    * the aggregation, so the answer is kept.
    */
  private def footerAggregateRows(
      aggregation: Aggregation
  ): Option[Array[InternalRow]] =
    footerAnswer match {
      case Some((answered, rows))
          if AggregatePushDownUtils.equivalentAggregations(
            answered,
            aggregation
          ) =>
        rows
      case _ =>
        val rows =
          if (
            !aggregatePushdownEnabled || pushedSample.nonEmpty || !options
              .getBoolean(
                VortexScanBuilder.CompleteAggregatePushdownOption,
                true
              )
          ) {
            None
          } else {
            pushedAggregationSchema(aggregation).flatMap { schema =>
              scan(schema, Some(aggregation)).footerAggregateRows()
            }
          }
        footerAnswer = Some((aggregation, rows))
        rows
    }

  override def build(): Scan = {
    val complete = pushedAggregation.flatMap(footerAggregateRows)
    complete match {
      case Some(rows) =>
        new VortexLocalScan(
          finalSchema,
          rows,
          s"VortexLocalScan [${pushedAggregation.get.aggregateExpressions().map(_.describe()).mkString(", ")}] " +
            s"from the footers of ${fileIndex.rootPaths.mkString(", ")}"
        )
      case None =>
        if (pushedAggregation.isEmpty) {
          finalSchema = readDataSchema()
        }
        scan(finalSchema, pushedAggregation)
    }
  }

  private def scan(
      schema: StructType,
      aggregation: Option[Aggregation]
  ): VortexScan =
    newScan(
      sparkSession,
      fileIndex,
      dataSchema,
      schema,
      readPartitionSchema(),
      options,
      pushedDataFilters,
      aggregation,
      if (aggregation.isEmpty) pushedLimit else None,
      if (aggregation.isEmpty) pushedTopN else None,
      pushedSample,
      catalystPredicates,
      pushedOffset,
      if (aggregation.isEmpty) variantColumns else Map.empty,
      partitionFilters,
      dataFilters
    )

  /** The read columns in table order, with struct columns pruned to the nested
    * fields Spark requires. See [[NestedSchemaPruning]] for which nested types
    * are pruned.
    */
  override protected def readDataSchema(): StructType = {
    val resolver = sparkSession.sessionState.conf.resolver
    StructType(super.readDataSchema().fields.map { field =>
      requiredSchema.fields
        .find(required => resolver(required.name, field.name)) match {
        case Some(required) =>
          field.copy(dataType =
            NestedSchemaPruning.prune(field.dataType, required.dataType)
          )
        case None => field
      }
    })
  }

  /** Creates the scan. A later Spark line returns a scan extended with what it
    * adds.
    */
  protected def newScan(
      sparkSession: SparkSession,
      fileIndex: PartitioningAwareFileIndex,
      dataSchema: StructType,
      readDataSchema: StructType,
      readPartitionSchema: StructType,
      options: VortexOptions,
      pushedFilters: Array[Filter],
      pushedAggregation: Option[Aggregation],
      pushedLimit: Option[Int],
      pushedTopN: Option[VortexTopN],
      pushedSample: Option[VortexSample],
      pushedCatalystPredicates: Seq[
        org.apache.spark.sql.catalyst.expressions.Expression
      ],
      pushedOffset: Option[Int],
      variantColumns: Map[String, StructType],
      partitionFilters: Seq[
        org.apache.spark.sql.catalyst.expressions.Expression
      ],
      dataFilters: Seq[org.apache.spark.sql.catalyst.expressions.Expression]
  ): VortexScan =
    new VortexScan(
      sparkSession,
      fileIndex,
      dataSchema,
      readDataSchema,
      readPartitionSchema,
      options,
      pushedFilters,
      pushedAggregation,
      pushedLimit,
      pushedTopN,
      pushedSample,
      pushedCatalystPredicates,
      pushedOffset,
      variantColumns,
      partitionFilters,
      dataFilters
    )

  /** Data filters without a V1 form, such as `LIKE` patterns and comparisons
    * between columns, that the reader still converts natively.
    */
  private def catalystPredicates: Seq[
    org.apache.spark.sql.catalyst.expressions.Expression
  ] =
    if (!options.getBoolean(VortexScanBuilder.PredicatePushdownOption, true)) {
      Seq.empty
    } else {
      dataFilters.filter(filter =>
        !CatalystPredicates.hasV1Form(filter) &&
          CatalystPredicates.isPushable(filter, dataSchema)
      )
    }

  /** Escape hatch matching `spark.sql.parquet.aggregatePushdown`, for when a
    * footer statistic must not be trusted.
    */
  private def aggregatePushdownEnabled: Boolean =
    options.getBoolean(VortexScanBuilder.AggregatePushdownOption, true)

  /** The schema of the partial aggregate rows, group-by columns first, or None
    * when the reader cannot answer `aggregation`.
    *
    * Every file holds one value of each partition column, so the reader can
    * group by any of them. An aggregate over a data column comes from the
    * footer statistics, or from a scan of that column alone where the footer
    * does not state it exactly.
    */
  private def pushedAggregationSchema(
      aggregation: Aggregation
  ): Option[StructType] = {
    // A filter above the scan must see every row, so no aggregate can be answered for it.
    if (pushedDataFilters.nonEmpty || dataFilters.nonEmpty) {
      return None
    }

    val groupBy = aggregation.groupByExpressions().map { expression =>
      columnName(expression).flatMap(findField(partitionSchema, _))
    }
    val aggregates = aggregation.aggregateExpressions().map(aggregateField)
    if (groupBy.exists(_.isEmpty) || aggregates.exists(_.isEmpty)) {
      None
    } else {
      Some(new StructType((groupBy ++ aggregates).collect { case Some(field) =>
        field
      }))
    }
  }

  private def aggregateField(aggregate: AggregateFunc): Option[StructField] =
    aggregate match {
      case _: CountStar => Some(StructField("count(*)", LongType))
      case count: Count if !count.isDistinct =>
        column(count.column).map(field =>
          StructField(s"count(${field.name})", LongType)
        )
      case min: Min =>
        extremumColumn(min.column).map(field =>
          field.copy(name = s"min(${field.name})", nullable = true)
        )
      case max: Max =>
        extremumColumn(max.column).map(field =>
          field.copy(name = s"max(${field.name})", nullable = true)
        )
      case sum: Sum if !sum.isDistinct => sumColumn(sum.column)
      case _                           => None
    }

  /** A data or partition column. */
  private def column(expression: Expression): Option[StructField] =
    columnName(expression).flatMap(name =>
      findField(partitionSchema, name).orElse(findField(dataSchema, name))
    )

  private def extremumColumn(expression: Expression): Option[StructField] =
    columnName(expression).flatMap { name =>
      findField(partitionSchema, name).orElse(
        findField(dataSchema, name)
          .filter(field => VortexScanBuilder.supportsExtrema(field.dataType))
      )
    }

  /** SUM over an integer data column, whose 64-bit result wraps on overflow
    * like Spark's own LONG addition. Under ANSI mode an overflow must fail, and
    * a per-file partial cannot tell, so Spark keeps the sum.
    */
  private def sumColumn(expression: Expression): Option[StructField] = {
    if (sparkSession.sessionState.conf.ansiEnabled) {
      return None
    }
    columnName(expression)
      .filter(findField(partitionSchema, _).isEmpty)
      .flatMap(findField(dataSchema, _))
      .collect {
        case field
            if Seq(ByteType, ShortType, IntegerType, LongType)
              .contains(field.dataType) =>
          StructField(s"sum(${field.name})", LongType)
      }
  }

  private def columnName(expression: Expression): Option[String] =
    expression match {
      case reference: NamedReference if reference.fieldNames().length == 1 =>
        Some(reference.fieldNames().head)
      case _ => None
    }

  private def findField(
      structType: StructType,
      name: String
  ): Option[StructField] =
    structType.fields
      .find(_.name == name)
      .orElse(
        if (caseSensitive) None
        else structType.fields.find(_.name.equalsIgnoreCase(name))
      )
}
