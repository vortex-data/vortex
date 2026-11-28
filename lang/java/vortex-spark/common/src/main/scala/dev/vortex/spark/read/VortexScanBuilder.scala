// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.expressions.{
  Expression,
  NamedReference
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
import org.apache.spark.sql.connector.read.{Scan, SupportsPushDownAggregates}
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

/** Builds Vortex file scans with projection, filter, and aggregate pushdown. */
class VortexScanBuilder(
    sparkSession: SparkSession,
    fileIndex: PartitioningAwareFileIndex,
    schema: StructType,
    dataSchema: StructType,
    options: VortexOptions
) extends FileScanBuilder(sparkSession, fileIndex, dataSchema)
    with SupportsPushDownAggregates {

  private val partitionSchema = fileIndex.partitionSchema
  private val caseSensitive =
    sparkSession.sessionState.conf.caseSensitiveAnalysis

  private var finalSchema = new StructType()
  private var pushedAggregation = Option.empty[Aggregation]

  /** The last aggregation asked about, and its final rows from the footers. */
  private var footerAnswer =
    Option.empty[(Aggregation, Option[Array[InternalRow]])]
  override protected def pushDataFilters(
      filters: Array[Filter]
  ): Array[Filter] =
    filters.filter(SparkFilterToVortexExpression.isPushable(_, dataSchema))

  override def pushAggregation(aggregation: Aggregation): Boolean = {
    // Footer statistics describe every row, not a sample of them.
    if (!aggregatePushdownEnabled) {
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
            !aggregatePushdownEnabled || !options
              .getBoolean(
                VortexScanBuilder.CompleteAggregatePushdownOption,
                true
              )
          ) {
            None
          } else {
            pushedAggregationSchema(aggregation).filter { schema =>
              schema.fields.take(aggregation.groupByExpressions().length).forall { field =>
                field.dataType match {
                  case BooleanType | ByteType | ShortType | IntegerType | LongType | DateType |
                      TimestampType | TimestampNTZType => true
                  case t => t == StringType
                }
              }
            }.flatMap(schema => scan(schema, Some(aggregation)).footerAggregateRows())
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
    new VortexScan(
      sparkSession,
      fileIndex,
      dataSchema,
      schema,
      readPartitionSchema(),
      options,
      pushedDataFilters,
      aggregation,
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

  /** SUM over an integer data column, or over such a column plus a constant.
    *
    * Without ANSI rules, the 64-bit sum wraps on overflow like Spark's own LONG
    * addition, so any integer column qualifies. Under ANSI rules an overflow
    * must fail, and a wrapped per-file partial cannot tell. A sum of `TINYINT`,
    * `SMALLINT`, or `INT` values cannot overflow in a file of up to 2^32 rows,
    * so only columns whose total and partial sums provably fit qualify. Spark
    * states `column + constant` to a source only under ANSI rules.
    */
  private def sumColumn(expression: Expression): Option[StructField] = {
    val ansi = sparkSession.sessionState.conf.ansiEnabled
    val sumTypes: Seq[DataType] =
      if (ansi) Seq(ByteType, ShortType, IntegerType)
      else Seq(ByteType, ShortType, IntegerType, LongType)

    def safe(field: StructField, offset: Long): Boolean = {
      val magnitude = (BigInt(1) << (field.dataType.defaultSize * 8 - 1)) + BigInt(offset).abs
      !ansi || (ansiSumRows.isPresent && magnitude * ansiSumRows.getAsLong <= Long.MaxValue)
    }

    columnName(expression) match {
      case Some(name) =>
        dataColumn(name)
          .filter(field => sumTypes.contains(field.dataType) && safe(field, 0L))
          .map(field => StructField(s"sum(${field.name})", LongType))
      case None if ansi =>
        Option(ShiftedSum.parse(expression).orElse(null))
          .flatMap(shifted =>
            dataColumn(shifted.column())
              .filter(field => shifted.isExact(field.dataType) && safe(field, shifted.offset()))
              .map(field =>
                StructField(
                  s"sum(${field.name} + ${shifted.offset()})",
                  LongType
                )
              )
          )
      case None => None
    }
  }

  // A bound on all selected rows proves every partial and intermediate addition fits, regardless
  // of the file order or the order in which Spark merges partials.
  private lazy val ansiSumRows = scan(new StructType(), None).exactInputRows()

  /** A data column that no partition column shadows. */
  private def dataColumn(name: String): Option[StructField] =
    if (findField(partitionSchema, name).nonEmpty) None
    else findField(dataSchema, name)

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
