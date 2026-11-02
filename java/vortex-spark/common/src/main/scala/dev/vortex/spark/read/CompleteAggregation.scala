// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import scala.collection.mutable

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.util.TypeUtils
import org.apache.spark.sql.connector.expressions.aggregate.{
  AggregateFunc,
  Aggregation,
  Count,
  CountStar,
  Max,
  Min,
  Sum
}
import org.apache.spark.sql.connector.read.LocalScan
import org.apache.spark.sql.types.{DataType, StructType}

/** Combines per-file partial aggregates into the final aggregate rows, which
  * Spark then takes as they are.
  */
private[read] object CompleteAggregation {

  /** The final rows of `aggregation` from the partial rows of every file.
    *
    * Each partial row holds the group-by values first, then one partial per
    * aggregate, as `schema` lays them out. Files that share their group-by
    * values merge into one row. With no group-by column, an empty dataset still
    * has one row, as Spark's own aggregate gives.
    */
  def merge(
      aggregation: Aggregation,
      schema: StructType,
      partials: Seq[InternalRow]
  ): Array[InternalRow] = {
    val groups = aggregation.groupByExpressions().length
    val aggregates = aggregation.aggregateExpressions()
    val types = schema.fields.map(_.dataType)

    val merged = mutable.LinkedHashMap.empty[Seq[Any], Array[Any]]
    for (row <- partials) {
      val key = (0 until groups).map(i => row.get(i, types(i)))
      val values: Array[Any] = Array.tabulate(aggregates.length) { j =>
        row.get(groups + j, types(groups + j))
      }

      merged.get(key) match {
        case None              => merged(key) = values
        case Some(accumulated) =>
          for (j <- aggregates.indices) {
            accumulated(j) = combine(
              aggregates(j),
              types(groups + j),
              accumulated(j),
              values(j)
            )
          }
      }
    }

    if (merged.isEmpty && groups == 0) {
      merged(Seq.empty) =
        Array.tabulate[Any](aggregates.length)(j => empty(aggregates(j)))
    }

    merged.map { case (key, values) =>
      InternalRow.fromSeq(key ++ values)
    }.toArray
  }

  private def combine(
      aggregate: AggregateFunc,
      dataType: DataType,
      accumulated: Any,
      value: Any
  ): Any =
    aggregate match {
      case _: CountStar | _: Count =>
        accumulated.asInstanceOf[Long] + value.asInstanceOf[Long]
      case _: Sum =>
        // Integer sums wrap to 64 bits, like the per-file partials and Spark's own LONG addition.
        if (accumulated == null) value
        else if (value == null) accumulated
        else accumulated.asInstanceOf[Long] + value.asInstanceOf[Long]
      case _: Min => extremum(dataType, accumulated, value, smaller = true)
      case _: Max => extremum(dataType, accumulated, value, smaller = false)
      case other  =>
        throw new IllegalArgumentException(s"Unsupported aggregate $other")
    }

  private def extremum(
      dataType: DataType,
      accumulated: Any,
      value: Any,
      smaller: Boolean
  ): Any = {
    if (accumulated == null) {
      return value
    }
    if (value == null) {
      return accumulated
    }

    // Spark's ordering, under which NaN is the largest value and -0.0 equals 0.0.
    val ordering = TypeUtils.getInterpretedOrdering(dataType)
    val replaces =
      if (smaller) ordering.lt(value, accumulated)
      else ordering.gt(value, accumulated)
    if (replaces) value else accumulated
  }

  private def empty(aggregate: AggregateFunc): Any =
    aggregate match {
      case _: CountStar | _: Count => 0L
      case _                       => null
    }
}

/** A scan whose rows the driver already knows, so Spark runs no task for it. */
private[read] final class VortexLocalScan(
    schema: StructType,
    localRows: Array[InternalRow],
    scanDescription: String
) extends LocalScan {
  override def rows(): Array[InternalRow] = localRows

  override def description(): String = scanDescription

  override def readSchema(): StructType = schema
}
