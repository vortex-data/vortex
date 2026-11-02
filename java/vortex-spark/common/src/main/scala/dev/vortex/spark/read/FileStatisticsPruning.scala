// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import org.apache.spark.sql.catalyst.CatalystTypeConverters
import org.apache.spark.sql.sources._

import dev.vortex.spark.read.VortexFooterReader.ColumnStatistics

/** Decides from the footer statistics of a file whether any of its rows can
  * satisfy a filter, so the driver can drop the files that cannot.
  *
  * The answer errs towards keeping a file: a filter, column, or value it does
  * not understand keeps the file, and Spark evaluates every filter above the
  * scan anyway.
  */
private[read] object FileStatisticsPruning {

  /** Whether a row of a file with `rowCount` rows can satisfy `filter`.
    *
    * @param column
    *   the statistics of the column a filter attribute names, or None when the
    *   file has no statistics for it
    */
  def mightMatch(
      filter: Filter,
      rowCount: Long,
      column: String => Option[ColumnStatistics]
  ): Boolean =
    filter match {
      case And(left, right) =>
        mightMatch(left, rowCount, column) &&
        mightMatch(right, rowCount, column)

      case Or(left, right) =>
        mightMatch(left, rowCount, column) ||
        mightMatch(right, rowCount, column)

      case _: AlwaysFalse => false

      case IsNull(attribute) => mayHoldNull(column(attribute))

      case IsNotNull(attribute) =>
        column(attribute).forall(statistics =>
          !statistics.holdsNoValues() &&
            (statistics.nullCount().isEmpty ||
              statistics.nullCount().getAsLong < rowCount)
        )

      case EqualNullSafe(attribute, null) => mayHoldNull(column(attribute))

      case EqualNullSafe(attribute, value) =>
        within(column(attribute), value)

      case EqualTo(attribute, value) => within(column(attribute), value)

      case In(attribute, values) =>
        val statistics = column(attribute)
        values.exists(within(statistics, _))

      case GreaterThan(attribute, value) =>
        compared(column(attribute), value)((_, max) => max > 0)

      case GreaterThanOrEqual(attribute, value) =>
        compared(column(attribute), value)((_, max) => max >= 0)

      case LessThan(attribute, value) =>
        compared(column(attribute), value)((min, _) => min < 0)

      case LessThanOrEqual(attribute, value) =>
        compared(column(attribute), value)((min, _) => min <= 0)

      case _ => true
    }

  private def mayHoldNull(statistics: Option[ColumnStatistics]): Boolean =
    statistics.forall(column =>
      column.nullCount().isEmpty || column.nullCount().getAsLong > 0
    )

  private def within(
      statistics: Option[ColumnStatistics],
      value: Any
  ): Boolean =
    compared(statistics, value)((min, max) => min <= 0 && max >= 0)

  /** Applies `test` to how the column's minimum and maximum compare with
    * `value`, as `compareTo` does. A comparison with null is never true, and a
    * column with only nulls holds no value to compare.
    */
  private def compared(statistics: Option[ColumnStatistics], value: Any)(
      test: (Int, Int) => Boolean
  ): Boolean =
    if (value == null) {
      false
    } else {
      statistics.forall { column =>
        if (column.holdsNoValues()) {
          false
        } else if (!column.min().isPresent || !column.max().isPresent) {
          true
        } else {
          val min = column.min().get
          val max = column.max().get
          val catalyst = CatalystTypeConverters.convertToCatalyst(value)
          if (!comparable(min, max, catalyst)) {
            true
          } else {
            test(
              min.asInstanceOf[Comparable[Any]].compareTo(catalyst),
              max.asInstanceOf[Comparable[Any]].compareTo(catalyst)
            )
          }
        }
      }
    }

  /** Whether the extrema and the value compare as Spark compares them. Floating
    * point values are left out: footers and Spark may order NaN differently.
    */
  private def comparable(min: Any, max: Any, value: Any): Boolean =
    value match {
      case _: java.lang.Float | _: java.lang.Double => false
      case _: Comparable[_]                         =>
        min.getClass == value.getClass && max.getClass == value.getClass
      case _ => false
    }
}
