// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.connector.expressions.aggregate.{
  Aggregation,
  CountStar
}
import org.apache.spark.sql.connector.read.SupportsPushDownAggregates
import org.apache.spark.sql.execution.datasources.{
  AggregatePushDownUtils,
  PartitioningAwareFileIndex
}
import org.apache.spark.sql.execution.datasources.v2.FileScanBuilder
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types.StructType

import dev.vortex.spark.VortexOptions

object VortexScanBuilder {

  /** Option turning COUNT(*) pushdown off. */
  val AggregatePushdownOption = "vortex.aggregatePushdown"
}

/** Builds Vortex file scans with projection, filter, and COUNT(*) pushdown. */
final class VortexScanBuilder(
    sparkSession: SparkSession,
    fileIndex: PartitioningAwareFileIndex,
    schema: StructType,
    dataSchema: StructType,
    options: VortexOptions
) extends FileScanBuilder(sparkSession, fileIndex, dataSchema)
    with SupportsPushDownAggregates {

  private var finalSchema = new StructType()
  private var pushedAggregation = Option.empty[Aggregation]

  override protected def pushDataFilters(
      filters: Array[Filter]
  ): Array[Filter] =
    filters.filter(SparkFilterToVortexExpression.isPushable(_, dataSchema))

  override def pushAggregation(aggregation: Aggregation): Boolean = {
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

  // The reader emits one partial aggregate per file. Spark must combine files that share a group.
  override def supportCompletePushDown(aggregation: Aggregation): Boolean =
    false

  override def build(): VortexScan = {
    if (pushedAggregation.isEmpty) {
      finalSchema = readDataSchema()
    }
    new VortexScan(
      sparkSession,
      fileIndex,
      dataSchema,
      finalSchema,
      readPartitionSchema(),
      options,
      pushedDataFilters,
      pushedAggregation,
      partitionFilters,
      dataFilters
    )
  }

  /** Escape hatch matching `spark.sql.parquet.aggregatePushdown`, for when a
    * footer count must not be trusted.
    */
  private def aggregatePushdownEnabled: Boolean =
    options.getBoolean(VortexScanBuilder.AggregatePushdownOption, true)

  private def pushedAggregationSchema(
      aggregation: Aggregation
  ): Option[StructType] = {
    if (
      pushedDataFilters.nonEmpty ||
      !aggregation.aggregateExpressions().forall(_.isInstanceOf[CountStar])
    ) {
      None
    } else {
      AggregatePushDownUtils.getSchemaForPushedAggregation(
        aggregation,
        schema,
        partitionNameSet,
        dataFilters
      )
    }
  }
}
