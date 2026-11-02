// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.connector.expressions.aggregate.Aggregation
import org.apache.spark.sql.connector.read.SampleMethod
import org.apache.spark.sql.execution.datasources.PartitioningAwareFileIndex
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types.StructType

import dev.vortex.spark.VortexOptions

/** The scan builder for Spark 4.2, which adds `TABLESAMPLE SYSTEM` and
  * runtime filters that Spark evaluates on partition values.
  *
  * A `SYSTEM` sample keeps or drops whole chunks of rows, so a reader decodes
  * only the chunks it keeps.
  */
class VortexScanBuilder42(
    sparkSession: SparkSession,
    fileIndex: PartitioningAwareFileIndex,
    dataSchema: StructType,
    options: VortexOptions
) extends VortexScanBuilder41(sparkSession, fileIndex, dataSchema, options) {

  override def pushTableSample(
      lowerBound: Double,
      upperBound: Double,
      withReplacement: Boolean,
      seed: Long,
      sampleMethod: SampleMethod
  ): Boolean =
    pushSample(
      lowerBound,
      upperBound,
      withReplacement,
      seed,
      blocks = sampleMethod == SampleMethod.SYSTEM
    )

  override protected def newScan(
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
      pushedCatalystPredicates: Seq[Expression],
      pushedOffset: Option[Int],
      variantColumns: Map[String, StructType],
      partitionFilters: Seq[Expression],
      dataFilters: Seq[Expression]
  ): VortexScan =
    new VortexScan42(
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
}
