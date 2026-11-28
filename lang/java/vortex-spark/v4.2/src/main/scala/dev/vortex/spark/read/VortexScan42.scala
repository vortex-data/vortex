// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.connector.expressions.aggregate.Aggregation
import org.apache.spark.sql.connector.expressions.filter.{
  PartitionPredicate,
  Predicate
}
import org.apache.spark.sql.execution.datasources.{
  PartitionedFile,
  PartitioningAwareFileIndex
}
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types.StructType

import dev.vortex.spark.VortexOptions

/** The scan for Spark 4.2, which takes part in iterative runtime filtering.
  *
  * Spark first offers the runtime filters it can translate. It then wraps the
  * rest that read only partition columns, such as a filter on a function of a
  * partition column, in partition predicates that Spark itself evaluates. The
  * scan evaluates them once per file on the driver, and drops the files they
  * rule out.
  *
  * A partition predicate reads every partition column, which Spark only
  * offers when the scan reads them all. Spark evaluates every runtime filter
  * above the scan, so a predicate the scan declines changes no result.
  */
class VortexScan42(
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
) extends VortexScan(
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
    ) {

  /** The partition predicates, across every call to `filter`. */
  @volatile private var partitionPredicates = Vector.empty[PartitionPredicate]

  override def supportsIterativePushdown(): Boolean = true

  /** Whether each file carries the values of every partition column. */
  private def readsAllPartitionColumns: Boolean =
    readPartitionSchema == fileIndex.partitionSchema

  override def filter(predicates: Array[Predicate]): Unit = {
    val (partition, others) =
      predicates.partition(_.isInstanceOf[PartitionPredicate])
    if (readsAllPartitionColumns) {
      partitionPredicates ++= partition.map(_.asInstanceOf[PartitionPredicate])
    }
    super.filter(others)
  }

  override def pushedPredicates(): Array[Predicate] =
    runtimePredicates ++ partitionPredicates

  override protected def filtersFiles: Boolean = partitionPredicates.nonEmpty

  override protected def acceptsFile(file: PartitionedFile): Boolean =
    partitionPredicates.forall(_.eval(file.partitionValues))
}
