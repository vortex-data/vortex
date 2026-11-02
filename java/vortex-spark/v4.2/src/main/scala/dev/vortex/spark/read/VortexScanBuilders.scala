// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.PartitioningAwareFileIndex
import org.apache.spark.sql.types.StructType

import dev.vortex.spark.VortexOptions

/** Creates the scan builder for Spark 4.2, with variant extraction pushdown, block
  * sampling, and partition predicates.
  */
object VortexScanBuilders {
  def create(
      sparkSession: SparkSession,
      fileIndex: PartitioningAwareFileIndex,
      dataSchema: StructType,
      options: VortexOptions
  ): VortexScanBuilder =
    new VortexScanBuilder42(sparkSession, fileIndex, dataSchema, options)
}
