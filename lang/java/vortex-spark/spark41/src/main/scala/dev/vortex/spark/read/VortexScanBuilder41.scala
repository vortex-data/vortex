// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.connector.read.{
  SupportsPushDownVariantExtractions,
  VariantExtraction
}
import org.apache.spark.sql.execution.datasources.PartitioningAwareFileIndex
import org.apache.spark.sql.types.{StructField, StructType}

import dev.vortex.spark.{SparkVariants, VortexOptions}

object VortexScanBuilder41 {

  /** Option turning on the pushdown of `variant_get` into the scan.
    *
    * Off by default: Spark 4.1 and 4.2 build a scan with pushed variant
    * extractions without pruning its columns, so every column of the table
    * is read. For unshredded variants, native extraction parses each variant
    * as Spark's own `variant_get` does, and is not faster.
    */
  val VariantPushdownOption = "vortex.variantPushdown"
}

/** The scan builder for Spark 4.1 and later, which adds variant extraction
  * pushdown.
  *
  * Each extraction reaches the reader as a struct field in place of its
  * variant column. Vortex extracts the variant at the field's path in the
  * file, and Spark's own `variant_get` cast turns it into the requested type,
  * so the values match Spark's exactly.
  */
class VortexScanBuilder41(
    sparkSession: SparkSession,
    fileIndex: PartitioningAwareFileIndex,
    dataSchema: StructType,
    options: VortexOptions
) extends VortexScanBuilder(sparkSession, fileIndex, dataSchema, options)
    with SupportsPushDownVariantExtractions {

  override def pushVariantExtractions(
      extractions: Array[VariantExtraction]
  ): Array[Boolean] = {
    val accepted = Array.fill(extractions.length)(false)
    if (
      !options.getBoolean(VortexScanBuilder41.VariantPushdownOption, false) ||
      aggregationPushed
    ) {
      return accepted
    }

    val rewritten = Map.newBuilder[String, StructType]
    extractions.zipWithIndex
      .groupBy { case (extraction, _) => extraction.columnName().toSeq }
      .foreach { case (path, group) =>
        // Spark replaces the whole variant column with the struct, so every extraction of a column is pushed or
        // none is. Variants nested in structs stay with Spark.
        val column =
          if (path.length == 1) {
            dataSchema.fields.find(field =>
              field.name == path.head && SparkVariants.isVariant(field.dataType)
            )
          } else {
            None
          }
        column.foreach { field =>
          // Spark numbers a column's extractions in the order it passes them, and names each field by that number.
          val extracted = new StructType(group.sortBy(_._2).zipWithIndex.map {
            case ((extraction, _), ordinal) =>
              StructField(
                ordinal.toString,
                extraction.expectedDataType(),
                nullable = true,
                extraction.metadata()
              )
          })
          if (SparkVariants.extractedFields(extracted).isPresent) {
            rewritten += field.name -> extracted
            group.foreach { case (_, index) => accepted(index) = true }
          }
        }
      }
    variantColumns = rewritten.result()
    accepted
  }
}
