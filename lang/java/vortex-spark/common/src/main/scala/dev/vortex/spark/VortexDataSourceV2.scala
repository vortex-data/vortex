// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark

import org.apache.spark.sql.connector.catalog.Table
import org.apache.spark.sql.execution.datasources.FileFormat
import org.apache.spark.sql.execution.datasources.v2.FileDataSourceV2
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

/** Spark file data source for Vortex files. */
final class VortexDataSourceV2 extends FileDataSourceV2 {
  override def fallbackFileFormat: Class[_ <: FileFormat] =
    classOf[VortexFileFormat]

  override def shortName(): String = "vortex"

  override protected def getTable(options: CaseInsensitiveStringMap): Table = {
    val paths = getPaths(options)
    new VortexTable(
      getTableName(options, paths),
      sparkSession,
      getOptionsWithoutPaths(options),
      paths,
      None,
      fallbackFileFormat
    )
  }

  override protected def getTable(
      options: CaseInsensitiveStringMap,
      schema: StructType
  ): Table = {
    val paths = getPaths(options)
    new VortexTable(
      getTableName(options, paths),
      sparkSession,
      getOptionsWithoutPaths(options),
      paths,
      Some(schema),
      fallbackFileFormat
    )
  }
}
