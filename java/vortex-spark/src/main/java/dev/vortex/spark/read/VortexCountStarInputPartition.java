// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import org.apache.spark.sql.connector.read.InputPartition;

/**
 * Input partition for a pushed-down {@code COUNT(*)}: a single Vortex file whose footer row count is read on an
 * executor.
 *
 * <p>Carrying exactly one file per partition is what makes the count exact: a single-file
 * {@link dev.vortex.api.DataSource} opens its only file eagerly, so its row count comes straight from the footer rather
 * than the multi-file extrapolation documented on {@link VortexScan#estimateStatistics()}.
 *
 * @param path the resolved {@code .vortex} file path
 */
public record VortexCountStarInputPartition(String path) implements InputPartition {}
