// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark

import dev.vortex.spark.read.{NestedDynamicPruning, ScanColumnAlignment}
import org.apache.spark.sql.{
  SparkSessionExtensions,
  SparkSessionExtensionsProvider
}
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.catalyst.rules.Rule

/** Spark session extensions for Vortex. Apply them with
  * `spark.sql.extensions=dev.vortex.spark.VortexSparkSessionExtensions`.
  *
  * With the extensions, dynamic pruning filters Vortex scans on data columns
  * too, not only on partition columns. The extensions add the
  * `NestedDynamicPruning` rule, which stops these filters from nesting into
  * each other.
  *
  * The extensions also let Spark merge subqueries that scan the same Vortex
  * table with the same filters but different columns. See
  * `ScanColumnAlignment`.
  */
final class VortexSparkSessionExtensions
    extends SparkSessionExtensionsProvider {

  override def apply(extensions: SparkSessionExtensions): Unit = {
    extensions.injectPostHocResolutionRule { session =>
      // No injection point runs after Spark's own dynamic pruning and subquery
      // merging, but the user provided optimizations do. Spark builds the
      // analyzer of a session before it optimizes a query of that session, so
      // the rules are in time.
      NestedDynamicPruning.install(session)
      ScanColumnAlignment.install(session)
      VortexSparkSessionExtensions.InstallNestedDynamicPruning
    }

    // The rules before cost-based optimization run after Spark builds the scans
    // of subqueries, and before Spark merges subqueries.
    extensions.injectPreCBORule(_ => ScanColumnAlignment.Widen)
  }
}

private object VortexSparkSessionExtensions {

  /** The rule Spark asks the extension for. It changes no plan. */
  private object InstallNestedDynamicPruning extends Rule[LogicalPlan] {
    override def apply(plan: LogicalPlan): LogicalPlan = plan
  }
}
