// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{
  And,
  DynamicPruningSubquery,
  Expression,
  PredicateHelper
}
import org.apache.spark.sql.catalyst.plans.logical.{Filter, LogicalPlan}
import org.apache.spark.sql.catalyst.rules.Rule

/** Drops each dynamic pruning filter whose filtering plan holds dynamic pruning
  * filters of its own.
  *
  * Spark adds one pruning subquery per join that can prune a scan, holding the
  * plan of the join's other side. When that side holds pruned scans too, as a
  * common table expression joined with itself does, each subquery copies their
  * subqueries, and the physical plan grows exponentially with the depth of the
  * joins. A filtering side that big is no broadcast dimension either, so its
  * pruning would not run.
  *
  * A dynamic pruning filter only skips rows the join drops anyway, so dropping
  * the filter leaves the result as it is.
  *
  * [[dev.vortex.spark.VortexSparkSessionExtensions]] installs the rule. The
  * rule is public so that a different Spark integration that reads Vortex
  * files, such as a table format, can install it in its own session extensions.
  */
object NestedDynamicPruning extends Rule[LogicalPlan] with PredicateHelper {

  /** Adds the rule to the user provided optimizations of `session`, once.
    *
    * Spark runs these rules after its own dynamic pruning, and reads them each
    * time it optimizes a query.
    */
  def install(session: SparkSession): Unit = {
    val experimental = session.experimental
    experimental.synchronized {
      if (!experimental.extraOptimizations.contains(this)) {
        experimental.extraOptimizations =
          experimental.extraOptimizations :+ this
      }
    }
  }

  /** Whether `session` runs the rule. */
  def installed(session: SparkSession): Boolean =
    session.experimental.extraOptimizations.contains(this)

  override def apply(plan: LogicalPlan): LogicalPlan =
    plan.transformUp {
      case Filter(condition, child) if condition.exists(isNested) =>
        splitConjunctivePredicates(condition)
          .filterNot(isNested)
          .reduceOption(And)
          .map(Filter(_, child))
          .getOrElse(child)
    }

  private def isNested(expression: Expression): Boolean =
    expression match {
      case subquery: DynamicPruningSubquery =>
        subquery.buildQuery.exists(
          _.expressions.exists(_.exists(_.isInstanceOf[DynamicPruningSubquery]))
        )
      case _ => false
    }
}
