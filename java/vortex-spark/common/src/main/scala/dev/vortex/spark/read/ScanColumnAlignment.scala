// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.catalyst.plans.logical.{
  Aggregate,
  LogicalPlan,
  Project
}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.execution.datasources.v2.DataSourceV2ScanRelation
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.StructType

import dev.vortex.spark.VortexTable

/** Lets Spark merge subqueries that scan the same Vortex table with the same
  * filters but different columns, as the scalar subqueries of TPC-DS q9 do.
  *
  * Spark merges such subqueries into one scan only when their scans are equal.
  * A DataSource V2 scan holds the columns that its subquery reads, so the scan
  * of `count(*)` and the scan of `avg(x)` are different, and Spark scans the
  * table once for each subquery. A file source of DataSource V1 keeps all the
  * columns in the logical plan and does not have this problem.
  *
  * [[ScanColumnAlignment.Widen]] runs before Spark merges subqueries. It finds
  * the Vortex scans in subqueries that are equal except for their columns, and
  * changes each of them to read the union of their columns. A projection above
  * each scan keeps the columns of the subquery as they were. Spark can then
  * merge the scans.
  *
  * [[ScanColumnAlignment.Narrow]] runs after Spark merges subqueries. A scan
  * that Spark did not merge reads columns that no operator uses. The rule
  * changes such a scan back to the scan that it was.
  */
private[spark] object ScanColumnAlignment {

  /** Adds [[Narrow]] to the user provided optimizations of `session`, once.
    * Spark runs these rules after it merges subqueries.
    */
  def install(session: SparkSession): Unit = {
    val experimental = session.experimental
    experimental.synchronized {
      if (!experimental.extraOptimizations.contains(Narrow)) {
        experimental.extraOptimizations =
          experimental.extraOptimizations :+ Narrow
      }
    }
  }

  /** The scan before [[Widen]] changed its columns. */
  private val Original =
    TreeNodeTag[DataSourceV2ScanRelation]("vortex.scanBeforeWiden")

  /** Changes the Vortex scans in subqueries that are equal except for their
    * columns to read the union of their columns.
    */
  private[spark] object Widen extends Rule[LogicalPlan] {

    override def apply(plan: LogicalPlan): LogicalPlan = {
      // The scans of the plan itself are not built yet at this step. The scans
      // of its subqueries are.
      val candidates = plan.collectWithSubqueries {
        case relation: DataSourceV2ScanRelation if isPlain(relation) =>
          relation
      }.distinct

      if (candidates.size < 2) {
        return plan
      }

      val replacements =
        groups(candidates).flatMap(widen).toMap[LogicalPlan, LogicalPlan]

      if (replacements.isEmpty) {
        plan
      } else {
        plan.transformWithSubqueries {
          case relation: DataSourceV2ScanRelation
              if replacements.contains(relation) =>
            replacements(relation)
        }
      }
    }

    /** Groups the scans that read the same files with the same pushdowns.
      *
      * A probe scan of each candidate reads all the columns of the table, so
      * the probes are equal exactly when the scans are equal except for their
      * columns.
      */
    private def groups(
        candidates: Seq[DataSourceV2ScanRelation]
    ): Seq[Seq[DataSourceV2ScanRelation]] = {
      val probes = candidates.flatMap(relation =>
        build(relation, relation.relation.schema).map(relation -> _)
      )

      probes
        .foldLeft(
          Vector.empty[(VortexScan, Vector[DataSourceV2ScanRelation])]
        ) { case (groups, (relation, probe)) =>
          groups.indexWhere(_._1 == probe) match {
            case -1    => groups :+ (probe -> Vector(relation))
            case index =>
              groups.updated(
                index,
                (groups(index)._1, groups(index)._2 :+ relation)
              )
          }
        }
        .map(_._2)
        .filter(_.size > 1)
    }

    /** Each scan of `group`, widened to the union of the columns of the group
      * below a projection to its own columns. None of them when a scan cannot
      * be widened.
      */
    private def widen(
        group: Seq[DataSourceV2ScanRelation]
    ): Seq[(DataSourceV2ScanRelation, LogicalPlan)] = {
      val readSchemas = group.map(_.scan.readSchema())

      if (readSchemas.distinct.size == 1) {
        return Seq.empty
      }

      val columns = union(readSchemas)

      if (columns.isEmpty) {
        return Seq.empty
      }

      val widened = group.flatMap(relation =>
        build(relation, columns.get).flatMap(scan =>
          outputOf(relation, scan).map(output => (relation, scan, output))
        )
      )

      val allEqual = widened.size == group.size &&
        widened.forall(_._2 == widened.head._2)

      if (!allEqual) {
        return Seq.empty
      }

      widened.map { case (relation, scan, output) =>
        val wide = relation.copy(scan = scan, output = output)
        wide.setTagValue(
          Original,
          relation.getTagValue(Original).getOrElse(relation)
        )
        relation -> Project(relation.output, wide)
      }
    }

    /** The union of the columns of `schemas`, or None when two of them read a
      * column at different types, as nested schema pruning can do.
      */
    private def union(schemas: Seq[StructType]): Option[StructType] = {
      val fields = schemas.flatMap(_.fields)
      val byName = fields.groupBy(field => key(field.name))
      val sameTypes = byName.values.forall(_.map(_.dataType).distinct.size == 1)

      if (sameTypes) {
        val names = fields.map(field => key(field.name)).distinct
        Some(StructType(names.map(name => byName(name).head)))
      } else {
        None
      }
    }

    /** The output of `relation` with `scan`: the attributes it has for the
      * columns it reads already, and new attributes for the other columns.
      */
    private def outputOf(
        relation: DataSourceV2ScanRelation,
        scan: VortexScan
    ): Option[Seq[AttributeReference]] = {
      val existing =
        relation.output.map(attribute => key(attribute.name) -> attribute).toMap
      val output = scan.readSchema().fields.toSeq.map { field =>
        existing.get(key(field.name)) match {
          case Some(attribute) if attribute.dataType == field.dataType =>
            Some(attribute)
          case Some(_) => None
          case None    =>
            Some(
              AttributeReference(
                field.name,
                field.dataType,
                field.nullable,
                field.metadata
              )()
            )
        }
      }

      val complete = output.forall(_.nonEmpty) &&
        relation.output.forall(attribute => output.flatten.contains(attribute))

      if (complete) Some(output.flatten) else None
    }
  }

  /** Changes each widened scan that Spark did not merge back to the scan that
    * it was.
    *
    * The rule only changes a scan below a projection or an aggregate that reads
    * none of the new columns. These operators do not pass the columns of the
    * scan on, so the change keeps the output of every other operator.
    */
  private[spark] object Narrow extends Rule[LogicalPlan] {

    override def apply(plan: LogicalPlan): LogicalPlan =
      plan.transformWithSubqueries {
        case parent @ (_: Project | _: Aggregate) if narrowable(parent) =>
          val original = parent.children.flatMap(_.getTagValue(Original))

          parent match {
            // The projection that Widen added above the scan.
            case project: Project
                if project.projectList == original.head.output =>
              original.head
            case _ => parent.withNewChildren(original)
          }
      }

    private def narrowable(parent: LogicalPlan): Boolean =
      parent.children match {
        case Seq(relation: DataSourceV2ScanRelation) =>
          relation
            .getTagValue(Original)
            .exists(original => parent.references.subsetOf(original.outputSet))
        case _ => false
      }
  }

  /** Whether `relation` is a Vortex scan of only columns and filters, which a
    * new scan builder can build again.
    */
  private def isPlain(relation: DataSourceV2ScanRelation): Boolean =
    relation.keyGroupedPartitioning.isEmpty && relation.ordering.forall(
      _.isEmpty
    ) &&
      relation.relation.table.isInstanceOf[VortexTable] &&
      (relation.scan match {
        case scan: VortexScan =>
          scan.pushedAggregation.isEmpty && scan.pushedLimit.isEmpty &&
          scan.pushedTopN.isEmpty && scan.pushedSample.isEmpty &&
          scan.pushedOffset.isEmpty && scan.variantColumns.isEmpty
        case _ => false
      })

  /** A scan with the filters of `relation` that reads `columns`, or None when
    * the builder does not give a plain Vortex scan.
    */
  private def build(
      relation: DataSourceV2ScanRelation,
      columns: StructType
  ): Option[VortexScan] = {
    val scan = relation.scan.asInstanceOf[VortexScan]
    val table = relation.relation.table.asInstanceOf[VortexTable]
    val builder = table.newScanBuilder(relation.relation.options)

    builder.pushFilters(scan.partitionFilters ++ scan.dataFilters)
    builder.pruneColumns(columns)

    builder.build() match {
      case built: VortexScan if isPlain(relation.copy(scan = built)) =>
        Some(built)
      case _ => None
    }
  }

  private def key(name: String): String =
    if (SQLConf.get.caseSensitiveAnalysis) name else name.toLowerCase
}
