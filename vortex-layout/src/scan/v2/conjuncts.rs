// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::analysis::referenced_field_paths;
use vortex_array::expr::bound::and as bound_and;
use vortex_error::VortexResult;

/// Merges conjuncts that read a common field back into one conjunct each.
///
/// Evaluating two conjuncts separately reads and decodes their shared columns twice, so the
/// conjuncts are grouped by the fields they share, transitively: in `a > 5 AND a < b AND b = 2`
/// all three read a field another one reads, so they become one conjunct. Groups keep the order of
/// their first member, and members keep their order within a group.
pub(super) fn group_conjuncts(conjuncts: &[BoundExpression]) -> VortexResult<Vec<BoundExpression>> {
    if conjuncts.len() < 2 {
        return Ok(conjuncts.to_vec());
    }
    let fields = conjuncts
        .iter()
        .map(|conjunct| Ok(referenced_field_paths(conjunct)?.into_iter().collect_vec()))
        .collect::<VortexResult<Vec<Vec<FieldPath>>>>()?;

    let mut sets = DisjointSets::new(conjuncts.len());
    for (index, paths) in fields.iter().enumerate() {
        for (other, other_paths) in fields[..index].iter().enumerate() {
            let shared = paths
                .iter()
                .any(|path| other_paths.iter().any(|other| path.overlap(other)));
            if shared {
                sets.union(index, other);
            }
        }
    }

    // Every set is rooted at its lowest member, so filling groups in index order keeps both the
    // groups and their members in input order.
    let mut groups = vec![Vec::new(); conjuncts.len()];
    for (index, conjunct) in conjuncts.iter().enumerate() {
        groups[sets.find(index)].push(conjunct.clone());
    }
    Ok(groups
        .into_iter()
        .filter_map(|group| group.into_iter().reduce(bound_and))
        .collect())
}

/// Union-find over conjunct indices, where each set is rooted at its lowest member.
struct DisjointSets(Vec<usize>);

impl DisjointSets {
    fn new(len: usize) -> Self {
        Self((0..len).collect())
    }

    fn find(&mut self, mut index: usize) -> usize {
        while self.0[index] != index {
            self.0[index] = self.0[self.0[index]];
            index = self.0[index];
        }
        index
    }

    fn union(&mut self, lhs: usize, rhs: usize) {
        let (lhs, rhs) = (self.find(lhs), self.find(rhs));
        if lhs < rhs {
            self.0[rhs] = lhs;
        } else {
            self.0[lhs] = rhs;
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::Expression;
    use vortex_array::expr::and;
    use vortex_array::expr::col;
    use vortex_array::expr::eq;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_error::VortexResult;

    use super::group_conjuncts;
    use crate::scan::filter::FilterExpr;

    fn dtype() -> DType {
        let i32 = DType::Primitive(PType::I32, NonNullable);
        DType::Struct(
            StructFields::from_iter([("a", i32.clone()), ("b", i32.clone()), ("c", i32)]),
            NonNullable,
        )
    }

    fn grouped(filter: Expression) -> VortexResult<Vec<String>> {
        let conjuncts = FilterExpr::new(filter.bind(&dtype())?).conjuncts().to_vec();
        Ok(group_conjuncts(&conjuncts)?
            .iter()
            .map(ToString::to_string)
            .collect())
    }

    #[test]
    fn groups_conjuncts_sharing_a_field_transitively() -> VortexResult<()> {
        let filter = and(
            and(gt(col("a"), lit(5)), lt(col("a"), col("b"))),
            and(eq(col("c"), lit(1)), eq(col("b"), lit(2))),
        );
        let groups = grouped(filter)?;
        assert_eq!(groups.len(), 2, "{groups:?}");
        assert!(
            groups[0].contains("$.a") && groups[0].contains("$.b"),
            "{groups:?}"
        );
        assert!(
            groups[1].contains("$.c") && !groups[1].contains("$.a"),
            "{groups:?}"
        );
        Ok(())
    }

    #[test]
    fn keeps_disjoint_conjuncts_apart() -> VortexResult<()> {
        let groups = grouped(and(gt(col("a"), lit(5)), lt(col("b"), lit(3))))?;
        assert_eq!(groups.len(), 2, "{groups:?}");
        Ok(())
    }
}
