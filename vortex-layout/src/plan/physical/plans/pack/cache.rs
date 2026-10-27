// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::collections::VecDeque;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use parking_lot::Mutex;
use rustc_hash::FxHasher;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::scalar_fn::fns::pack::Pack as PackFn;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::descendent_bound_annotations;
use vortex_expr::make_bound_free_field_annotator;
use vortex_expr::transform::BoundPartitionedExpr;
use vortex_expr::transform::partition_bound_annotations;
use vortex_session::SessionVar;

use super::expand_struct_root;
use super::is_identity_expression;
use super::rewrite_partition_root;
use super::step_into_struct_field;

const MAX_ENTRIES: usize = 64;
const MAX_NODES: usize = 256;

#[derive(Debug)]
pub(super) struct Partitioned {
    pub(super) expanded: BoundExpression,
    pub(super) referenced_fields: Vec<FieldName>,
    pub(super) partitions: BoundPartitionedExpr<FieldName>,
    pub(super) lowered: Option<Lowered>,
}

#[derive(Debug)]
pub(super) struct Lowered {
    pub(super) fields: Vec<LoweredField>,
    pub(super) residual: BoundExpression,
    pub(super) fields_changed: bool,
}

#[derive(Debug)]
pub(super) struct LoweredField {
    pub(super) index: usize,
    pub(super) expression: BoundExpression,
    pub(super) identity: bool,
}

#[derive(Debug)]
struct Entry {
    hash: u64,
    expression: BoundExpression,
    fields: StructFields,
    result: Arc<Partitioned>,
}

/// Pure expression rewrites shared across file scans, without retaining file plans or values.
#[derive(Clone, Debug, Default)]
pub(super) struct PartitionCache(Arc<Mutex<VecDeque<Entry>>>);

impl SessionVar for PartitionCache {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl PartitionCache {
    pub(super) fn partition(
        &self,
        expression: &BoundExpression,
        fields: &StructFields,
    ) -> VortexResult<Arc<Partitioned>> {
        let Some(hash) = fingerprint(expression) else {
            return partition(expression, fields).map(Arc::new);
        };
        let mut entries = self.0.lock();
        if let Some(entry) = entries.iter().find(|entry| {
            entry.hash == hash && entry.fields == *fields && entry.expression == *expression
        }) {
            return Ok(Arc::clone(&entry.result));
        }
        // Partitioning is a pure rewrite. Holding this lock avoids concurrent scans repeating
        // the same allocation-heavy work; it never waits for file IO or invokes an optimizer.
        let result = Arc::new(partition(expression, fields)?);
        if fingerprint(&result.expanded).is_some() {
            if entries.len() == MAX_ENTRIES {
                entries.pop_front();
            }
            entries.push_back(Entry {
                hash,
                expression: expression.clone(),
                fields: fields.clone(),
                result: Arc::clone(&result),
            });
        }
        Ok(result)
    }
}

fn partition(expression: &BoundExpression, fields: &StructFields) -> VortexResult<Partitioned> {
    let expanded = expand_struct_root(expression.clone(), fields)?;
    let annotations =
        descendent_bound_annotations(&expanded, make_bound_free_field_annotator(fields));
    let referenced_fields = annotations
        .get(&ExactBoundExpr(expanded.clone()))
        .vortex_expect("Bound expression missing free-field annotations")
        .iter()
        .cloned()
        .collect();
    let partitions = partition_bound_annotations(expanded.clone(), annotations)?;
    let lowered = if partitions.partition_names.len() > 1 {
        Some(lower_partitions(&partitions, fields)?)
    } else {
        None
    };
    Ok(Partitioned {
        lowered,
        expanded,
        referenced_fields,
        partitions,
    })
}

fn lower_partitions(
    partitioned: &BoundPartitionedExpr<FieldName>,
    fields: &StructFields,
) -> VortexResult<Lowered> {
    let mut collapsed = Vec::with_capacity(partitioned.partitions.len());
    let mut lowered = Vec::with_capacity(partitioned.partitions.len());
    for (name, partition) in partitioned.partition_names.iter().zip(partitioned.partitions.iter()) {
        let index = fields.find(name).ok_or_else(|| {
            vortex_err!("Struct expression references unknown field '{name}'")
        })?;
        let dtype = fields.field(name).vortex_expect("Partition field exists");
        let expression = if let Some(pack) = partition
            .as_scalar()
            .and_then(|scalar_fn| scalar_fn.as_opt::<PackFn>())
            && partition.children().len() == 1
        {
            let value_name = pack.names.get(0).ok_or_else(|| {
                vortex_err!("Struct expression partition pack is empty")
            })?;
            collapsed.push((name.clone(), value_name.clone()));
            partition.children()[0].clone()
        } else {
            partition.clone()
        };
        let expression = step_into_struct_field(expression, name, dtype.clone())?;
        let identity = is_identity_expression(&expression, &dtype)?;
        lowered.push(LoweredField {
            index,
            expression,
            identity,
        });
    }
    lowered.sort_by_key(|field| field.index);
    let fields_changed = lowered.len() != fields.nfields()
        || lowered.iter().any(|field| !field.identity);
    let root_dtype = DType::Struct(
        StructFields::from_iter(lowered.iter().map(|field| {
            (fields.names()[field.index].clone(), field.expression.dtype().clone())
        })),
        Nullability::NonNullable,
    );
    let residual = rewrite_partition_root(partitioned.root.clone(), root_dtype, &collapsed)?;
    Ok(Lowered {
        fields: lowered,
        residual,
        fields_changed,
    })
}

fn fingerprint(expression: &BoundExpression) -> Option<u64> {
    let mut remaining = MAX_NODES;
    let mut hasher = FxHasher::default();
    hash_expression(expression, &mut remaining, &mut hasher)?;
    Some(hasher.finish())
}

fn hash_expression(
    expression: &BoundExpression,
    remaining: &mut usize,
    hasher: &mut FxHasher,
) -> Option<()> {
    *remaining = remaining.checked_sub(1)?;
    if let Some(function) = expression.as_scalar() {
        hasher.write_u8(1);
        function.hash(hasher);
        expression.children().len().hash(hasher);
        for child in expression.children() {
            hash_expression(child, remaining, hasher)?;
        }
    } else {
        hasher.write_u8(0);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;

    use super::*;

    fn fields(name: &str) -> StructFields {
        StructFields::from_iter([(name.into(), DType::Bool(Nullability::NonNullable))])
    }

    #[test]
    fn independent_bindings_share_partitions() -> VortexResult<()> {
        let cache = PartitionCache::default();
        let fields = fields("a");
        let dtype = DType::Struct(fields.clone(), Nullability::NonNullable);
        let first = cache.partition(&root().bind(&dtype)?, &fields)?;
        let second = cache.partition(&root().bind(&dtype)?, &fields)?;
        assert!(Arc::ptr_eq(&first, &second));
        Ok(())
    }

    #[test]
    fn constant_partitions_keep_scopes_separate() -> VortexResult<()> {
        let cache = PartitionCache::default();
        let a = fields("a");
        let b = fields("b");
        let expression = lit(1_i32).bind(&DType::Struct(a.clone(), Nullability::NonNullable))?;
        let first = cache.partition(&expression, &a)?;
        let second = cache.partition(&expression, &b)?;
        assert!(!Arc::ptr_eq(&first, &second));
        Ok(())
    }

    #[test]
    fn retained_entries_are_bounded() -> VortexResult<()> {
        let cache = PartitionCache::default();
        let fields = fields("a");
        let dtype = DType::Struct(fields.clone(), Nullability::NonNullable);
        for value in 0..=MAX_ENTRIES {
            cache.partition(&lit(i32::try_from(value)?).bind(&dtype)?, &fields)?;
        }
        assert_eq!(cache.0.lock().len(), MAX_ENTRIES);
        Ok(())
    }
}
