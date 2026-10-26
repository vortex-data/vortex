// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::lit;
use vortex_array::expr::bound::mask;
use vortex_array::expr::bound::not;
use vortex_array::expr::bound::pack;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::scalar_fn::fns::select::Select;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::partition::replacement_scope;
use crate::partition::slot_read;
use crate::partition_session::FieldIndex;
use crate::partition_session::NodeTouches;
use crate::partition_session::PartitionRule;
use crate::partition_session::PartitionSession;
use crate::plan::physical::Pack;
use crate::plan::physical::PlanVTable;

pub(crate) const VALIDITY_SLOT: FieldIndex = FieldIndex::MAX;

pub(crate) fn register_struct_partition_rules(session: &PartitionSession) {
    session.register_root(Pack.id(), &StructRootRule);
    session.register(Pack.id(), GetItem.id(), &StructGetItemRule);
    session.register(Pack.id(), Select.id(), &StructSelectRule);
    session.register(
        Pack.id(),
        IsNotNull.id(),
        &StructNullCheckRule { negated: false },
    );
    session.register(
        Pack.id(),
        IsNull.id(),
        &StructNullCheckRule { negated: true },
    );
}

fn struct_fields(scope: &DType) -> VortexResult<&StructFields> {
    scope
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("Struct partition rules require a struct scope"))
}

fn field_dtype(fields: &StructFields, index: FieldIndex) -> VortexResult<DType> {
    fields
        .field_by_index(index)
        .ok_or_else(|| vortex_err!("No field {index} in the struct scope"))
}

fn validity_dtype() -> DType {
    DType::Bool(Nullability::NonNullable)
}

fn parent_of_root_scope(node: &BoundExpression) -> Option<&DType> {
    node.children()
        .first()
        .filter(|child| child.is_root())
        .map(BoundExpression::dtype)
}

fn packed_replacement(names: FieldNames, dtypes: &[DType], nullable: bool) -> BoundExpression {
    let scope = replacement_scope(dtypes);
    let offset = usize::from(nullable);
    let packed = pack(
        names
            .iter()
            .enumerate()
            .map(|(position, name)| (name.clone(), slot_read(&scope, position + offset))),
        Nullability::NonNullable,
    );
    if nullable {
        mask(packed, slot_read(&scope, 0))
    } else {
        packed
    }
}

fn select_fields(
    scope: &DType,
    fields: &StructFields,
    names: FieldNames,
    slots: Vec<FieldIndex>,
) -> VortexResult<NodeTouches> {
    if !scope.is_nullable() {
        if let [index] = slots.as_slice() {
            let field = BoundExpression::new_root(field_dtype(fields, *index)?);
            let packed = pack([(names[0].clone(), field)], Nullability::NonNullable);
            return Ok(NodeTouches {
                slots,
                replacement: packed,
            });
        }
        if slots.is_empty() {
            let replacement = pack(
                Vec::<(FieldName, BoundExpression)>::new(),
                Nullability::NonNullable,
            );
            return Ok(NodeTouches { slots, replacement });
        }
    }

    let mut out_slots = Vec::with_capacity(slots.len() + 1);
    let mut out_dtypes = Vec::with_capacity(slots.len() + 1);
    if scope.is_nullable() {
        out_slots.push(VALIDITY_SLOT);
        out_dtypes.push(validity_dtype());
    }
    for slot in slots {
        out_slots.push(slot);
        out_dtypes.push(field_dtype(fields, slot)?);
    }
    let replacement = packed_replacement(names, &out_dtypes, scope.is_nullable());
    Ok(NodeTouches {
        slots: out_slots,
        replacement,
    })
}

/// Struct { ... } as $ is select(... + optional validity)
struct StructRootRule;
impl PartitionRule for StructRootRule {
    fn apply(&self, node: &BoundExpression) -> VortexResult<Option<NodeTouches>> {
        let scope = node.dtype();
        let fields = struct_fields(scope)?;
        select_fields(
            scope,
            fields,
            fields.names().clone(),
            (0..fields.nfields()).collect(),
        )
        .map(Some)
    }
}

/// $.x is replaced with $ (non-nullable) or mask($.x, $.validity)
struct StructGetItemRule;
impl PartitionRule for StructGetItemRule {
    fn apply(&self, node: &BoundExpression) -> VortexResult<Option<NodeTouches>> {
        let Some(scope) = parent_of_root_scope(node) else {
            return Ok(None);
        };
        let field_name = node.as_::<GetItem>();
        let fields = struct_fields(scope)?;
        let index = fields
            .find(field_name)
            .ok_or_else(|| vortex_err!("Field {field_name} not found in the struct scope"))?;
        let dtype = field_dtype(fields, index)?;
        if !scope.is_nullable() {
            return Ok(Some(NodeTouches {
                slots: vec![index],
                replacement: BoundExpression::new_root(dtype),
            }));
        }

        let scope = replacement_scope(&[validity_dtype(), dtype]);
        let replacement = mask(slot_read(&scope, 1), slot_read(&scope, 0));
        Ok(Some(NodeTouches {
            slots: vec![VALIDITY_SLOT, index],
            replacement,
        }))
    }
}

/// select(..., $) is replaced with select(... + validity, $)
struct StructSelectRule;
impl PartitionRule for StructSelectRule {
    fn apply(&self, node: &BoundExpression) -> VortexResult<Option<NodeTouches>> {
        let Some(scope) = parent_of_root_scope(node) else {
            return Ok(None);
        };
        let selection = node.as_::<Select>();
        let fields = struct_fields(scope)?;
        let names = selection.normalize_to_included_fields(fields.names())?;
        let indices = names
            .iter()
            .map(|name| {
                fields
                    .find(name)
                    .ok_or_else(|| vortex_err!("field {name} not found"))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        select_fields(scope, fields, names, indices).map(Some)
    }
}

struct StructNullCheckRule {
    negated: bool,
}
/// is[_not]_null($) is replaced with lit() or $.validity
impl PartitionRule for StructNullCheckRule {
    fn apply(&self, node: &BoundExpression) -> VortexResult<Option<NodeTouches>> {
        let Some(scope) = parent_of_root_scope(node) else {
            return Ok(None);
        };
        Ok(Some(if scope.is_nullable() {
            let read = BoundExpression::new_root(validity_dtype());
            let slots = vec![VALIDITY_SLOT];
            let replacement = if self.negated { not(read) } else { read };
            NodeTouches { slots, replacement }
        } else {
            let slots = vec![];
            let replacement = lit(!self.negated);
            NodeTouches { slots, replacement }
        }))
    }
}

#[cfg(test)]
mod tests {
    use insta::assert_snapshot;
    use vortex_array::dtype::PType;
    use vortex_array::expr as unbound;
    use vortex_array::expr::Expression;
    use vortex_array::expr::bound::get_item;
    use vortex_array::expr::bound::is_not_null;

    use super::*;
    use crate::partition::PartitionedExpression;
    use crate::partition::Partitions;
    use crate::partition::partition_expression;
    use crate::partition::scope_of;
    use crate::plan::physical::Pack;
    use crate::plan::physical::PlanVTable;
    use crate::test::SESSION;

    fn i32() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    fn struct_of(fields: &[(&str, DType)], nullability: Nullability) -> DType {
        DType::Struct(
            StructFields::from_iter(fields.iter().map(|(name, dtype)| (*name, dtype.clone()))),
            nullability,
        )
    }

    fn two_field_struct() -> DType {
        struct_of(&[("a", i32()), ("b", i32())], Nullability::NonNullable)
    }

    fn fmt_slot(slot: Option<FieldIndex>) -> String {
        match slot {
            None => "none".to_string(),
            Some(FieldIndex::MAX) => "validity".to_string(),
            Some(slot) => slot.to_string(),
        }
    }

    fn line(label: &str, expr: &BoundExpression) -> String {
        let scope = scope_of(expr)
            .map(|dtype| dtype.to_string())
            .unwrap_or_else(|| "-".to_string());
        format!("{label} scope={scope}: {expr}\n")
    }

    fn snap(partitioning: PartitionedExpression) -> String {
        match partitioning {
            PartitionedExpression::One(slot, expr) => {
                line(&format!("mono {}", fmt_slot(slot)), &expr)
            }
            PartitionedExpression::Many(Partitions { root, partitions }) => {
                let mut out = String::new();
                for partition in &partitions {
                    out.push_str(&line(
                        &format!("partition {}", fmt_slot(Some(partition.slot))),
                        &partition.expr,
                    ));
                }
                out.push_str(&line("root", &root));
                out
            }
        }
    }

    fn partition(expr: Expression, scope: &DType) -> VortexResult<String> {
        Ok(snap(partition_expression(
            &expr.bind(scope)?,
            Pack.id(),
            &SESSION,
        )?))
    }

    #[test]
    fn is_not_null_of_non_nullable() -> VortexResult<()> {
        let root = BoundExpression::new_root(two_field_struct());
        let expr = pack(
            [
                ("v", is_not_null(root.clone())),
                ("a", get_item("a", root.clone())),
                ("b", get_item("b", root)),
            ],
            Nullability::NonNullable,
        );
        assert_snapshot!(snap(partition_expression(&expr, Pack.id(), &SESSION)?), @"
        partition 0 scope=i32: $
        partition 1 scope=i32: $
        root scope={0=i32, 1=i32}: pack(v: true, a: $.0, b: $.1)
        ");
        Ok(())
    }

    #[test]
    fn equal_cuts() -> VortexResult<()> {
        let root = BoundExpression::new_root(two_field_struct());
        let a = get_item("a", root.clone());
        let b = get_item("b", root);
        let expr = pack(
            [("x", a.clone()), ("y", b), ("z", a)],
            Nullability::NonNullable,
        );
        assert_snapshot!(snap(partition_expression(&expr, Pack.id(), &SESSION)?), @"
        partition 0 scope=i32: $
        partition 1 scope=i32: $
        root scope={0=i32, 1=i32}: pack(x: $.0, y: $.1, z: $.0)
        ");
        Ok(())
    }

    #[test]
    fn is_not_null_one_partition() -> VortexResult<()> {
        let expr = unbound::is_not_null(unbound::get_item("a", unbound::root()));
        assert_snapshot!(partition(expr, &two_field_struct())?, @"mono 0 scope=i32: is_not_null($)");
        Ok(())
    }

    #[test]
    fn is_not_null_two_fields() -> VortexResult<()> {
        let scope = struct_of(&[("a", i32()), ("b", i32())], Nullability::Nullable);
        let expr = unbound::is_not_null(unbound::get_item("a", unbound::root()));
        assert_snapshot!(partition(expr, &scope)?, @"
        partition validity scope=bool: $
        partition 0 scope=i32: $
        root scope={0=bool, 1=i32}: $.0
        ");
        Ok(())
    }

    #[test]
    fn test_pack() -> VortexResult<()> {
        let expr = unbound::get_item(
            "a",
            unbound::get_item(
                "a",
                unbound::pack([("a", unbound::root())], Nullability::NonNullable),
            ),
        );
        assert_snapshot!(partition(expr, &two_field_struct())?, @"
        partition 0 scope=i32: $
        partition 1 scope=i32: $
        root scope={0=i32, 1=i32}: $.0
        ");
        Ok(())
    }

    #[test]
    fn test_ab_greater_c() -> VortexResult<()> {
        let inner = struct_of(&[("b", i32())], Nullability::NonNullable);
        let scope = struct_of(&[("a", inner), ("c", i32())], Nullability::NonNullable);
        let expr = unbound::gt(
            unbound::get_item("b", unbound::get_item("a", unbound::root())),
            unbound::get_item("c", unbound::root()),
        );
        assert_snapshot!(partition(expr, &scope)?, @"
        partition 0 scope={b=i32}: $.b
        partition 1 scope=i32: $
        root scope={0=i32, 1=i32}: ($.0 > $.1)
        ");
        Ok(())
    }

    #[test]
    fn test_x_plus_1() -> VortexResult<()> {
        let scope = struct_of(&[("x", i32()), ("y", i32())], Nullability::NonNullable);
        let expr = unbound::gt(
            unbound::checked_add(unbound::get_item("x", unbound::root()), unbound::lit(1i32)),
            unbound::get_item("y", unbound::root()),
        );
        assert_snapshot!(partition(expr, &scope)?, @"
        partition 0 scope=i32: ($ + 1i32)
        partition 1 scope=i32: $
        root scope={0=i32, 1=i32}: ($.0 > $.1)
        ");
        Ok(())
    }

    #[test]
    fn test_aa_greater_ba() -> VortexResult<()> {
        let inner = struct_of(&[("a", i32())], Nullability::NonNullable);
        let scope = struct_of(
            &[("a", inner.clone()), ("b", inner)],
            Nullability::NonNullable,
        );
        let expr = unbound::gt(
            unbound::get_item("a", unbound::get_item("a", unbound::root())),
            unbound::get_item("a", unbound::get_item("b", unbound::root())),
        );
        assert_snapshot!(partition(expr, &scope)?, @"
        partition 0 scope={a=i32}: $.a
        partition 1 scope={a=i32}: $.a
        root scope={0=i32, 1=i32}: ($.0 > $.1)
        ");
        Ok(())
    }

    #[test]
    fn test_nested_and() -> VortexResult<()> {
        let a = unbound::get_item("a", unbound::root());
        let b = unbound::get_item("b", unbound::root());
        let expr = unbound::and(
            unbound::and(
                unbound::gt(a.clone(), unbound::lit(5i32)),
                unbound::gt(b, unbound::lit(5i32)),
            ),
            unbound::lt(a, unbound::lit(1i32)),
        );
        assert_snapshot!(partition(expr, &two_field_struct())?, @"
        partition 0 scope=i32: pack(0: ($ > 5i32), 1: ($ < 1i32))
        partition 1 scope=i32: ($ > 5i32)
        root scope={0={0=bool, 1=bool}, 1=bool}: (($.0.0 and $.1) and $.0.1)
        ");
        Ok(())
    }

    #[test]
    fn test_duplicate_predicate() -> VortexResult<()> {
        let a = unbound::get_item("a", unbound::root());
        let b = unbound::get_item("b", unbound::root());
        let gt5_a = unbound::gt(a, unbound::lit(5i32));
        let expr = unbound::and(
            unbound::and(gt5_a.clone(), unbound::gt(b, unbound::lit(5i32))),
            gt5_a,
        );
        assert_snapshot!(partition(expr, &two_field_struct())?, @"
        partition 0 scope=i32: ($ > 5i32)
        partition 1 scope=i32: ($ > 5i32)
        root scope={0=bool, 1=bool}: (($.0 and $.1) and $.0)
        ");
        Ok(())
    }

    #[test]
    fn test_abc_greater_abd() -> VortexResult<()> {
        let leaf = struct_of(&[("c", i32()), ("d", i32())], Nullability::NonNullable);
        let middle = struct_of(&[("b", leaf)], Nullability::NonNullable);
        let scope = struct_of(&[("a", middle)], Nullability::NonNullable);
        let expr = unbound::gt(
            unbound::get_item(
                "c",
                unbound::get_item("b", unbound::get_item("a", unbound::root())),
            ),
            unbound::get_item(
                "d",
                unbound::get_item("b", unbound::get_item("a", unbound::root())),
            ),
        );
        assert_snapshot!(partition(expr, &scope)?, @"mono 0 scope={b={c=i32, d=i32}}: ($.b.c > $.b.d)");
        Ok(())
    }
}
