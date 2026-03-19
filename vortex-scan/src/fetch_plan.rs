// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;

use itertools::Itertools;
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::scalar_fn::fns::root::Root;
use vortex_array::scalar_fn::fns::select::Select;
use vortex_error::VortexResult;
use vortex_layout::LayoutReader;
use vortex_layout::ProjectionFetchHint;

const IMMEDIATE_FIELD_ROW_BYTES_THRESHOLD: usize = 16;
pub(crate) const DEFERRED_WAIT_BUDGET_BYTES: usize = 8 << 20;
pub(crate) const DEFERRED_IN_FLIGHT_BUDGET_BYTES: usize = 16 << 20;
const DEFAULT_VARIABLE_ROW_BYTES: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MaterializationPlan {
    Monolithic {
        projected_row_bytes: usize,
        projection_aligned_splits: bool,
    },
    Deferred(DeferredMaterializationPlan),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeferredMaterializationPlan {
    final_fields: FieldNames,
    immediate_fields: FieldNames,
    deferred_groups: Vec<DeferredFieldGroup>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeferredFieldGroup {
    field_names: FieldNames,
    field_masks: Vec<FieldMask>,
    fetch_row_bytes: usize,
}

impl MaterializationPlan {
    pub(crate) fn from_projection(
        projection: &Expression,
        dtype: &DType,
        filter_present: bool,
        projection_field_mask: &[FieldMask],
        filter_field_names: &BTreeSet<FieldName>,
    ) -> Self {
        let projected_row_bytes = estimate_field_mask_row_bytes(dtype, projection_field_mask);
        let projection_aligned_splits =
            filter_present && projection_masks_include_wide_fields(dtype, projection_field_mask);
        if !filter_present {
            return Self::Monolithic {
                projected_row_bytes,
                projection_aligned_splits: false,
            };
        }

        let Some(final_fields) = simple_root_projection_fields(projection, dtype) else {
            return Self::Monolithic {
                projected_row_bytes,
                projection_aligned_splits,
            };
        };
        if final_fields.is_empty() || !final_fields.iter().all_unique() {
            return Self::Monolithic {
                projected_row_bytes,
                projection_aligned_splits,
            };
        }

        let Some(struct_fields) = dtype.as_struct_fields_opt() else {
            return Self::Monolithic {
                projected_row_bytes,
                projection_aligned_splits,
            };
        };
        if final_fields.len() == struct_fields.nfields()
            || final_fields.len().saturating_mul(2) >= struct_fields.nfields()
        {
            return Self::Monolithic {
                projected_row_bytes,
                projection_aligned_splits: true,
            };
        }

        let mut immediate = Vec::new();
        let mut deferred_groups = Vec::new();
        let mut immediate_carry_cost = 0usize;
        let mut deferred_carry_cost = 0usize;

        for name in final_fields.iter() {
            let Some(field_dtype) = struct_fields.field(name) else {
                return Self::Monolithic {
                    projected_row_bytes,
                    projection_aligned_splits,
                };
            };

            let carry_cost_bytes_per_row = estimate_dtype_row_bytes(&field_dtype);
            // Fields shared with the filter are already fetched during filter evaluation,
            // so keep them immediate to avoid double IO.
            if filter_field_names.contains(name) {
                immediate_carry_cost =
                    immediate_carry_cost.saturating_add(carry_cost_bytes_per_row);
                immediate.push(name.clone());
                continue;
            }
            if should_defer_field(&field_dtype, carry_cost_bytes_per_row) {
                deferred_carry_cost = deferred_carry_cost.saturating_add(carry_cost_bytes_per_row);
                deferred_groups.push(DeferredFieldGroup {
                    field_names: FieldNames::from([name.clone()]),
                    field_masks: vec![FieldMask::Prefix(FieldPath::from(Field::Name(
                        name.clone(),
                    )))],
                    fetch_row_bytes: carry_cost_bytes_per_row,
                });
            } else {
                immediate_carry_cost =
                    immediate_carry_cost.saturating_add(carry_cost_bytes_per_row);
                immediate.push(name.clone());
            }
        }

        if deferred_groups.is_empty() {
            return Self::Monolithic {
                projected_row_bytes,
                projection_aligned_splits,
            };
        }

        let total_carry_cost = immediate_carry_cost.saturating_add(deferred_carry_cost);
        if total_carry_cost == 0 || deferred_carry_cost.saturating_mul(2) < total_carry_cost {
            return Self::Monolithic {
                projected_row_bytes,
                projection_aligned_splits,
            };
        }

        Self::Deferred(DeferredMaterializationPlan {
            final_fields,
            immediate_fields: FieldNames::from(immediate),
            deferred_groups,
        })
    }

    pub(crate) fn fetch_hints(
        &self,
        reader: &dyn LayoutReader,
        projection_field_mask: &[FieldMask],
        row_range: &Range<u64>,
    ) -> VortexResult<Vec<ProjectionFetchHint>> {
        match self {
            Self::Monolithic {
                projected_row_bytes,
                ..
            } => reader.projection_fetch_hints(
                projection_field_mask.to_vec(),
                row_range.clone(),
                *projected_row_bytes,
            ),
            Self::Deferred(plan) => {
                let mut hints = Vec::new();
                for group in &plan.deferred_groups {
                    hints.extend(reader.projection_fetch_hints(
                        group.field_masks.clone(),
                        row_range.clone(),
                        group.fetch_row_bytes,
                    )?);
                }
                Ok(hints)
            }
        }
    }

    pub(crate) fn prefers_projection_aligned_splits(&self) -> bool {
        match self {
            Self::Monolithic {
                projection_aligned_splits,
                ..
            } => *projection_aligned_splits,
            Self::Deferred(_) => false,
        }
    }

    #[cfg(test)]
    pub(crate) fn deferred(&self) -> Option<&DeferredMaterializationPlan> {
        match self {
            Self::Monolithic { .. } => None,
            Self::Deferred(plan) => Some(plan),
        }
    }
}

impl DeferredMaterializationPlan {
    pub(crate) fn final_fields(&self) -> &FieldNames {
        &self.final_fields
    }

    pub(crate) fn immediate_expr(&self) -> Option<Expression> {
        (!self.immediate_fields.is_empty()).then(|| select(self.immediate_fields.clone(), root()))
    }

    pub(crate) fn deferred_groups(&self) -> &[DeferredFieldGroup] {
        &self.deferred_groups
    }
}

impl DeferredFieldGroup {
    pub(crate) fn projection_expr(&self) -> Expression {
        select(self.field_names.clone(), root())
    }
}

fn projection_masks_include_wide_fields(dtype: &DType, field_masks: &[FieldMask]) -> bool {
    field_masks
        .iter()
        .any(|mask| mask_targets_wide_field(dtype, mask))
}

fn mask_targets_wide_field(dtype: &DType, field_mask: &FieldMask) -> bool {
    match field_mask {
        FieldMask::All => true,
        FieldMask::Prefix(path) | FieldMask::Exact(path) => {
            if path.is_root() {
                return true;
            }

            path.resolve(dtype.clone())
                .map(|target| is_wide_projection_dtype(&target))
                .unwrap_or_else(|| is_wide_projection_dtype(dtype))
        }
    }
}

fn is_wide_projection_dtype(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::Utf8(_)
            | DType::Binary(_)
            | DType::List(..)
            | DType::FixedSizeList(..)
            | DType::Struct(..)
    )
}

fn simple_root_projection_fields(projection: &Expression, dtype: &DType) -> Option<FieldNames> {
    let struct_fields = dtype.as_struct_fields_opt()?;
    if projection.is::<Root>() {
        return Some(struct_fields.names().clone());
    }

    projection
        .as_opt::<Select>()
        .filter(|_| projection.child(0).is::<Root>())
        .and_then(|selection| {
            selection
                .normalize_to_included_fields(struct_fields.names())
                .ok()
        })
}

fn should_defer_field(dtype: &DType, row_cost_bytes: usize) -> bool {
    is_wide_projection_dtype(dtype) || row_cost_bytes > IMMEDIATE_FIELD_ROW_BYTES_THRESHOLD
}

pub(crate) fn estimate_field_mask_row_bytes(dtype: &DType, field_masks: &[FieldMask]) -> usize {
    if field_masks.is_empty() {
        return 0;
    }

    if field_masks.iter().any(FieldMask::matches_all) {
        return estimate_dtype_row_bytes(dtype);
    }

    field_masks.iter().fold(0usize, |sum, mask| {
        sum.saturating_add(estimate_single_mask_row_bytes(dtype, mask))
    })
}

fn estimate_single_mask_row_bytes(dtype: &DType, field_mask: &FieldMask) -> usize {
    match field_mask {
        FieldMask::All => estimate_dtype_row_bytes(dtype),
        FieldMask::Prefix(path) | FieldMask::Exact(path) => {
            if path.is_root() {
                return estimate_dtype_row_bytes(dtype);
            }

            path.resolve(dtype.clone())
                .map(|dtype| estimate_dtype_row_bytes(&dtype))
                .unwrap_or_else(|| estimate_dtype_row_bytes(dtype))
        }
    }
}

fn estimate_dtype_row_bytes(dtype: &DType) -> usize {
    dtype.element_size().unwrap_or_else(|| match dtype {
        DType::Struct(fields, _) => fields.fields().fold(0usize, |sum, child| {
            sum.saturating_add(estimate_dtype_row_bytes(&child))
        }),
        DType::List(elem_dtype, _) => {
            DEFAULT_VARIABLE_ROW_BYTES.saturating_add(estimate_dtype_row_bytes(elem_dtype))
        }
        DType::FixedSizeList(elem_dtype, list_size, _) => {
            estimate_dtype_row_bytes(elem_dtype).saturating_mul(*list_size as usize)
        }
        DType::Extension(ext_dtype) => estimate_dtype_row_bytes(ext_dtype.storage_dtype()),
        _ => DEFAULT_VARIABLE_ROW_BYTES,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::Field;
    use vortex_array::dtype::FieldMask;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::FieldPath;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::root;
    use vortex_array::expr::select;

    use super::MaterializationPlan;
    use super::estimate_field_mask_row_bytes;

    fn scan_dtype() -> DType {
        DType::Struct(
            StructFields::new(
                FieldNames::from(["id", "score", "payload", "nested", "tag"]),
                vec![
                    DType::Primitive(PType::I64, Nullability::NonNullable),
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Utf8(Nullability::Nullable),
                    DType::Struct(
                        StructFields::new(
                            FieldNames::from(["child"]),
                            vec![DType::Primitive(PType::I32, Nullability::Nullable)],
                        ),
                        Nullability::Nullable,
                    ),
                    DType::Utf8(Nullability::Nullable),
                ],
            ),
            Nullability::Nullable,
        )
    }

    #[test]
    fn deferred_plan_activates_for_narrow_filtered_projection() {
        let projection = select(["id", "payload"], root());
        let mask = vec![FieldMask::Prefix(FieldPath::from(Field::Name("id".into())))];
        let plan = MaterializationPlan::from_projection(
            &projection,
            &scan_dtype(),
            true,
            &mask,
            &BTreeSet::new(),
        );
        let deferred = plan.deferred().expect("deferred plan");
        assert_eq!(
            deferred.final_fields(),
            &FieldNames::from(["id", "payload"])
        );
        assert!(deferred.immediate_expr().is_some());
        assert_eq!(deferred.deferred_groups().len(), 1);
    }

    #[test]
    fn deferred_plan_stays_off_for_unfiltered_projection() {
        let projection = select(["id", "payload"], root());
        let mask = vec![FieldMask::Prefix(FieldPath::from(Field::Name("id".into())))];
        let plan = MaterializationPlan::from_projection(
            &projection,
            &scan_dtype(),
            false,
            &mask,
            &BTreeSet::new(),
        );
        assert!(plan.deferred().is_none());
    }

    #[test]
    fn deferred_plan_stays_off_for_root_projection() {
        let mask = vec![FieldMask::Prefix(FieldPath::from(Field::Name("id".into())))];
        let plan = MaterializationPlan::from_projection(
            &root(),
            &scan_dtype(),
            true,
            &mask,
            &BTreeSet::new(),
        );
        assert!(plan.deferred().is_none());
    }

    #[test]
    fn deferred_plan_stays_off_for_wide_projection() {
        let projection = select(["id", "score", "payload", "nested"], root());
        let mask = vec![FieldMask::Prefix(FieldPath::from(Field::Name("id".into())))];
        let plan = MaterializationPlan::from_projection(
            &projection,
            &scan_dtype(),
            true,
            &mask,
            &BTreeSet::new(),
        );
        assert!(plan.deferred().is_none());
    }

    #[test]
    fn estimate_field_mask_row_bytes_uses_only_requested_fields() {
        let dtype = scan_dtype();
        let id_mask = [FieldMask::Prefix(FieldPath::from(Field::Name("id".into())))];
        let payload_mask = [FieldMask::Prefix(FieldPath::from(Field::Name(
            "payload".into(),
        )))];
        assert!(
            estimate_field_mask_row_bytes(&dtype, &payload_mask)
                > estimate_field_mask_row_bytes(&dtype, &id_mask)
        );
    }
}
