// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SQL-style CASE WHEN: evaluates `(condition, value)` pairs in order and returns
//! the value from the first matching condition (first-match-wins). NULL conditions
//! are treated as false. If no ELSE clause is provided, unmatched rows produce NULL;
//! otherwise they get the ELSE value.
//!
//! Unlike SQL which coerces all branches to a common supertype, all THEN/ELSE
//! branches must share the same base dtype (ignoring nullability). The result
//! nullability is the union of all branches (forced nullable if no ELSE).
//!
//! The serialized expression format is not part of a Vortex edition and is not
//! guaranteed to remain compatible across Vortex versions.

use std::fmt;
use std::fmt::Formatter;
use std::hash::Hash;
use std::sync::Arc;

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::proto::expr as pb;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ReduceNode;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnRef;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::builtin::fill_null::FillNull;
use vortex_array::scalar_fn::builtin::is_not_null::IsNotNull;
use vortex_array::scalar_fn::display::ExprDisplay;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::is_null::IsNull;
use crate::zip::zip_nullability_union;

/// Options for the n-ary CaseWhen expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CaseWhenOptions {
    /// Number of WHEN/THEN pairs.
    pub num_when_then_pairs: u32,
    /// Whether an ELSE clause is present.
    /// If false, unmatched rows return NULL.
    pub has_else: bool,
}

impl CaseWhenOptions {
    /// Total number of child expressions: 2 per WHEN/THEN pair, plus 1 if ELSE is present.
    pub fn num_children(&self) -> usize {
        self.num_when_then_pairs as usize * 2 + usize::from(self.has_else)
    }
}

impl fmt::Display for CaseWhenOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "case_when(pairs={}, else={})",
            self.num_when_then_pairs, self.has_else
        )
    }
}

/// An n-ary CASE WHEN expression.
///
/// Children are in order: `[when_0, then_0, when_1, then_1, ..., else?]`.
#[derive(Clone)]
pub struct CaseWhen;

impl ScalarFnVTable for CaseWhen {
    type Options = CaseWhenOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.case_when");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        vortex_ensure!(
            options.num_when_then_pairs > 0,
            "CaseWhen must have at least one WHEN/THEN pair"
        );
        let Some(num_children) = options
            .num_when_then_pairs
            .checked_mul(2)
            .and_then(|count| count.checked_add(u32::from(options.has_else)))
        else {
            vortex_bail!("CaseWhen has too many children to serialize");
        };
        Ok(Some(pb::CaseWhenOpts { num_children }.encode_to_vec()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::CaseWhenOpts::decode(metadata)?;
        if opts.num_children < 2 {
            vortex_bail!(
                "CaseWhen expects at least 2 children, got {}",
                opts.num_children
            );
        }
        Ok(CaseWhenOptions {
            num_when_then_pairs: opts.num_children / 2,
            has_else: opts.num_children % 2 == 1,
        })
    }

    fn arity(&self, options: &Self::Options) -> Arity {
        Arity::Exact(options.num_children())
    }

    fn child_name(&self, options: &Self::Options, child_idx: usize) -> ChildName {
        let num_pair_children = options.num_when_then_pairs as usize * 2;
        if child_idx < num_pair_children {
            let pair_idx = child_idx / 2;
            if child_idx.is_multiple_of(2) {
                ChildName::from(Arc::from(format!("when_{pair_idx}")))
            } else {
                ChildName::from(Arc::from(format!("then_{pair_idx}")))
            }
        } else if options.has_else && child_idx == num_pair_children {
            ChildName::from("else")
        } else {
            unreachable!("Invalid child index {} for CaseWhen", child_idx)
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "CASE")?;
        for i in 0..options.num_when_then_pairs as usize {
            write!(
                f,
                " WHEN {} THEN {}",
                expr.display_child(i * 2),
                expr.display_child(i * 2 + 1)
            )?;
        }
        if options.has_else {
            let else_idx = options.num_when_then_pairs as usize * 2;
            write!(f, " ELSE {}", expr.display_child(else_idx))?;
        }
        write!(f, " END")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        if options.num_when_then_pairs == 0 {
            vortex_bail!("CaseWhen must have at least one WHEN/THEN pair");
        }

        let expected_len = options.num_children();
        if arg_dtypes.len() != expected_len {
            vortex_bail!(
                "CaseWhen expects {expected_len} argument dtypes, got {}",
                arg_dtypes.len()
            );
        }

        // Unlike SQL which coerces all branches to a common supertype, we require
        // all THEN/ELSE branches to have the same base dtype (ignoring nullability).
        // The result nullability is the union of all branches, at every nesting level.
        let first_then = &arg_dtypes[1];
        let mut result_dtype = first_then.clone();

        for i in 1..options.num_when_then_pairs as usize {
            let then_i = &arg_dtypes[i * 2 + 1];
            let Some(union) = zip_nullability_union(&result_dtype, then_i) else {
                vortex_bail!(
                    "CaseWhen THEN dtypes must match (ignoring nullability), got {} and {}",
                    first_then,
                    then_i
                );
            };
            result_dtype = union;
        }

        if options.has_else {
            let else_dtype = &arg_dtypes[options.num_when_then_pairs as usize * 2];
            let Some(union) = zip_nullability_union(&result_dtype, else_dtype) else {
                vortex_bail!(
                    "CaseWhen THEN and ELSE dtypes must match (ignoring nullability), got {} and {}",
                    first_then,
                    else_dtype
                );
            };
            result_dtype = union;
        } else {
            // No ELSE means unmatched rows are NULL
            result_dtype = result_dtype.as_nullable();
        }

        Ok(result_dtype)
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let row_count = args.row_count();
        let num_pairs = options.num_when_then_pairs as usize;
        let dtypes = (0..args.num_inputs())
            .map(|index| Ok(args.get(index)?.dtype().clone()))
            .collect::<VortexResult<Vec<_>>>()?;
        let output_dtype = self.return_dtype(options, &dtypes)?;

        let mut remaining = Mask::new_true(row_count);
        let mut branches: Vec<(Mask, ArrayRef)> = Vec::with_capacity(num_pairs);

        for i in 0..num_pairs {
            if remaining.all_false() {
                break;
            }

            let condition = args.get(i * 2)?.filter(remaining.clone())?;
            let cond_bool = condition.execute::<BoolArray>(ctx)?;
            let cond_mask = cond_bool.to_mask_fill_null_false(ctx);
            let effective_mask = remaining.intersect_by_rank(&cond_mask);

            if effective_mask.all_false() {
                continue;
            }

            let then_value = args.get(i * 2 + 1)?;
            remaining = remaining.bitand_not(&effective_mask);
            branches.push((effective_mask, then_value));
        }

        let else_value: ArrayRef = if options.has_else {
            args.get(num_pairs * 2)?
        } else {
            let then_dtype = args.get(1)?.dtype().as_nullable();
            ConstantArray::new(Scalar::null(then_dtype), row_count).into_array()
        };

        if branches.is_empty() {
            return else_value.cast(output_dtype);
        }

        // `remaining` now selects the rows that no branch matched.
        merge_case_branches(&branches, &remaining, &else_value, &output_dtype, ctx)
    }

    fn simplify<T: ReduceNode>(
        &self,
        options: &Self::Options,
        node: &T,
    ) -> VortexResult<Option<T>> {
        // Rewrite the COALESCE-shaped CASE WHEN into `fill_null`, which references `x`
        // once and lowers to a single fill kernel instead of a `zip`/merge that resolves
        // `x` twice (once for the `is_null` predicate, once for the value branch).
        //
        //   CASE WHEN is_null(x)     THEN c ELSE x END  ==>  fill_null(x, c)
        //   CASE WHEN is_not_null(x) THEN x ELSE c END  ==>  fill_null(x, c)
        //
        // The fill `c` must be a constant: `fill_null`'s kernel reads the fill value via
        // `as_constant()`, so a non-constant fill would produce an unexecutable node.
        if options.num_when_then_pairs != 1 || !options.has_else {
            return Ok(None);
        }

        let when = node.child(0);
        let then = node.child(1);
        let els = node.child(2);

        let when_is = |f: &dyn Fn(&ScalarFnRef) -> bool| when.scalar_fn().is_some_and(f);

        // `is_null(x) ? c : x` — predicate operand and ELSE are the same `x`, fill is THEN.
        let (x, fill) = if when_is(&|f| f.is::<IsNull>()) && when.child(0).is_same(&els) {
            (els, then)
        // `is_not_null(x) ? x : c` — predicate operand and THEN are the same `x`, fill is ELSE.
        } else if when_is(&|f| f.is::<IsNotNull>()) && when.child(0).is_same(&then) {
            (then, els)
        } else {
            return Ok(None);
        };

        let Some(scalar) = fill.as_constant() else {
            return Ok(None);
        };

        if scalar.is_null() {
            // Filling the nulls of `x` with NULL is a no-op
            return Ok(Some(x));
        }

        let x_dtype = x.node_dtype()?;
        if !FillNull::supports_dtype(&x_dtype)
            || x_dtype.as_nullable() != node.node_dtype()?.as_nullable()
        {
            return Ok(None);
        }

        // `FillNull` preserves `x`'s nested nullability (e.g. `list(i32?)` or `struct{a: i32?}`),
        // while `short_circuit` and `fill_null_constant` adopt `fill`'s dtype directly. Widen
        // `fill`'s nested nullability to match `x` while keeping `fill`'s own outer nullability.
        let fill_dtype = x_dtype.with_nullability(scalar.dtype().nullability());
        let fill = if scalar.dtype() != &fill_dtype {
            node.new_constant(scalar.cast(&fill_dtype)?)
        } else {
            fill
        };

        Ok(Some(
            node.new_node(FillNull.bind(EmptyOptions), &[x, fill])?,
        ))
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        // A null in an unselected branch does not force a null output.
        false
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

/// Assemble coarse runs directly; compact fragmented branches before restoring row positions.
///
/// `else_mask` selects the rows that no branch matched. Every branch and the ELSE value
/// are cast to `output_dtype`, the dtype that `return_dtype` reports.
fn merge_case_branches(
    branches: &[(Mask, ArrayRef)],
    else_mask: &Mask,
    else_value: &ArrayRef,
    output_dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let Some((_, value)) = branches.iter().find(|(mask, _)| mask.all_true()) {
        return value.cast(output_dtype.clone());
    }

    if prefer_runs(branches, else_value.len()) {
        return merge_runs(branches, else_value, output_dtype, ctx);
    }

    merge_compact(branches, else_mask, else_value, output_dtype, ctx)
}

fn prefer_runs(branches: &[(Mask, ArrayRef)], len: usize) -> bool {
    // A span may add an ELSE run. Coarse runs amortize slicing without the
    // extra compact-output buffer and gather; isolated branch selections do too.
    const MIN_ROWS_PER_RUN: usize = 128;

    let mut remaining_spans = branches.len().max(len / (2 * MIN_ROWS_PER_RUN));
    for (mask, _) in branches {
        let count = match mask {
            Mask::AllTrue(_) => 1,
            Mask::AllFalse(_) => 0,
            Mask::Values(values) => values.cached_slices().map_or_else(
                || {
                    values
                        .bit_buffer()
                        .set_slices()
                        .take(remaining_spans.saturating_add(1))
                        .count()
                },
                |slices| slices.len(),
            ),
        };

        if count > remaining_spans {
            return false;
        }
        remaining_spans -= count;
    }

    true
}

fn merge_compact(
    branches: &[(Mask, ArrayRef)],
    else_mask: &Mask,
    else_value: &ArrayRef,
    output_dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = else_value.len();
    let mut builder = builder_with_capacity_in(output_dtype, len, ctx.allocator());
    let mut indices = BufferMut::<u64>::zeroed_in(len, ctx.allocator().clone());

    let else_branch = (!else_mask.all_false()).then_some((else_mask, else_value));
    let selections = branches
        .iter()
        .map(|(mask, value)| (mask, value))
        .chain(else_branch);

    for (mask, value) in selections {
        let offset = builder.len();
        value
            .filter(mask.clone())?
            .cast(output_dtype.clone())?
            .append_to_builder(builder.as_mut(), ctx)?;
        assign_positions(&mut indices, mask, offset);
    }

    builder.finish().take(indices.freeze().into_array())
}

fn merge_runs(
    branches: &[(Mask, ArrayRef)],
    else_value: &ArrayRef,
    output_dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut spans = Vec::new();
    for (index, (mask, _)) in branches.iter().enumerate() {
        if let AllOr::Some(slices) = mask.slices() {
            spans.extend(slices.iter().map(|&(start, end)| (start, end, index)));
        }
    }
    spans.sort_unstable_by_key(|&(start, ..)| start);

    // Cast once up front: every span below is a slice of these arrays.
    let values = branches
        .iter()
        .map(|(_, value)| value.cast(output_dtype.clone())?.optimize_ctx(ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    let else_value = else_value.cast(output_dtype.clone())?.optimize_ctx(ctx)?;
    let len = else_value.len();

    let mut builder = builder_with_capacity_in(output_dtype, len, ctx.allocator());
    for (start, end, branch) in spans {
        if builder.len() < start {
            else_value
                .slice(builder.len()..start)?
                .append_to_builder(builder.as_mut(), ctx)?;
        }

        values[branch]
            .slice(start..end)?
            .append_to_builder(builder.as_mut(), ctx)?;
    }

    if builder.len() < len {
        else_value
            .slice(builder.len()..len)?
            .append_to_builder(builder.as_mut(), ctx)?;
    }

    Ok(builder.finish())
}

fn assign_positions(indices: &mut [u64], mask: &Mask, offset: usize) {
    match mask.indices() {
        AllOr::All => {
            for (position, index) in indices.iter_mut().enumerate() {
                *index = (offset + position) as u64;
            }
        }
        AllOr::None => {}
        AllOr::Some(selected) => {
            for (position, &row) in selected.iter().enumerate() {
                indices[row] = (offset + position) as u64;
            }
        }
    }
}

#[cfg(test)]
mod tests;
