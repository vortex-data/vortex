// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared execution for the binary geo scalar functions.
//!
//! [`execute_null_propagating`] runs a binary geo kernel (`ST_Distance`, `ST_Intersects`,
//! `ST_Contains`) over its two operands, decoding to `geo_types` and computing per row. Nulls
//! propagate as in SQL — the result is null wherever either operand is null — which the kernels
//! also expose via `vortex_array::expr::union_child_validities` as their `validity()`, so the
//! planner can derive the output null mask without executing them.

use geo_types::Geometry;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::extension::geometries;
use crate::extension::single_geometry;

/// The result type a binary geo kernel produces. Today that is `f64` (for `ST_Distance`) and
/// `bool` (for the `ST_Intersects` / `ST_Contains` predicates), and the trait is implemented for
/// both. A kernel that returns some other type just adds its own `impl GeoOutput`.
pub(crate) trait GeoOutput: Copy {
    /// Convert this computed value into a Vortex [`Scalar`] (one typed, nullable value). Used
    /// only when both operands are constant: the kernel computes a single result, and this wraps
    /// it so a constant array can repeat that one value across every row.
    fn into_scalar(self, nullability: Nullability) -> Scalar;

    /// Assemble the `len`-row output: `values` (one per valid row, in row order) land at the set
    /// positions of `valid`, and every other row is null. With an empty `valid` this is the
    /// all-null output.
    fn build_array(
        len: usize,
        valid: &Mask,
        values: Vec<Self>,
        nullability: Nullability,
    ) -> ArrayRef;
}

impl GeoOutput for f64 {
    fn into_scalar(self, nullability: Nullability) -> Scalar {
        Scalar::primitive(self, nullability)
    }

    fn build_array(
        len: usize,
        valid: &Mask,
        values: Vec<Self>,
        nullability: Nullability,
    ) -> ArrayRef {
        let validity = Validity::from_mask(valid.clone(), nullability);
        match valid.indices() {
            // No nulls: `values` already lines up one-to-one with the rows.
            AllOr::All => PrimitiveArray::new(values, validity).into_array(),
            // No valid rows: the whole output is null.
            AllOr::None => PrimitiveArray::new(vec![0.0f64; len], validity).into_array(),
            // Some nulls: scatter each computed value back to the row it came from.
            AllOr::Some(rows) => {
                let mut data = vec![0.0f64; len];
                for (&row, value) in rows.iter().zip(values) {
                    data[row] = value;
                }
                PrimitiveArray::new(data, validity).into_array()
            }
        }
    }
}

impl GeoOutput for bool {
    fn into_scalar(self, nullability: Nullability) -> Scalar {
        Scalar::bool(self, nullability)
    }

    fn build_array(
        len: usize,
        valid: &Mask,
        values: Vec<Self>,
        nullability: Nullability,
    ) -> ArrayRef {
        let validity = Validity::from_mask(valid.clone(), nullability);
        match valid.indices() {
            // No nulls: `values` already lines up one-to-one with the rows.
            AllOr::All => BoolArray::new(BitBuffer::from_iter(values), validity).into_array(),
            // No valid rows: the whole output is null.
            AllOr::None => BoolArray::new(BitBuffer::new_unset(len), validity).into_array(),
            // Some nulls: scatter each computed value back to the row it came from.
            AllOr::Some(rows) => {
                let mut data = vec![false; len];
                for (&row, value) in rows.iter().zip(values) {
                    data[row] = value;
                }
                BoolArray::new(BitBuffer::from_iter(data), validity).into_array()
            }
        }
    }
}

/// Run a binary geo kernel over operands `a` and `b`, each a column or a constant literal.
///
/// The output is null wherever either operand is null, and its type is nullable if either operand
/// is: equivalently, the output validity is the intersection of the operands' validities.
///
/// The core idea: a geo kernel decodes each operand into a `geo_types` geometry, and a null row
/// has no geometry to decode, so it can't compute over every row and mask the nulls afterwards
/// (the way numeric kernels do). Instead it skips the nulls up front: keep the rows valid in both
/// operands, decode and compute only those, then scatter the results back to their rows and leave
/// every other row null.
pub(crate) fn execute_null_propagating<T, F>(
    a: &ArrayRef,
    b: &ArrayRef,
    compute: F,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: GeoOutput,
    F: Fn(&Geometry<f64>, &Geometry<f64>) -> T + Copy,
{
    let len = a.len();
    let nullability = Nullability::from(a.dtype().is_nullable() || b.dtype().is_nullable());

    // A null constant operand makes every row null (an empty mask builds the all-null output).
    for operand in [a, b] {
        if operand
            .as_opt::<Constant>()
            .is_some_and(|c| c.scalar().is_null())
        {
            return Ok(T::build_array(
                len,
                &Mask::new_false(len),
                vec![],
                nullability,
            ));
        }
    }

    match (a.as_opt::<Constant>(), b.as_opt::<Constant>()) {
        // Both constant: compute once and broadcast across every row.
        (Some(qa), Some(qb)) => {
            let ga = single_geometry(qa.scalar(), ctx)?;
            let gb = single_geometry(qb.scalar(), ctx)?;
            Ok(ConstantArray::new(compute(&ga, &gb).into_scalar(nullability), len).into_array())
        }
        // One constant, one column: fix the constant geometry and evaluate down the column.
        (Some(qa), None) => {
            let ga = single_geometry(qa.scalar(), ctx)?;
            eval_column(b, |g| compute(&ga, g), nullability, ctx)
        }
        (None, Some(qb)) => {
            let gb = single_geometry(qb.scalar(), ctx)?;
            eval_column(a, |g| compute(g, &gb), nullability, ctx)
        }
        // Two columns: evaluate row by row.
        (None, None) => {
            vortex_ensure_eq!(
                a.len(),
                b.len(),
                "geo binary: operand length mismatch {} vs {}",
                a.len(),
                b.len()
            );
            eval_column_pair(a, b, compute, nullability, ctx)
        }
    }
}

/// Evaluate `f` over each valid row of one geometry `column`, propagating the column's nulls.
fn eval_column<T, F>(
    column: &ArrayRef,
    f: F,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: GeoOutput,
    F: Fn(&Geometry<f64>) -> T,
{
    let len = column.len();
    let valid = column.validity()?.execute_mask(len, ctx)?;
    // Decode only the non-null rows, since a null row has no geometry to decode. The common
    // all-valid case decodes the column directly and skips the filter.
    let decoded = if valid.all_true() {
        geometries(column, ctx)?
    } else {
        geometries(&column.filter(valid.clone())?, ctx)?
    };
    let values = decoded.iter().map(f).collect();
    Ok(T::build_array(len, &valid, values, nullability))
}

/// Evaluate `compute` over each row where both geometry columns are valid, propagating the nulls
/// of either column.
fn eval_column_pair<T, F>(
    a: &ArrayRef,
    b: &ArrayRef,
    compute: F,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: GeoOutput,
    F: Fn(&Geometry<f64>, &Geometry<f64>) -> T,
{
    let len = a.len();
    let a_present = a.validity()?.execute_mask(len, ctx)?;
    let b_present = b.validity()?.execute_mask(len, ctx)?;
    // A row survives only where both columns are present.
    let valid = &a_present & &b_present;
    // Keep only the rows valid in both columns, so decoding never sees a null geometry. The
    // common all-valid case decodes the columns directly and skips the filter.
    let (a, b) = if valid.all_true() {
        (a.clone(), b.clone())
    } else {
        (a.filter(valid.clone())?, b.filter(valid.clone())?)
    };
    let ag = geometries(&a, ctx)?;
    let bg = geometries(&b, ctx)?;
    let values = ag.iter().zip(&bg).map(|(x, y)| compute(x, y)).collect();
    Ok(T::build_array(len, &valid, values, nullability))
}
