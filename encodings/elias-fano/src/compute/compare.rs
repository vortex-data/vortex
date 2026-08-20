// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Comparing an Elias-Fano array against a constant, without decoding it.
//!
//! The sequence is sorted, so the matching rows form a contiguous run — or, for `NotEq`, the
//! complement of one. Two sampled searches find its bounds and the answer is a bit-buffer fill.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;

use crate::EliasFano;
use crate::EliasFanoCursor;

impl CompareKernel for EliasFano {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only a constant right-hand side reduces to a range. The adaptor has already normalised
        // operand order, flipping the operator if the encoded array arrived on the right.
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        // A null operand makes every row null rather than a range. The adaptor answers that before
        // a kernel is reached, but this is a public trait impl and the cursor takes no nulls.
        if constant.is_null() {
            return Ok(None);
        }
        // The cursor probes in the array's own dtype. A literal differing only in nullability casts
        // cleanly; one outside the column's range is left to the generic path, which promotes both.
        let Ok(constant) = constant.cast(lhs.dtype()) else {
            return Ok(None);
        };

        let len = lhs.len();
        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();

        // `rank` and `rank_inclusive` bracket the run equal to the constant, and every comparison
        // is one side of that pair, so no operator costs more than two searches and most cost one.
        let mut cursor = EliasFanoCursor::try_new(lhs, ctx)?;
        let result = match operator {
            // These need only the lower bound.
            CompareOperator::Lt => run(0..cursor.rank(&constant)?, len, nullability),
            CompareOperator::Gte => run(cursor.rank(&constant)?..len, len, nullability),
            // These need only the upper bound.
            CompareOperator::Lte => run(0..cursor.rank_inclusive(&constant)?, len, nullability),
            CompareOperator::Gt => run(cursor.rank_inclusive(&constant)?..len, len, nullability),
            CompareOperator::Eq => {
                let lo = cursor.rank(&constant)?;
                run(lo..cursor.rank_inclusive(&constant)?, len, nullability)
            }
            // The one answer that is not a single run.
            CompareOperator::NotEq => {
                let lo = cursor.rank(&constant)?;
                complement(lo..cursor.rank_inclusive(&constant)?, len, nullability)
            }
        };
        Ok(Some(result))
    }
}

fn validity(nullability: Nullability) -> Validity {
    match nullability {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => Validity::AllValid,
    }
}

/// A boolean array true exactly on `range`. An all-true or all-false answer becomes a constant, so
/// whatever consumes it can skip the array entirely.
fn run(range: Range<usize>, len: usize, nullability: Nullability) -> ArrayRef {
    if range.start >= range.end {
        return ConstantArray::new(Scalar::bool(false, nullability), len).into_array();
    }
    if range.start == 0 && range.end == len {
        return ConstantArray::new(Scalar::bool(true, nullability), len).into_array();
    }
    let mut buffer = BitBufferMut::new_unset(len);
    buffer.fill_range(range.start, range.end, true);
    BoolArray::new(buffer.freeze(), validity(nullability)).into_array()
}

fn complement(range: Range<usize>, len: usize, nullability: Nullability) -> ArrayRef {
    if range.start >= range.end {
        return ConstantArray::new(Scalar::bool(true, nullability), len).into_array();
    }
    if range.start == 0 && range.end == len {
        return ConstantArray::new(Scalar::bool(false, nullability), len).into_array();
    }
    let mut buffer = BitBufferMut::new_set(len);
    buffer.fill_range(range.start, range.end, false);
    BoolArray::new(buffer.freeze(), validity(nullability)).into_array()
}
