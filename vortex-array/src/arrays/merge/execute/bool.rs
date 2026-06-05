// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Boolean-selector execution: the optimized two-branch [`Merge`](super::super::Merge) path.

use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use super::super::Merge;
use super::super::MergeArrayExt;
use crate::array::Array;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::require_child;
use crate::validity::Validity;

/// Merges two compact boolean branches under a non-nullable boolean selector, scattering each
/// branch's bits (and validity) into the positions the selector routes to it.
pub(super) fn execute(
    array: Array<Merge>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    debug_assert_eq!(
        array.num_branches(),
        2,
        "a boolean selector implies exactly two branches"
    );

    // Drive the branches and selector to canonical `Bool` so we can operate on raw bits.
    let array = require_child!(array, array.branch(0), 0 => Bool);
    let array = require_child!(array, array.branch(1), 1 => Bool);
    let array = require_child!(array, array.selector(), 2 => Bool);

    let dtype = array.as_ref().dtype().clone();
    let len = array.as_ref().len();
    let nullable = dtype.is_nullable();

    let b0 = array.branch(0).as_::<Bool>();
    let b1 = array.branch(1).as_::<Bool>();
    // The selector is non-nullable (enforced at construction), so its bits are the routing mask
    // directly: `false` selects branch 0, `true` selects branch 1.
    let selector = Mask::from_buffer(array.selector().as_::<Bool>().to_bit_buffer());

    let b0_bits = b0.to_bit_buffer();
    let b1_bits = b1.to_bit_buffer();
    vortex_ensure!(
        selector.true_count() == b1_bits.len() && len - selector.true_count() == b0_bits.len(),
        "merge selector does not partition into the branch lengths"
    );

    // Branch validity is only materialized when the output can be null.
    let (v0, v1) = if nullable {
        (
            Some(b0.validity()?.execute_mask(b0_bits.len(), ctx)?),
            Some(b1.validity()?.execute_mask(b1_bits.len(), ctx)?),
        )
    } else {
        (None, None)
    };

    let mut values = BitBufferMut::new_unset(len);
    let mut validity = nullable.then(|| BitBufferMut::new_set(len));

    let (mut c0, mut c1) = (0usize, 0usize);
    for i in 0..len {
        // Scatter one bit (and its validity) from the selected branch's current cursor.
        let (bit, valid) = if selector.value(i) {
            let out = (b1_bits.value(c1), v1.as_ref().is_none_or(|m| m.value(c1)));
            c1 += 1;
            out
        } else {
            let out = (b0_bits.value(c0), v0.as_ref().is_none_or(|m| m.value(c0)));
            c0 += 1;
            out
        };
        if bit {
            values.set(i);
        }
        // `validity` is `Some` exactly when `nullable`, and `valid` is always true otherwise.
        if !valid {
            validity
                .as_mut()
                .vortex_expect("validity buffer present when nullable")
                .unset(i);
        }
    }

    let validity = match validity {
        Some(bits) => Validity::from(bits.freeze()),
        None => Validity::NonNullable,
    };
    Ok(ExecutionResult::done(BoolArray::try_new(
        values.freeze(),
        validity,
    )?))
}
