// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::List;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::dict::take_referenced_canonical;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArrayExt;
use crate::OnPairArraySlotsExt;

/// A take with fewer indices than `1 / SPARSE_TAKE_DENOMINATOR` of the rows gathers
/// token runs instead of decoding the referenced rows. Uniformly random indices that sparse
/// repeat too rarely for decoding each referenced row once to pay for finding them; see
/// `benches/`.
const SPARSE_TAKE_DENOMINATOR: usize = 2;

impl TakeExecute for OnPair {
    /// Gathers token runs for a sparse take, and otherwise decodes each referenced row once and
    /// gathers the decoded strings.
    ///
    /// Gathering token runs decodes a row again for every index that repeats it, which is far
    /// slower when the indices are dense, as in a dictionary over OnPair values. A sparse take has
    /// few repeats to save, so there the cost of finding the referenced rows dominates.
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if indices.len().saturating_mul(SPARSE_TAKE_DENOMINATOR) < array.len() {
            take_token_runs(array, indices, ctx).map(Some)
        } else {
            take_referenced_canonical(array.array(), indices, ctx).map(Some)
        }
    }
}

/// Takes the token runs, sharing the pair dictionary, without decoding.
fn take_token_runs(
    array: ArrayView<'_, OnPair>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // SAFETY: `codes_offsets` delimit the token runs in `codes`, as for the filter kernel.
    let codes = unsafe {
        ListArray::new_unchecked(
            array.codes().clone(),
            array.codes_offsets().clone(),
            Validity::NonNullable,
        )
    };
    let taken_codes = <List as TakeExecute>::take(codes.as_view(), indices, ctx)?
        .vortex_expect("List take kernel always returns Some")
        .try_downcast::<List>()
        .ok()
        .vortex_expect("List take returns a List");

    // Null indices take a zero length, matching their empty token runs.
    let uncompressed_lengths = array.uncompressed_lengths().clone();
    let zero = Scalar::zero_value(uncompressed_lengths.dtype());
    let uncompressed_lengths = uncompressed_lengths
        .take(indices.clone())?
        .fill_null(zero)?;
    let validity = array.array_validity().take(indices)?;

    // SAFETY: the dictionary is unchanged and the codes, lengths and validity were all taken
    // with the same indices, so every row still decodes to its original string.
    Ok(unsafe {
        OnPair::new_unchecked(
            array
                .dtype()
                .clone()
                .union_nullability(indices.dtype().nullability()),
            array.data().clone(),
            array.dict_offsets().clone(),
            taken_codes.elements().clone(),
            taken_codes.offsets().clone(),
            uncompressed_lengths,
            validity,
        )
    }
    .into_array())
}
