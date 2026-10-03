// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;

use goldenfile::Mint;
use goldenfile::differs::binary_diff;
use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::assert_arrays_eq;
use crate::chunk_iter::ValueType;
use crate::chunk_iter::execute_via_chunks;
use crate::chunk_iter::without_chunked_execute;

#[cfg(not(codspeed))]
pub mod trace;

/// Check that a named metadata matches its previous versioning.
///
/// Goldenfile takes care of checking for equality against a checked-in file.
#[expect(clippy::unwrap_used)]
pub fn check_metadata(name: &str, metadata: &[u8]) {
    let mut mint = Mint::new("goldenfiles/");
    let mut f = mint
        .new_goldenfile_with_differ(name, Box::new(binary_diff))
        .unwrap();
    f.write_all(metadata).unwrap();
}

/// Outputs the indices of the true values in a BoolArray
pub fn to_int_indices(indices_bits: BoolArray, ctx: &mut ExecutionCtx) -> VortexResult<Vec<u64>> {
    let buffer = indices_bits.to_bit_buffer();
    let mask = indices_bits
        .as_ref()
        .validity()?
        .execute_mask(indices_bits.as_ref().len(), ctx)?;
    Ok(buffer
        .iter()
        .enumerate()
        .filter_map(|(idx, v)| (v && mask.value(idx)).then_some(idx as u64))
        .collect_vec())
}

/// Assert that `array` streams through
/// [`ArrayRef::decompress_chunks`](crate::ArrayRef::decompress_chunks) to the same values and
/// validity as executing it level-wise, and that a decimal array streams the integer type
/// executing it stores its values in.
pub fn assert_streams_like_execute(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    let value_type = array
        .decompress_chunks_type()
        .ok_or_else(|| vortex_err!("{} does not support decompress_chunks", array.encoding_id()))?;
    let streamed = execute_via_chunks(array, ctx)?;
    // The executor would stream a deep enough tree too, which would make no independent reference.
    let executed = without_chunked_execute(|| array.clone().execute::<Canonical>(ctx))?;
    if let Canonical::Decimal(decimal) = &executed {
        assert_eq!(
            ValueType::from(decimal.values_type()),
            value_type,
            "{} streams another type than executing it stores",
            array.encoding_id()
        );
    }
    assert_arrays_eq!(streamed, executed.into_array(), ctx);
    Ok(())
}
