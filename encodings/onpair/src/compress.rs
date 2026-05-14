// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Train + compress entry points for the OnPair encoding.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_onpair_sys::Column;
use vortex_onpair_sys::OnPairTrainingConfig;

use crate::OnPair;
use crate::OnPairArray;

/// Default OnPair training configuration: 12-bit codes ("dict-12").
pub const DEFAULT_DICT12_CONFIG: OnPairTrainingConfig = vortex_onpair_sys::DEFAULT_DICT12_CONFIG;

/// Build a training config with a custom bit width.
pub fn config_with_bits(bits: u32) -> OnPairTrainingConfig {
    OnPairTrainingConfig {
        bits,
        threshold: 0.5,
        seed: 0,
    }
}

/// Compress an iterable of optional byte strings via the OnPair C++ library.
///
/// Null entries are still indexed by the column (they map to empty payloads);
/// their nullness is preserved on the outer Vortex array's validity slot.
pub fn onpair_compress_iter<'a, I>(
    iter: I,
    len: usize,
    dtype: DType,
    config: OnPairTrainingConfig,
) -> VortexResult<OnPairArray>
where
    I: Iterator<Item = Option<&'a [u8]>>,
{
    let mut flat: Vec<u8> = Vec::with_capacity(len * 16);
    let mut offsets: Vec<u64> = Vec::with_capacity(len + 1);
    let mut uncompressed_lengths: BufferMut<i32> = BufferMut::with_capacity(len);
    let mut validity: Vec<bool> = Vec::with_capacity(len);
    offsets.push(0);

    for item in iter {
        match item {
            Some(bytes) => {
                flat.extend_from_slice(bytes);
                offsets.push(flat.len() as u64);
                uncompressed_lengths.push(
                    i32::try_from(bytes.len()).vortex_expect("string length must fit in i32"),
                );
                validity.push(true);
            }
            None => {
                offsets.push(flat.len() as u64);
                uncompressed_lengths.push(0);
                validity.push(false);
            }
        }
    }

    let column = Column::compress(&flat, &offsets, config)
        .map_err(|e| vortex_err!("OnPair compress failed: {e}"))?;

    let serialised = column
        .to_bytes()
        .map_err(|e| vortex_err!("OnPair serialise failed: {e}"))?;
    let column_bytes = BufferHandle::new_host(ByteBuffer::from(serialised));

    let uncompressed_lengths = uncompressed_lengths.into_array();
    let validity = match dtype.nullability() {
        vortex_array::dtype::Nullability::NonNullable => Validity::NonNullable,
        vortex_array::dtype::Nullability::Nullable => Validity::from_iter(validity),
    };

    OnPair::try_new(dtype, column, column_bytes, uncompressed_lengths, validity)
}

/// Compress a byte-string accessor (typically a `VarBinArray` or
/// `VarBinViewArray`).
pub fn onpair_compress<A: ArrayAccessor<[u8]>>(
    array: A,
    len: usize,
    dtype: &DType,
    config: OnPairTrainingConfig,
) -> VortexResult<OnPairArray> {
    array.with_iterator(|iter| onpair_compress_iter(iter, len, dtype.clone(), config))
}

/// Compress any [`ArrayRef`] whose canonical form is a string array, by first
/// canonicalising to `VarBinViewArray`.
pub fn onpair_compress_array(
    array: &ArrayRef,
    config: OnPairTrainingConfig,
    ctx: &mut ExecutionCtx,
) -> VortexResult<OnPairArray> {
    let view = array.clone().execute::<VarBinViewArray>(ctx)?;
    let len = view.len();
    let dtype = view.dtype().clone();
    onpair_compress(&view, len, &dtype, config)
}

/// Convenience: build a default `ExecutionCtx` from `LEGACY_SESSION`.
pub fn onpair_compress_array_default(
    array: &ArrayRef,
    config: OnPairTrainingConfig,
) -> VortexResult<OnPairArray> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    onpair_compress_array(array, config, &mut ctx)
}
