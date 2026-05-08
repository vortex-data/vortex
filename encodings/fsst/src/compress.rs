// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Compress a set of values into an Array.

use fsst::Compressor;
use fsst::Symbol;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::varbin::builder::VarBinBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;

/// Compress a string array using FSST.
use crate::FSST;
use crate::FSSTArray;
pub fn fsst_compress<A: ArrayAccessor<[u8]>>(
    strings: A,
    len: usize,
    total_uncompressed: usize,
    dtype: &DType,
    compressor: &Compressor,
    ctx: &mut ExecutionCtx,
) -> FSSTArray {
    // Pick the narrowest sufficient codes-offsets type. The FSST contract
    // bounds the compressed size at `2 * uncompressed + 7` per string, so
    // if the upper bound fits in `i32::MAX` the actual offsets are
    // guaranteed to fit; otherwise we widen to `i64` to avoid the overflow
    // tracked in #7833.
    if upper_bound_fits_i32(total_uncompressed, len) {
        strings.with_iterator(|iter| {
            fsst_compress_iter_with::<i32, _>(iter, len, dtype.clone(), compressor, ctx)
        })
    } else {
        strings.with_iterator(|iter| {
            fsst_compress_iter_with::<i64, _>(iter, len, dtype.clone(), compressor, ctx)
        })
    }
}

/// Train a compressor from an array.
///
/// # Panics
///
/// If the provided array is not FSST compressible.
pub fn fsst_train_compressor<A: ArrayAccessor<[u8]>>(array: &A) -> Compressor {
    array.with_iterator(|iter| fsst_train_compressor_iter(iter))
}

/// Train a [compressor][Compressor] from an iterator of bytestrings.
fn fsst_train_compressor_iter<'a, I>(iter: I) -> Compressor
where
    I: Iterator<Item = Option<&'a [u8]>>,
{
    let mut lines = Vec::with_capacity(8_192);

    for string in iter {
        match string {
            None => {}
            Some(b) => lines.push(b),
        }
    }

    Compressor::train(&lines)
}

/// Most strings are small in practice. If we encounter a larger string, we reallocate
/// the buffer to hold enough capacity for the worst-case compressed value.
const DEFAULT_BUFFER_LEN: usize = 1024 * 1024;

/// Whether the FSST worst-case compressed size for `len` strings totalling
/// `total_uncompressed` bytes fits in an `i32` offset.
fn upper_bound_fits_i32(total_uncompressed: usize, len: usize) -> bool {
    // 2 * total + 7 * n — computed in u64 so the arithmetic itself can't overflow.
    let max_compressed = 2_u64
        .saturating_mul(total_uncompressed as u64)
        .saturating_add(7_u64.saturating_mul(len as u64));
    max_compressed <= i32::MAX as u64
}

/// Compress from an iterator of bytestrings using FSST.
///
/// `total_uncompressed` is the total byte length of all strings in the input;
/// callers typically have it cheaply available (e.g. `VarBinArray::bytes().len()`).
/// It selects the narrowest codes-offsets type that the FSST upper bound
/// (`2 * total_uncompressed + 7 * len`) is guaranteed to fit into.
pub fn fsst_compress_iter<'a, I>(
    iter: I,
    len: usize,
    total_uncompressed: usize,
    dtype: DType,
    compressor: &Compressor,
    ctx: &mut ExecutionCtx,
) -> FSSTArray
where
    I: Iterator<Item = Option<&'a [u8]>>,
{
    if upper_bound_fits_i32(total_uncompressed, len) {
        fsst_compress_iter_with::<i32, _>(iter, len, dtype, compressor, ctx)
    } else {
        fsst_compress_iter_with::<i64, _>(iter, len, dtype, compressor, ctx)
    }
}

fn fsst_compress_iter_with<'a, O, I>(
    iter: I,
    len: usize,
    dtype: DType,
    compressor: &Compressor,
    ctx: &mut ExecutionCtx,
) -> FSSTArray
where
    O: IntegerPType,
    I: Iterator<Item = Option<&'a [u8]>>,
{
    let mut buffer = Vec::with_capacity(DEFAULT_BUFFER_LEN);
    let mut builder = VarBinBuilder::<O>::with_capacity(len);
    let mut uncompressed_lengths: BufferMut<i32> = BufferMut::with_capacity(len);
    for string in iter {
        match string {
            None => {
                builder.append_null();
                uncompressed_lengths.push(0);
            }
            Some(s) => {
                uncompressed_lengths.push(
                    s.len()
                        .try_into()
                        .vortex_expect("string length must fit in i32"),
                );

                // make sure the buffer is 2x+7 larger than the input
                let target_size = 2 * s.len() + 7;
                if target_size > buffer.len() {
                    let additional_capacity = target_size - buffer.len();
                    buffer.reserve(additional_capacity);
                }

                // SAFETY: buffer is always sized to be large enough
                unsafe { compressor.compress_into(s, &mut buffer) };

                builder.append_value(&buffer);
            }
        }
    }

    let codes = builder.finish(DType::Binary(dtype.nullability()));
    let symbols: Buffer<Symbol> = Buffer::copy_from(compressor.symbol_table());
    let symbol_lengths: Buffer<u8> = Buffer::<u8>::copy_from(compressor.symbol_lengths());

    let uncompressed_lengths = uncompressed_lengths.into_array();

    FSST::try_new(
        dtype,
        symbols,
        symbol_lengths,
        codes,
        uncompressed_lengths,
        ctx,
    )
    .vortex_expect("FSST parts must be valid")
}

#[cfg(test)]
mod tests {
    use fsst::CompressorBuilder;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::varbin::VarBinArrayExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;

    use crate::FSSTArrayExt;
    use crate::compress::DEFAULT_BUFFER_LEN;
    use crate::compress::upper_bound_fits_i32;
    use crate::fsst_compress;
    use crate::fsst_compress_iter;
    use crate::fsst_train_compressor;

    #[test]
    fn test_large_string() {
        let big_string: String = "abc"
            .chars()
            .cycle()
            .take(10 * DEFAULT_BUFFER_LEN)
            .collect();

        let compressor = CompressorBuilder::default().build();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let compressed = fsst_compress_iter(
            [Some(big_string.as_bytes())].into_iter(),
            1,
            big_string.len(),
            DType::Utf8(Nullability::NonNullable),
            &compressor,
            &mut ctx,
        );

        let decoded = compressed.execute_scalar(0, &mut ctx).unwrap();

        let expected = Scalar::utf8(big_string, Nullability::NonNullable);

        assert_eq!(decoded, expected);
    }

    #[test]
    fn upper_bound_fits_i32_handles_zero() {
        assert!(upper_bound_fits_i32(0, 0));
    }

    #[test]
    fn upper_bound_fits_i32_handles_small_inputs() {
        assert!(upper_bound_fits_i32(1024, 100));
        assert!(upper_bound_fits_i32(1 << 20, 1024));
    }

    #[test]
    fn upper_bound_fits_i32_at_boundary() {
        // 2 * total + 7 * n == i32::MAX exactly
        let n = 1;
        let total = (i32::MAX as usize - 7) / 2;
        assert!(upper_bound_fits_i32(total, n));
        // One more byte tips us over
        assert!(!upper_bound_fits_i32(total + 1, n));
    }

    #[test]
    fn upper_bound_fits_i32_rejects_huge() {
        assert!(!upper_bound_fits_i32(usize::MAX / 4, 1000));
    }

    /// Regression for #7833: small inputs keep i32 codes offsets so the FSST
    /// output retains its compact layout. The matching i64 path is exercised
    /// only for inputs whose worst-case compressed size exceeds `i32::MAX`,
    /// which is too expensive to test directly; the boundary unit tests above
    /// cover the dispatch.
    #[test]
    fn fsst_compress_keeps_i32_offsets_for_small_inputs() {
        let array = VarBinArray::from_iter(
            [
                Some("The Greeks never said that the limit could not be overstepped"),
                Some("They said it existed and that whoever dared to exceed it was struck down"),
                Some("Nothing in present history can contradict them"),
            ],
            DType::Utf8(Nullability::NonNullable),
        );
        let compressor = fsst_train_compressor(&array);
        let len = array.len();
        let dtype = array.dtype().clone();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let total_uncompressed = array.bytes().len();
        let fsst = fsst_compress(
            &array,
            len,
            total_uncompressed,
            &dtype,
            &compressor,
            &mut ctx,
        );

        let codes_offsets_ptype = PType::try_from(fsst.codes().offsets().dtype()).unwrap();
        assert_eq!(codes_offsets_ptype, PType::I32);
    }
}
