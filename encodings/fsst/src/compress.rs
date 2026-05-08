// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fsst::Compressor;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;

use crate::FSST;
use crate::FSSTArray;

/// Compress a string array using FSST.
pub fn fsst_compress<A: ArrayAccessor<[u8]>>(
    strings: A,
    len: usize,
    dtype: &DType,
    compressor: &Compressor,
    ctx: &mut ExecutionCtx,
) -> FSSTArray {
    strings.with_iterator(|iter| fsst_compress_iter(iter, len, dtype.clone(), compressor, ctx))
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
    lines.extend(iter.flatten());
    Compressor::train(&lines)
}

/// Most strings are small in practice. If we encounter a larger string, we reallocate
/// the buffer to hold enough capacity for the worst-case compressed value.
const DEFAULT_BUFFER_LEN: usize = 1024 * 1024;

/// Compress an iterator of bytestrings into an FSST array. Codes-offsets are
/// `i32` for small (typical) inputs, promoted to `i64` once cumulative
/// compressed bytes would exceed `i32::MAX`.
pub fn fsst_compress_iter<'a, I>(
    iter: I,
    len: usize,
    dtype: DType,
    compressor: &Compressor,
    ctx: &mut ExecutionCtx,
) -> FSSTArray
where
    I: Iterator<Item = Option<&'a [u8]>>,
{
    let mut buffer = Vec::with_capacity(DEFAULT_BUFFER_LEN);
    let mut data: BufferMut<u8> = BufferMut::empty();
    let mut validity = BitBufferMut::with_capacity(len);
    let mut uncompressed_lengths: BufferMut<i32> = BufferMut::with_capacity(len);
    let mut offsets_i32: BufferMut<i32> = BufferMut::with_capacity(len + 1);
    offsets_i32.push(0);
    let mut offsets_i64: Option<BufferMut<i64>> = None;

    for string in iter {
        match string {
            None => {
                validity.append_false();
                uncompressed_lengths.push(0);
            }
            Some(s) => {
                validity.append_true();
                uncompressed_lengths
                    .push(s.len().try_into().vortex_expect("string length fits i32"));
                let target = 2 * s.len() + 7;
                if target > buffer.len() {
                    buffer.reserve(target - buffer.len());
                }
                // SAFETY: buffer holds at least 2*s.len()+7 bytes per the FSST contract.
                unsafe { compressor.compress_into(s, &mut buffer) };
                data.extend_from_slice(&buffer);
            }
        }

        let off = data.len();
        if offsets_i64.is_none() && off > i32::MAX as usize {
            let mut o64 = BufferMut::with_capacity(len + 1);
            for i in 0..offsets_i32.len() {
                o64.push(i64::from(offsets_i32[i]));
            }
            offsets_i64 = Some(o64);
        }
        match &mut offsets_i64 {
            Some(o64) => o64.push(off as i64),
            None => offsets_i32.push(i32::try_from(off).vortex_expect("offset fits i32")),
        }
    }

    let offsets = match offsets_i64 {
        Some(o64) => PrimitiveArray::new(o64.freeze(), Validity::NonNullable),
        None => PrimitiveArray::new(offsets_i32.freeze(), Validity::NonNullable),
    };
    offsets
        .statistics()
        .set(Stat::IsSorted, Precision::Exact(true.into()));
    let codes = VarBinArray::new(
        offsets.into_array(),
        data.freeze(),
        DType::Binary(dtype.nullability()),
        Validity::from_bit_buffer(validity.freeze(), dtype.nullability()),
    );

    FSST::try_new(
        dtype,
        Buffer::copy_from(compressor.symbol_table()),
        Buffer::<u8>::copy_from(compressor.symbol_lengths()),
        codes,
        uncompressed_lengths.into_array(),
        ctx,
    )
    .vortex_expect("FSST parts must be valid")
}

#[cfg(test)]
mod tests {
    use fsst::CompressorBuilder;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::seq::IndexedRandom;
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
            DType::Utf8(Nullability::NonNullable),
            &compressor,
            &mut ctx,
        );

        let decoded = compressed.execute_scalar(0, &mut ctx).unwrap();
        let expected = Scalar::utf8(big_string, Nullability::NonNullable);

        assert_eq!(decoded, expected);
    }

    fn assert_codes_offsets_ptype(
        array: &VarBinArray,
        compressor: &fsst::Compressor,
        expected: PType,
    ) {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let fsst = fsst_compress(array, array.len(), array.dtype(), compressor, &mut ctx);
        assert_eq!(fsst.len(), array.len());
        let actual = PType::try_from(fsst.codes().offsets().dtype()).unwrap();
        assert_eq!(actual, expected);
    }

    /// Regression for #7833: typical inputs keep `i32` codes-offsets.
    #[test]
    fn fsst_compress_keeps_i32_offsets_for_small_inputs() {
        let array = VarBinArray::from_iter(
            [Some("hello world"), Some("hello rust"), Some("hello vortex")],
            DType::Utf8(Nullability::NonNullable),
        );
        let compressor = fsst_train_compressor(&array);
        assert_codes_offsets_ptype(&array, &compressor, PType::I32);
    }

    /// Regression for #7833: in-loop promotion when cumulative compressed bytes
    /// cross `i32::MAX`. Gated to CI runs (skipped when `CI` is unset; opt-out
    /// with `VORTEX_SKIP_SLOW_TESTS=1`); peak ~4.5 GiB.
    #[test_with::env(CI)]
    #[test_with::no_env(VORTEX_SKIP_SLOW_TESTS)]
    fn fsst_compress_promotes_in_loop_via_data_size() {
        // High-entropy ASCII: pseudo-random data resists FSST symbol-table
        // compression, so output stays close to input size and crosses i32::MAX.
        const STRING_LEN: usize = 64 * 1024;
        const TOTAL_BYTES: usize = (1usize << 31) + (256 << 20); // ~2.25 GiB
        const N: usize = TOTAL_BYTES / STRING_LEN;
        const POOL_LEN: usize = 64 * 1024 * 1024;

        // Printable ASCII so the result is valid UTF-8.
        const ALPHABET: &[u8; 95] =
            b" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";

        let mut rng = StdRng::seed_from_u64(0xC0DE_C011_B711);
        let pool: Vec<u8> = (0..POOL_LEN)
            .map(|_| *ALPHABET.choose(&mut rng).unwrap())
            .collect();
        let array = VarBinArray::from_iter(
            (0..N).map(|i| {
                let off = i.wrapping_mul(31337) % (POOL_LEN - STRING_LEN);
                Some(&pool[off..off + STRING_LEN])
            }),
            DType::Utf8(Nullability::NonNullable),
        );

        let compressor = fsst_train_compressor(&array);
        assert_codes_offsets_ptype(&array, &compressor, PType::I64);
    }
}
