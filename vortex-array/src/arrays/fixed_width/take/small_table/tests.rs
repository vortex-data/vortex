// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;

use super::super::HAS_AVX2;
use super::avx2::take_avx2;
use super::avx512::HAS_AVX512_VBMI;
use super::avx512::take_avx512;

fn lookup(values: &[u8], codes: &[u8], vbmi: bool) -> Option<Buffer<u8>> {
    let allocator = BufferAllocatorRef::statically_allocated();
    if vbmi {
        if !*HAS_AVX512_VBMI {
            return None;
        }
        // SAFETY: Features are detected above; tests provide 1..=32 one-byte values.
        Some(unsafe { take_avx512(values, codes, &allocator) })
    } else {
        if !*HAS_AVX2 {
            return None;
        }
        // SAFETY: AVX2 is detected above; tests provide 1..=32 one-byte values.
        Some(unsafe {
            if values.len() <= 16 {
                take_avx2::<_, false>(values, codes, &allocator)
            } else {
                take_avx2::<_, true>(values, codes, &allocator)
            }
        })
    }
}

#[rstest]
fn register_lookup_matches_scalar(
    #[values(false, true)] vbmi: bool,
    #[values(1, 2, 3, 4, 8, 15, 16, 17, 31, 32)] cardinality: u8,
    #[values(0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1025)] len: usize,
) {
    let values = (0..cardinality)
        .map(|i| i.wrapping_mul(37).wrapping_add(129))
        .collect::<Vec<_>>();
    // Offset slices exercise unaligned loads, with all valid codes and non-identity values.
    let codes = (0..cardinality)
        .rev()
        .cycle()
        .take(len + 1)
        .collect::<Vec<_>>();
    let codes = &codes[1..];
    let expected = codes
        .iter()
        .map(|&i| values[usize::from(i)])
        .collect::<Vec<_>>();
    if let Some(actual) = lookup(&values, codes, vbmi) {
        assert_eq!(actual.as_slice(), expected);
    }
}

#[rstest]
fn register_lookup_rejects_invalid_codes(
    #[values(false, true)] vbmi: bool,
    #[values(2, 16, 17, 32)] cardinality: u8,
    #[values(0, 31, 32, 63, 64, 127, 128)] offset: usize,
    #[values(0, 32, 127, 128, 255)] invalid: u8,
) {
    if (vbmi && !*HAS_AVX512_VBMI) || (!vbmi && !*HAS_AVX2) {
        return;
    }
    let values = (0..cardinality).collect::<Vec<_>>();
    let mut codes = vec![0; 129];
    codes[offset] = invalid.max(cardinality);
    let result = std::panic::catch_unwind(|| lookup(&values, &codes, vbmi));
    assert!(result.is_err());
}
