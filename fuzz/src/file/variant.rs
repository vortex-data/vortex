// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Unstructured;
use parquet_variant::Variant;
use vortex_array::dtype::Nullability;

use super::FileRoundTrip;
use crate::FUZZ_FILE_ARRAY_MAX_LEN;
use crate::variant::VariantRow;

// The default WriteStrategyBuilder row block size.
const ROW_BLOCK: usize = 8192;

pub(super) fn arbitrary_file(u: &mut Unstructured<'_>) -> arbitrary::Result<FileRoundTrip> {
    let (len, split_at) = match u.int_in_range(0..=2)? {
        0 => (ROW_BLOCK + 2, Some(ROW_BLOCK - 1)),
        1 => {
            let len = u.int_in_range(ROW_BLOCK + 1..=FUZZ_FILE_ARRAY_MAX_LEN)?;
            (len, Some(u.int_in_range(1..=len - 1)?))
        }
        _ => {
            let len = u.int_in_range(ROW_BLOCK + 1..=FUZZ_FILE_ARRAY_MAX_LEN)?;
            (len, None)
        }
    };
    let pool_len = u.int_in_range(1..=16)?;
    // Bound the payload before repeating it thousands of times.
    let mut payload = Unstructured::new(u.bytes(u.len().min(256))?);
    let pool = VariantRow::arbitrary_rows(&mut payload, Nullability::Nullable, pool_len)?;
    let absent = VariantRow::new(None);
    let present_null = VariantRow::new(Some(Variant::Null));
    let marker = VariantRow::new(Some(Variant::from(true)));
    let row_at = |i: usize| match i {
        i if i == ROW_BLOCK - 2 => &marker,
        i if i == ROW_BLOCK - 1 => &absent,
        ROW_BLOCK => &present_null,
        _ => &pool[i % pool.len()],
    };
    Ok(FileRoundTrip::Variant {
        array: VariantRow::array((0..len).map(row_at), Nullability::Nullable),
        split_at,
    })
}

#[cfg(test)]
mod tests {
    use arbitrary::Arbitrary;
    use arbitrary::Unstructured;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::FileRoundTrip;

    #[test]
    fn roundtrip_variant_across_row_block() -> VortexResult<()> {
        // One repeated false value, plus the three distinct boundary rows.
        let seed = [0xff, 0, 0, 1, 1, 0];
        let input = FileRoundTrip::arbitrary(&mut Unstructured::new(&seed))
            .map_err(|e| vortex_err!("invalid Variant file seed: {e}"))?;
        let FileRoundTrip::Variant { array, split_at } = &input else {
            return Err(vortex_err!("seed must generate a Variant column"));
        };
        assert_eq!(array.len(), 8194);
        assert_eq!(*split_at, Some(8191));
        assert!(input.check());
        Ok(())
    }
}
