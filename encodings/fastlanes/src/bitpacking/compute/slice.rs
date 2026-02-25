// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::SliceReduce;
use vortex_error::VortexResult;

use crate::BitPackedArray;
use crate::BitPackedVTable;

impl SliceReduce for BitPackedVTable {
    fn slice(array: &BitPackedArray, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let offset_start = range.start + array.offset() as usize;
        let offset_stop = range.end + array.offset() as usize;
        let offset = offset_start % 1024;
        let block_start = max(0, offset_start - offset);
        let block_stop = offset_stop.div_ceil(1024) * 1024;

        let encoded_start = (block_start / 8) * array.bit_width() as usize;
        let encoded_stop = (block_stop / 8) * array.bit_width() as usize;

        // slice the buffer using the encoded start/stop values
        // SAFETY: slicing packed values without decoding preserves invariants
        Ok(Some(unsafe {
            BitPackedArray::new_unchecked(
                array.packed().slice(encoded_start..encoded_stop),
                array.dtype().clone(),
                array.validity()?.slice(range.clone())?,
                array
                    .patches()
                    .map(|p| p.slice(range.clone()))
                    .transpose()?
                    .flatten(),
                array.bit_width(),
                range.len(),
                offset as u16,
            )
            .into_array()
        }))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::Array;
    use vortex_array::arrays::SliceReduce;
    use vortex_array::arrays::SliceVTable;
    use vortex_error::VortexResult;

    use crate::BitPackedVTable;
    use crate::bitpack_compress::bitpack_encode;

    #[test]
    fn test_slice_returns_bitpacked() -> VortexResult<()> {
        let values = vortex_array::arrays::PrimitiveArray::from_iter(0u32..2048);
        let bitpacked = bitpack_encode(&values, 11, None)?;

        let result =
            BitPackedVTable::slice(&bitpacked, 500..1500)?.expect("expected slice to succeed");

        assert!(result.is::<BitPackedVTable>());
        let result_bp = result.as_::<BitPackedVTable>();
        assert_eq!(result_bp.offset(), 500);
        assert_eq!(result.len(), 1000);

        Ok(())
    }

    #[test]
    fn test_slice_via_array_trait() -> VortexResult<()> {
        let values = vortex_array::arrays::PrimitiveArray::from_iter(0u32..2048);
        let bitpacked = bitpack_encode(&values, 11, None)?;

        let sliced = bitpacked.as_ref().slice(500..1500)?;

        // After optimize, the SliceArray should have been reduced away.
        assert!(
            !sliced.is::<SliceVTable>(),
            "expected SliceReduce to eliminate the SliceArray wrapper"
        );
        assert!(sliced.is::<BitPackedVTable>());
        assert_eq!(sliced.len(), 1000);

        Ok(())
    }
}
