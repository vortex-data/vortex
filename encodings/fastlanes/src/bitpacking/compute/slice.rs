// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArray;

impl SliceReduce for BitPacked {
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
    use vortex_array::DynArray;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SliceArray;
    use vortex_error::VortexResult;

    use crate::BitPacked;
    use crate::bitpack_compress::bitpack_encode;

    #[test]
    fn test_reduce_parent_returns_bitpacked_slice() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter(0u32..2048);
        let bitpacked = bitpack_encode(&values, 11, None)?;

        let slice_array = SliceArray::new(bitpacked.clone().into_array(), 500..1500);

        let bitpacked_ref = bitpacked.into_array();
        let reduced = bitpacked_ref
            .vtable()
            .reduce_parent(&bitpacked_ref, &slice_array.into_array(), 0)?
            .expect("expected slice kernel to execute");

        assert!(reduced.is::<BitPacked>());
        let reduced_bp = reduced.as_::<BitPacked>();
        assert_eq!(reduced_bp.offset(), 500);
        assert_eq!(reduced.len(), 1000);

        Ok(())
    }
}
