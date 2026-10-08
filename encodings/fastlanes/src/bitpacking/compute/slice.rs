// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::patches::Patches;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitWidthsView;
use crate::bitpacking::array::BitPackedArrayExt;

impl SliceReduce for BitPacked {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let BitWidthsView::Global(bit_width) = array.bit_widths() else {
            return Ok(None);
        };
        let patches = match array.patches() {
            None => None,
            Some(patches) if patches.is_cheap_to_slice() => patches.slice(range.clone())?,
            // Slicing these patches would read their buffers.
            Some(_) => return Ok(None),
        };

        Ok(Some(slice_bitpacked(array, bit_width, range, patches)?))
    }
}

impl SliceKernel for BitPacked {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let BitWidthsView::Global(bit_width) = array.bit_widths() else {
            return Ok(None);
        };
        let patches = array
            .patches()
            .map(|p| p.slice(range.clone()))
            .transpose()?
            .flatten();

        Ok(Some(slice_bitpacked(array, bit_width, range, patches)?))
    }
}

fn slice_bitpacked(
    array: ArrayView<'_, BitPacked>,
    bit_width: u8,
    range: Range<usize>,
    patches: Option<Patches>,
) -> VortexResult<ArrayRef> {
    let offset_start = range.start + array.offset() as usize;
    let offset_stop = range.end + array.offset() as usize;
    let offset = offset_start % 1024;
    let block_start = max(0, offset_start - offset);
    let block_stop = offset_stop.div_ceil(1024) * 1024;

    let encoded_start = (block_start / 8) * bit_width as usize;
    let encoded_stop = (block_stop / 8) * bit_width as usize;

    Ok(BitPacked::try_new(
        array.packed().slice(encoded_start..encoded_stop),
        array.dtype().as_ptype(),
        array.validity()?.slice(range.clone())?,
        patches,
        bit_width,
        range.len(),
        offset as u16,
    )?
    .into_array())
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SliceArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;

    use crate::BitPacked;
    use crate::BitPackedArrayExt;
    use crate::bitpack_compress::bitpack_encode;

    #[test]
    fn test_reduce_parent_returns_bitpacked_slice() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values = PrimitiveArray::from_iter(0u32..2048);
        let bitpacked = bitpack_encode(&values, 11, None, &mut ctx)?;

        let slice_array = SliceArray::new(bitpacked.clone().into_array(), 500..1500);

        let bitpacked_ref = bitpacked.into_array();
        let reduced = bitpacked_ref
            .reduce_parent(&slice_array.into_array(), 0)?
            .expect("expected slice kernel to execute");

        assert!(reduced.is::<BitPacked>());
        let reduced_bp = reduced.as_::<BitPacked>();
        assert_eq!(reduced_bp.offset(), 500);
        assert_eq!(reduced.len(), 1000);

        Ok(())
    }

    #[test]
    fn slice_reduces_with_host_patches() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values = PrimitiveArray::from_iter(
            (0u32..4096).map(|i| if i % 91 == 0 { 100_000 + i } else { i % 100 }),
        );
        let bitpacked = bitpack_encode(&values, 7, None, &mut ctx)?;
        assert!(bitpacked.patches().is_some(), "test setup expects patches");

        let sliced = bitpacked.into_array().slice(700..3500)?;

        assert!(sliced.is::<BitPacked>());
        assert_arrays_eq!(sliced, values.into_array().slice(700..3500)?, &mut ctx);
        Ok(())
    }
}
