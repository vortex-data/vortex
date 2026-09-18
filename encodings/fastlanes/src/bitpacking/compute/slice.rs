// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use crate::BitPackedArraySlotsExt;
use crate::bitpacking::array::BitPackedArrayExt;

impl SliceReduce for BitPacked {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // We cannot access buffers (to slice the patches).
        if array.patches().is_some() {
            return Ok(None);
        }

        let Some(widths) = array.materialized_chunk_widths()? else {
            return Ok(None);
        };
        let (chunks, _) = slice_chunks(array.offset(), &range);
        let encoded = widths.byte_offset(chunks.start)..widths.byte_offset(chunks.end);
        Ok(Some(slice_bitpacked(array, encoded, range, None)?))
    }
}

impl SliceKernel for BitPacked {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let patches = array
            .patches()
            .map(|p| p.slice(range.clone()))
            .transpose()?
            .flatten();

        let (chunks, _) = slice_chunks(array.offset(), &range);
        let widths = array.chunk_widths(ctx)?;
        let encoded = widths.byte_offset(chunks.start)..widths.byte_offset(chunks.end);
        Ok(Some(slice_bitpacked(array, encoded, range, patches)?))
    }
}

fn slice_bitpacked(
    array: ArrayView<'_, BitPacked>,
    encoded: Range<usize>,
    range: Range<usize>,
    patches: Option<Patches>,
) -> VortexResult<ArrayRef> {
    let (chunks, offset) = slice_chunks(array.offset(), &range);
    let chunk_start = chunks.start;
    let chunk_stop = chunks.end;

    Ok(BitPacked::try_new(
        array.packed().slice(encoded),
        array.dtype().as_ptype(),
        array.validity()?.slice(range.clone())?,
        patches,
        array.width_table().slice(chunk_start..chunk_stop)?,
        range.len(),
        offset as u16,
    )?
    .into_array())
}

fn slice_chunks(offset: u16, range: &Range<usize>) -> (Range<usize>, usize) {
    let start = range.start + offset as usize;
    let stop = range.end + offset as usize;
    (start / 1024..stop.div_ceil(1024), start % 1024)
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SliceArray;
    use vortex_error::VortexResult;

    use crate::BitPacked;
    use crate::bitpacking::bitpack_compress::bitpack_encode;

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
}
