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
use vortex_error::vortex_ensure;

use crate::BitPacked;
use crate::BitWidths;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking::array::BitPackedArrayExt;

impl SliceReduce for BitPacked {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // Slicing per-block bit widths reads block boundaries, which needs the kernel.
        let BitWidths::Global(bit_width) = array.bit_widths() else {
            return Ok(None);
        };
        // We cannot access buffers (to slice the patches).
        if array.patches().is_some() {
            return Ok(None);
        }

        Ok(Some(slice_bitpacked(array, bit_width, range, None)?))
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

        Ok(Some(match array.bit_widths() {
            BitWidths::Global(bit_width) => slice_bitpacked(array, bit_width, range, patches)?,
            BitWidths::Blocked(block_offsets) => {
                slice_blocked(array, &block_offsets, range, patches, ctx)?
            }
        }))
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

/// Keep the blocks that `range` overlaps.
///
/// Block boundaries are relative to the first one, so the overlapped blocks' boundaries are the
/// new block offsets as they are, and the packed buffer is cut between the first and last of them.
///
/// E.g. rows 1500..4100 of an array with block offsets `[b0, b1, b2, b3, b4, b5]` keep blocks 1 to
/// 4: the block offsets become `[b1, b2, b3, b4, b5]`, the packed buffer is cut to bytes
/// `b1 - b0..b5 - b0`, and the offset is 476 (row 1500 is 476 rows into block 1).
fn slice_blocked(
    array: ArrayView<'_, BitPacked>,
    block_offsets: &ArrayRef,
    range: Range<usize>,
    patches: Option<Patches>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let offset_start = range.start + array.offset() as usize;
    let offset_stop = range.end + array.offset() as usize;
    let first_block = offset_start / FL_CHUNK_SIZE;
    let end_block = offset_stop.div_ceil(FL_CHUNK_SIZE);

    let mut boundary = |block: usize| u64::try_from(&block_offsets.execute_scalar(block, ctx)?);
    let base = boundary(0)?;
    let start = boundary(first_block)?;
    let end = boundary(end_block)?;
    // Release builds don't validate boundaries on construction, so check them before slicing.
    vortex_ensure!(
        base <= start && start <= end && end - base <= array.packed().len() as u64,
        "Block boundaries {start} and {end} are outside the packed buffer (base {base})"
    );

    // Both differences are at most the packed length, so they fit in `usize`.
    let packed = array
        .packed()
        .slice((start - base) as usize..(end - base) as usize);
    Ok(BitPacked::try_new_with_block_offsets(
        packed,
        array.dtype().as_ptype(),
        array.validity()?.slice(range.clone())?,
        patches,
        block_offsets.slice(first_block..end_block + 1)?,
        range.len(),
        (offset_start % FL_CHUNK_SIZE) as u16,
    )?
    .into_array())
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SliceArray;
    use vortex_array::arrays::slice::SliceKernel;
    use vortex_array::assert_arrays_eq;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;

    use crate::BitPacked;
    use crate::BitPackedArray;
    use crate::BitPackedArrayExt;
    use crate::BitWidths;
    use crate::FoR;
    use crate::bitpack_compress::bitpack_encode;
    use crate::bitpack_compress::bitpack_encode_blocked;
    use crate::test::SESSION;

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

    /// Values whose 1024-value blocks need 2 to 6 bits.
    fn drifting() -> PrimitiveArray {
        PrimitiveArray::from_iter((0..5000u32).map(|i| i % (4 << (i / 1024))))
    }

    /// Pack `values` at `bit_widths`, with FoR-encoded block offsets if `encoded_offsets`.
    fn blocked(
        values: &PrimitiveArray,
        bit_widths: &[u8],
        encoded_offsets: bool,
    ) -> VortexResult<BitPackedArray> {
        let array = bitpack_encode_blocked(
            values,
            bit_widths,
            None,
            &mut SESSION.create_execution_ctx(),
        )?;
        if !encoded_offsets {
            return Ok(array);
        }
        let BitWidths::Blocked(block_offsets) = array.bit_widths() else {
            vortex_bail!("expected block offsets");
        };
        BitPacked::try_new_with_block_offsets(
            array.packed().clone(),
            array.dtype().as_ptype(),
            array.validity()?,
            array.patches(),
            FoR::try_new(
                block_offsets.clone(),
                Scalar::zero_value(block_offsets.dtype()),
            )?
            .into_array(),
            array.len(),
            array.offset(),
        )
    }

    /// Slice `array` with the kernel, which must keep it blocked.
    fn slice_kernel(array: &ArrayRef, range: Range<usize>) -> VortexResult<BitPackedArray> {
        let sliced = <BitPacked as SliceKernel>::slice(
            array.as_::<BitPacked>(),
            range,
            &mut SESSION.create_execution_ctx(),
        )?
        .ok_or_else(|| vortex_err!("expected the slice kernel to slice"))?;
        let sliced: BitPackedArray = sliced
            .try_downcast()
            .map_err(|a| vortex_err!("expected BitPacked, got {}", a.encoding_id()))?;
        assert!(!sliced.bit_widths().is_global());
        Ok(sliced)
    }

    #[rstest]
    #[case::whole(0..5000)]
    #[case::within_block(1100..1900)]
    #[case::across_blocks(1500..4100)]
    #[case::block_aligned(1024..3072)]
    #[case::to_end(3000..5000)]
    #[case::empty_at_boundary(2048..2048)]
    #[case::empty_within_block(2100..2100)]
    fn slice_blocked(
        #[case] range: Range<usize>,
        #[values(false, true)] encoded_offsets: bool,
    ) -> VortexResult<()> {
        let values = drifting();
        let array = blocked(&values, &[2, 3, 4, 5, 6], encoded_offsets)?.into_array();
        let sliced = slice_kernel(&array, range.clone())?;
        assert_arrays_eq!(
            sliced,
            values.into_array().slice(range)?,
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn slice_blocked_keeps_only_the_overlapped_blocks() -> VortexResult<()> {
        let array = blocked(&drifting(), &[2, 3, 4, 5, 6], false)?.into_array();
        // Rows 1500..4100 overlap blocks 1 to 4, packed at 3, 4, 5 and 6 bits.
        let sliced = slice_kernel(&array, 1500..4100)?;
        assert_eq!(sliced.offset(), 476);
        assert_eq!(sliced.packed().len(), 128 * (3 + 4 + 5 + 6));
        let BitWidths::Blocked(block_offsets) = sliced.bit_widths() else {
            vortex_bail!("expected block offsets");
        };
        assert_arrays_eq!(
            block_offsets,
            PrimitiveArray::from_iter([256u16, 640, 1152, 1792, 2560]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn slice_blocked_with_nulls_and_patches() -> VortexResult<()> {
        let values = PrimitiveArray::from_option_iter(
            (0..5000u32).map(|i| (i % 7 != 0).then_some(i % (4 << (i / 1024)))),
        );
        // The last block packs values up to 63 at 3 bits, so it has patches.
        let array = blocked(&values, &[2, 3, 4, 5, 3], false)?.into_array();
        assert!(array.as_::<BitPacked>().patches().is_some());
        let sliced = slice_kernel(&array, 3000..4900)?;
        assert!(sliced.patches().is_some());
        assert_arrays_eq!(
            sliced,
            values.into_array().slice(3000..4900)?,
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn slice_blocked_twice() -> VortexResult<()> {
        let values = drifting();
        let array = blocked(&values, &[2, 3, 4, 5, 6], true)?.into_array();
        let once = slice_kernel(&array, 1500..4900)?.into_array();
        let twice = slice_kernel(&once, 300..2000)?;
        assert_eq!(twice.offset(), 776);
        assert_arrays_eq!(
            twice,
            values.into_array().slice(1800..3500)?,
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }
}
