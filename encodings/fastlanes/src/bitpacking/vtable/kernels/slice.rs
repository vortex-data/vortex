// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::SliceVTable;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::matchers::Exact;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::BitPackedArray;
use crate::BitPackedVTable;
use crate::bitpack_decompress::unpack_array;

/// Kernel to execute slicing fused with bit-packed decoding.
#[derive(Debug)]
pub(crate) struct BitPackingSliceKernel;

impl ExecuteParentKernel<BitPackedVTable> for BitPackingSliceKernel {
    type Parent = Exact<SliceVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::new()
    }

    fn execute_parent(
        &self,
        array: &BitPackedArray,
        parent: &SliceArray,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Device-resident buffers cannot perform the patch index search needed for slicing yet
        if !array.is_host() {
            return Ok(None);
        }

        let range = parent.slice_range().clone();

        let sliced = array.slice(range)?;
        let sliced_bp = sliced.as_::<BitPackedVTable>();

        let primitive =
            match_each_integer_ptype!(sliced_bp.ptype(), |P| { unpack_array(sliced_bp, ctx)? });

        Ok(Some(primitive.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SliceArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::bitpack_compress::bitpack_encode;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_slice_then_decode_optimization() -> VortexResult<()> {
        // Create an array with patches
        let values = PrimitiveArray::from_iter((0u16..2048).map(|x| x % 512 + x / 512 * 1000));
        let bitpacked = bitpack_encode(&values, 9, None)?;
        assert!(bitpacked.patches().is_some(), "Should have patches");

        // Wrap in SliceArray
        let slice_array = SliceArray::new(bitpacked.into_array(), 500..1500);

        // Execute - this should trigger our kernel
        let mut ctx = SESSION.create_execution_ctx();
        let result = slice_array.into_array().execute::<Canonical>(&mut ctx)?;

        // Verify result matches expected
        let expected: Vec<u16> = (500u16..1500).map(|x| x % 512 + x / 512 * 1000).collect();
        assert_arrays_eq!(result.into_primitive(), PrimitiveArray::from_iter(expected));

        Ok(())
    }

    #[test]
    fn test_slice_without_patches() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter(0u32..1024);
        let bitpacked = bitpack_encode(&values, 10, None)?;
        assert!(bitpacked.patches().is_none(), "Should not have patches");

        let slice_array = SliceArray::new(bitpacked.into_array(), 100..900);

        let mut ctx = SESSION.create_execution_ctx();
        let result = slice_array.into_array().execute::<Canonical>(&mut ctx)?;

        assert_arrays_eq!(
            result.into_primitive(),
            PrimitiveArray::from_iter(100u32..900)
        );

        Ok(())
    }

    #[test]
    fn test_slice_across_chunk_boundaries() -> VortexResult<()> {
        // Create array spanning multiple 1024-element chunks
        let values = PrimitiveArray::from_iter(0u32..4096);
        let bitpacked = bitpack_encode(&values, 12, None)?;

        // Slice across chunk boundaries
        let slice_array = SliceArray::new(bitpacked.into_array(), 900..2200);

        let mut ctx = SESSION.create_execution_ctx();
        let result = slice_array.into_array().execute::<Canonical>(&mut ctx)?;

        assert_arrays_eq!(
            result.into_primitive(),
            PrimitiveArray::from_iter(900u32..2200)
        );

        Ok(())
    }

    #[test]
    fn test_slice_with_patches_at_boundaries() -> VortexResult<()> {
        // Create array where patches occur at slice boundaries
        let values: Vec<u16> = (0..3072)
            .map(|i| {
                if i == 500 || i == 1000 || i == 1500 || i == 2000 {
                    60000 // Force patches at specific positions
                } else {
                    (i % 256) as u16
                }
            })
            .collect();
        let array = PrimitiveArray::from_iter(values.clone());
        let bitpacked = bitpack_encode(&array, 8, None)?;
        assert!(bitpacked.patches().is_some());

        // Slice that includes some patches
        let slice_array = SliceArray::new(bitpacked.into_array(), 600..1800);

        let mut ctx = SESSION.create_execution_ctx();
        let result = slice_array.into_array().execute::<Canonical>(&mut ctx)?;

        let expected: Vec<u16> = values[600..1800].to_vec();
        assert_arrays_eq!(result.into_primitive(), PrimitiveArray::from_iter(expected));

        Ok(())
    }

    #[test]
    fn test_nested_slice() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter(0u32..2048);
        let bitpacked = bitpack_encode(&values, 11, None)?;

        // First slice
        let slice1 = SliceArray::new(bitpacked.into_array(), 200..1800);
        // Nested slice
        let slice2 = SliceArray::new(slice1.into_array(), 100..1400);

        let mut ctx = SESSION.create_execution_ctx();
        let result = slice2.into_array().execute::<Canonical>(&mut ctx)?;

        // Should be elements 300..1600 from original
        assert_arrays_eq!(
            result.into_primitive(),
            PrimitiveArray::from_iter(300u32..1600)
        );

        Ok(())
    }

    #[test]
    fn test_small_slice() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter(0u16..1024);
        let bitpacked = bitpack_encode(&values, 10, None)?;

        // Very small slice
        let slice_array = SliceArray::new(bitpacked.into_array(), 500..505);

        let mut ctx = SESSION.create_execution_ctx();
        let result = slice_array.into_array().execute::<Canonical>(&mut ctx)?;

        assert_arrays_eq!(
            result.into_primitive(),
            PrimitiveArray::from_iter(500u16..505)
        );

        Ok(())
    }

    #[test]
    fn test_full_slice() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter(0u32..1024);
        let bitpacked = bitpack_encode(&values, 10, None)?;

        // Slice the entire array
        let slice_array = SliceArray::new(bitpacked.into_array(), 0..1024);

        let mut ctx = SESSION.create_execution_ctx();
        let result = slice_array.into_array().execute::<Canonical>(&mut ctx)?;

        assert_arrays_eq!(
            result.into_primitive(),
            PrimitiveArray::from_iter(0u32..1024)
        );

        Ok(())
    }
}
