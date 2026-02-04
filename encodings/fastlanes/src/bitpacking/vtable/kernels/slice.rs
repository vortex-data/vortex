// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::SliceVTable;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;

use crate::BitPackedArray;
use crate::BitPackedVTable;

/// Kernel to execute slicing fused with bit-packed decoding.
#[derive(Debug)]
pub(crate) struct BitPackingSliceKernel;

impl ExecuteParentKernel<BitPackedVTable> for BitPackingSliceKernel {
    type Parent = SliceVTable;

    fn execute_parent(
        &self,
        array: &BitPackedArray,
        parent: &SliceArray,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        assert!(
            array.is_host(),
            "BitPackingSliceKernel requires host-resident buffers"
        );

        let range = parent.slice_range().clone();
        <BitPackedVTable as VTable>::slice(array, range)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::Array;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::SliceArray;
    use vortex_array::session::ArraySession;
    use vortex_array::vtable::VTable;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPackedVTable;
    use crate::bitpack_compress::bitpack_encode;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_execute_parent_returns_bitpacked_slice() -> VortexResult<()> {
        let values = vortex_array::arrays::PrimitiveArray::from_iter(0u32..2048);
        let bitpacked = bitpack_encode(&values, 11, None)?;

        let slice_array = SliceArray::new(bitpacked.clone().into_array(), 500..1500);

        let mut ctx = SESSION.create_execution_ctx();
        let reduced = <BitPackedVTable as VTable>::execute_parent(
            &bitpacked,
            &slice_array.into_array(),
            0,
            &mut ctx,
        )?
        .expect("expected slice kernel to execute");

        assert!(reduced.is::<BitPackedVTable>());
        let reduced_bp = reduced.as_::<BitPackedVTable>();
        assert_eq!(reduced_bp.offset(), 500);
        assert_eq!(reduced.len(), 1000);

        Ok(())
    }
}
