// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ScalarFn;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::Ref;
use crate::buffer::BufferHandle;
use crate::executor::ExecutionCtx;
use crate::kernel::ExecuteParentKernel;
use crate::scalar_fn::fns::substring::Substring;
use crate::scalar_fn::fns::substring::parse_byte_range;

#[derive(Default, Debug)]
pub(crate) struct SubstringVarBinView;

impl ExecuteParentKernel<VarBinView> for SubstringVarBinView {
    type Parent = ExactScalarFn<Substring>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, VarBinView>,
        parent: ScalarFnArrayView<'_, Substring>,
        child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let children = scalar_fn_array.children();

        let (byte_start, byte_length) = parse_byte_range(&children[1], children.get(2))?;

        let validity = array.validity()?;
        let dtype = array.dtype().clone();
        let views = array.views();

        let new_views: Buffer<BinaryView> =
            Buffer::from_trusted_len_iter(views.iter().enumerate().map(|(idx, view)| {
                if validity.is_null(idx).unwrap_or(false) {
                    return BinaryView::empty_view();
                }

                if view.is_inlined() {
                    let data = view.as_inlined().value();
                    let start = byte_start.min(data.len());
                    let end = byte_length
                        .map(|l| (start + l).min(data.len()))
                        .unwrap_or(data.len());
                    BinaryView::new_inlined(&data[start..end])
                } else {
                    let r = view.as_view();
                    let total_len = r.size as usize;
                    let start = byte_start.min(total_len);
                    let end = byte_length
                        .map(|l| (start + l).min(total_len))
                        .unwrap_or(total_len);
                    let new_len = end - start;
                    let new_offset = r.offset as usize + start;

                    if new_len <= BinaryView::MAX_INLINED_SIZE {
                        let buf = array.buffer(r.buffer_index as usize);
                        BinaryView::new_inlined(&buf[new_offset..new_offset + new_len])
                    } else {
                        let buf = array.buffer(r.buffer_index as usize);
                        let prefix: [u8; 4] = buf[new_offset..new_offset + 4]
                            .try_into()
                            .ok()
                            .vortex_expect("prefix must be exactly 4 bytes");
                        BinaryView::from(Ref {
                            size: u32::try_from(new_len)
                                .vortex_expect("substring length must fit in u32"),
                            prefix,
                            buffer_index: r.buffer_index,
                            offset: u32::try_from(new_offset)
                                .vortex_expect("substring offset must fit in u32"),
                        })
                    }
                }
            }));

        // SAFETY: we reuse existing valid data buffers and construct views with correct
        // offsets/sizes/prefixes derived from the original validated views.
        let result = unsafe {
            VarBinViewArray::new_handle_unchecked(
                BufferHandle::new_host(new_views.into_byte_buffer()),
                Arc::clone(array.data_buffers()),
                dtype,
                validity,
            )
        };

        Ok(Some(result.into_array()))
    }
}
