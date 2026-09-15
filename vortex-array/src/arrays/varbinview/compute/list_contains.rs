// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_utils::aliases::hash_set::HashSet;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::VarBinView;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::collect_bits;
use crate::scalar_fn::fns::list_contains::ListContainsElementKernel;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::list_contains::ListContainsSet;

/// Single-pass membership of UTF-8 or binary needles in a constant set, hashed by bytes.
impl ListContainsElementKernel for VarBinView {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(set) = ListContainsSet::try_new(list, element.dtype(), options) else {
            return Ok(None);
        };

        let bytes: HashSet<Vec<u8>> = set.elements().iter().filter_map(element_bytes).collect();

        let buffers: Vec<&[u8]> = (0..element.data_buffers().len())
            .map(|idx| element.buffer(idx).as_slice())
            .collect();
        let bits = collect_bits(
            element.views(),
            |view: BinaryView| {
                let value: &[u8] = if view.is_inlined() {
                    view.as_inlined().value()
                } else {
                    let reference = view.as_view();
                    &buffers[reference.buffer_index as usize][reference.as_range()]
                };
                bytes.contains(value)
            },
            ctx.allocator(),
        );

        set.finish(bits, element.validity()?).map(Some)
    }
}

/// The bytes of a non-null UTF-8 or binary element.
fn element_bytes(element: &Scalar) -> Option<Vec<u8>> {
    match element.dtype() {
        DType::Utf8(_) => element
            .as_utf8()
            .value()
            .map(|s| s.as_str().as_bytes().to_vec()),
        _ => element.as_binary().value().map(|b| b.to_vec()),
    }
}
