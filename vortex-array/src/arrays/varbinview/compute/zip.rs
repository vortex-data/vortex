// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::builders::DeduplicatedBuffers;
use crate::builders::LazyBitBufferBuilder;
use crate::scalar_fn::fns::zip::ZipKernel;

// A dedicated VarBinView zip kernel that builds the result directly by adjusting views and validity,
// instead of routing through the generic builder (which would redo buffer lookups per mask slice).
impl ZipKernel for VarBinView {
    fn zip(
        if_true: ArrayView<'_, VarBinView>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<VarBinView>() else {
            return Ok(None);
        };

        if !if_true.dtype().eq_ignore_nullability(if_false.dtype()) {
            vortex_bail!("input arrays to zip must have the same dtype");
        }

        let len = if_true.len();
        let dtype = if_true
            .dtype()
            .union_nullability(if_false.dtype().nullability());

        // build buffer lookup tables for both arrays, these map from the original buffer idx
        // to the new buffer index in the result array
        let mut buffers = DeduplicatedBuffers::default();
        let true_lookup =
            buffers.extend_from_iter(if_true.data_buffers().iter().map(|b| b.as_host().clone()));
        let false_lookup =
            buffers.extend_from_iter(if_false.data_buffers().iter().map(|b| b.as_host().clone()));

        let mut views_builder = BufferMut::<BinaryView>::with_capacity(len);
        let mut validity_builder = LazyBitBufferBuilder::new(len);

        let true_validity = if_true.validity_mask();
        let false_validity = if_false.validity_mask();

        let mask = mask.try_to_mask_fill_null_false(ctx)?;
        let if_false_view = if_false;
        match mask.slices() {
            AllOr::All => push_range(
                if_true,
                &true_lookup,
                &true_validity,
                0..len,
                &mut views_builder,
                &mut validity_builder,
            ),
            AllOr::None => push_range(
                if_false_view,
                &false_lookup,
                &false_validity,
                0..len,
                &mut views_builder,
                &mut validity_builder,
            ),
            AllOr::Some(slices) => {
                let mut pos = 0;
                for (start, end) in slices {
                    if pos < *start {
                        push_range(
                            if_false_view,
                            &false_lookup,
                            &false_validity,
                            pos..*start,
                            &mut views_builder,
                            &mut validity_builder,
                        );
                    }
                    push_range(
                        if_true,
                        &true_lookup,
                        &true_validity,
                        *start..*end,
                        &mut views_builder,
                        &mut validity_builder,
                    );
                    pos = *end;
                }
                if pos < len {
                    push_range(
                        if_false_view,
                        &false_lookup,
                        &false_validity,
                        pos..len,
                        &mut views_builder,
                        &mut validity_builder,
                    );
                }
            }
        }

        let validity = validity_builder.finish_with_nullability(dtype.nullability());

        // SAFETY: views are built with adjusted buffer indices, validity tracked alongside;
        // buffers come from `DeduplicatedBuffers`, dtype/nullability preserved.
        let array = unsafe {
            VarBinViewArray::new_unchecked(
                views_builder.freeze(),
                buffers.finish(),
                dtype,
                validity,
            )
        };

        Ok(Some(array.into_array()))
    }
}

fn push_range(
    array: ArrayView<'_, VarBinView>,
    buffer_lookup: &[u32],
    validity: &Mask,
    range: Range<usize>,
    views_builder: &mut BufferMut<BinaryView>,
    validity_builder: &mut LazyBitBufferBuilder,
) {
    let views = array.views();

    match validity.bit_buffer() {
        AllOr::All => {
            for idx in range {
                push_view(
                    views[idx],
                    buffer_lookup,
                    true,
                    views_builder,
                    validity_builder,
                );
            }
        }
        AllOr::None => {
            for _ in range {
                push_view(
                    BinaryView::empty_view(),
                    buffer_lookup,
                    false,
                    views_builder,
                    validity_builder,
                );
            }
        }
        AllOr::Some(bit_buffer) => {
            for idx in range {
                let is_valid = bit_buffer.value(idx);
                push_view(
                    views[idx],
                    buffer_lookup,
                    is_valid,
                    views_builder,
                    validity_builder,
                );
            }
        }
    }
}

fn push_view(
    view: BinaryView,
    buffer_lookup: &[u32],
    is_valid: bool,
    views_builder: &mut BufferMut<BinaryView>,
    validity_builder: &mut LazyBitBufferBuilder,
) {
    if !is_valid {
        views_builder.push(BinaryView::empty_view());
        validity_builder.append_null();
        return;
    }

    let adjusted = if view.is_inlined() {
        view
    } else {
        let view_ref = view.as_view();
        view_ref
            .with_buffer_and_offset(
                buffer_lookup[view_ref.buffer_index as usize],
                view_ref.offset,
            )
            .into()
    };

    views_builder.push(adjusted);
    validity_builder.append_non_null();
}

#[cfg(test)]
mod tests {
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::accessor::ArrayAccessor;
    use crate::arrays::VarBinViewArray;
    use crate::builtins::ArrayBuiltins;
    use crate::canonical::ToCanonical;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    #[test]
    fn zip_varbinview_kernel_zips() {
        let a = VarBinViewArray::from_iter(
            [
                Some("aaaaaaaaaaaaa_long"), // outlined
                Some("short"),
                None,
                Some("bbbbbbbbbbbbbbbb_long"),
                Some("tiny"),
                Some("cccccccccccccccc_long"),
            ],
            DType::Utf8(Nullability::Nullable),
        );

        let b = VarBinViewArray::from_iter(
            [
                Some("dddddddddddddddd_long"),
                Some("eeeeeeeeeeeeeeee_long"),
                Some("ffff"),
                Some("gggggggggggggggg_long"),
                None,
                Some("hhhhhhhhhhhhhhhh_long"),
            ],
            DType::Utf8(Nullability::Nullable),
        );

        let mask = Mask::from_iter([true, false, true, false, false, true]);

        let zipped = mask
            .clone()
            .into_array()
            .zip(a.into_array(), b.into_array())
            .unwrap()
            .to_varbinview();

        let values = zipped.with_iterator(|it| {
            it.map(|v| v.map(|bytes| String::from_utf8(bytes.to_vec()).unwrap()))
                .collect::<Vec<_>>()
        });

        assert_eq!(
            values,
            vec![
                Some("aaaaaaaaaaaaa_long".to_string()),
                Some("eeeeeeeeeeeeeeee_long".to_string()),
                None,
                Some("gggggggggggggggg_long".to_string()),
                None,
                Some("cccccccccccccccc_long".to_string())
            ]
        );
        assert_eq!(zipped.len(), mask.len());
        assert_eq!(zipped.dtype(), &DType::Utf8(Nullability::Nullable));
    }
}
