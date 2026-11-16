// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayView;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Filter;
use crate::arrays::VarBinView;
use crate::arrays::filter::FilterArraySlotsExt;
use crate::arrays::varbinview::VarBinViewArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::VarBinViewBuilder;

pub(super) fn append_filter_to_builder(
    array: ArrayView<'_, Filter>,
    builder: &mut dyn ArrayBuilder,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if array.is_empty() {
        return Ok(());
    }

    if array.filter_mask().all_true() {
        return array.child().append_to_builder(builder, ctx);
    }

    if let Some(values) = array.child().as_opt::<VarBinView>()
        && let Some(builder) = builder.as_any_mut().downcast_mut::<VarBinViewBuilder>()
    {
        let selection = array
            .filter_mask()
            .values()
            .vortex_expect("empty and unfiltered masks are handled above");

        let validity = values
            .varbinview_validity()
            .filter(array.filter_mask())?
            .execute_mask(array.len(), ctx)?;
        builder.append_views_filtered(
            values
                .data_buffers()
                .iter()
                .map(|buffer| buffer.as_host().clone()),
            values.views(),
            selection,
            &validity,
        );

        return Ok(());
    }

    let canonical = array
        .array()
        .clone()
        .execute::<Canonical>(ctx)?
        .into_array();
    canonical.append_to_builder(builder, ctx)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::VortexSessionExecute;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::varbinview::BinaryView;
    use crate::arrays::varbinview::VarBinViewArrayExt;
    use crate::assert_arrays_eq;
    use crate::buffer::BufferHandle;
    use crate::builders::VarBinViewBuilder;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    const FIRST: &str = "the first selected long string";
    const LAST: &str = "the last selected long string";

    #[test]
    fn selected_null_views_do_not_dereference_payloads() -> VortexResult<()> {
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let values = VarBinViewArray::from_iter_nullable_str([
            Some(FIRST),
            None,
            Some("discarded"),
            Some(LAST),
        ]);
        let mut views = values.views().to_vec();
        views[1] = BinaryView::make_view(FIRST.as_bytes(), u32::MAX, u32::MAX);
        let values = VarBinViewArray::new_handle(
            BufferHandle::new_host(Buffer::from(views).into_byte_buffer()),
            Arc::clone(values.data_buffers()),
            values.dtype().clone(),
            values.varbinview_validity(),
        );
        let filtered = values.filter(Mask::from_iter([true, true, false, true]))?;
        let mut builder = VarBinViewBuilder::with_capacity_in(
            DType::Utf8(Nullability::Nullable),
            3,
            ctx.allocator().clone(),
        );
        filtered.append_to_builder(&mut builder, &mut ctx)?;
        assert_arrays_eq!(
            builder.finish_into_varbinview(),
            VarBinViewArray::from_iter_nullable_str([Some(FIRST), None, Some(LAST)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn selected_null_views_append_as_nulls() -> VortexResult<()> {
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let values = VarBinViewArray::from_iter_nullable_str([Some(FIRST), None, Some(LAST), None]);
        let filtered = values.filter(Mask::from_iter([false, true, false, true]))?;
        let mut builder = VarBinViewBuilder::with_capacity_in(
            DType::Utf8(Nullability::Nullable),
            2,
            ctx.allocator().clone(),
        );
        filtered.append_to_builder(&mut builder, &mut ctx)?;
        assert_arrays_eq!(
            builder.finish_into_varbinview(),
            VarBinViewArray::from_iter_nullable_str([None::<&str>, None]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn filtered_views_append_after_existing_values() -> VortexResult<()> {
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let values = VarBinViewArray::from_iter_nullable_str([
            Some(FIRST),
            None,
            Some("discarded"),
            Some(LAST),
            Some("x"),
        ]);
        let filtered = values.filter(Mask::from_iter([true, true, false, true, true]))?;
        let mut builder = VarBinViewBuilder::with_capacity_in(
            DType::Utf8(Nullability::Nullable),
            4,
            ctx.allocator().clone(),
        );
        builder.append_value("an existing long string");
        filtered.append_to_builder(&mut builder, &mut ctx)?;
        assert_arrays_eq!(
            builder.finish_into_varbinview(),
            VarBinViewArray::from_iter_nullable_str([
                Some("an existing long string"),
                Some(FIRST),
                None,
                Some(LAST),
                Some("x"),
            ]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn filtered_views_respect_builder_compaction() -> VortexResult<()> {
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let discarded = "discarded".repeat(1_024);
        let values = VarBinViewArray::from_iter_nullable_str([
            Some(discarded.as_str()),
            Some(FIRST),
            Some("also discarded"),
            None,
        ]);
        let filtered = values.filter(Mask::from_iter([false, true, false, true]))?;
        let mut builder = VarBinViewBuilder::with_compaction_in(
            DType::Utf8(Nullability::Nullable),
            2,
            1.0,
            ctx.allocator().clone(),
        );
        filtered.append_to_builder(&mut builder, &mut ctx)?;
        let result = builder.finish_into_varbinview();
        assert_arrays_eq!(
            result,
            VarBinViewArray::from_iter_nullable_str([Some(FIRST), None]),
            &mut ctx
        );
        let retained_bytes: usize = result
            .data_buffers()
            .iter()
            .map(|b| b.as_host().len())
            .sum();
        assert_eq!(retained_bytes, FIRST.len());
        Ok(())
    }
}
