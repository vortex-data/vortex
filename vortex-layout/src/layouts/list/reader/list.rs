// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reader implementation for ListLayouts containing lists with reified offsets buffers.

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::try_join_all;
use vortex_array::MaskFuture;
use vortex_array::arrays::ListArray;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_dtype::DType;
use vortex_dtype::FieldMask;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::layouts::list::ListLayout;

/// `LayoutReader` for the list layout holding
/// `List`-typed data.
pub struct ListReader {
    name: Arc<str>,
    layout: ListLayout,
    offsets: LayoutReaderRef,
    elements: LayoutReaderRef,
    validity: Option<LayoutReaderRef>,
}

impl ListReader {
    pub fn new(
        name: Arc<str>,
        layout: ListLayout,
        offsets: LayoutReaderRef,
        elements: LayoutReaderRef,
        validity: Option<LayoutReaderRef>,
    ) -> Self {
        Self {
            name,
            layout,
            offsets,
            elements,
            validity,
        }
    }

    // TODO(aduffy): does it make more sense to read offsets in their entirety, or
    //  read them in ranges?
}

const LIST_STEP_SIZE: usize = 1024;

#[async_trait]
impl LayoutReader for ListReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        &self.layout.dtype
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        let start = row_range.start;
        let end = start + self.layout.row_count;

        // TODO(aduffy): what you really want here is use the elements (possibly nested fields)
        //  to set the splits. But the elements row indices can't be mapped back to the table row
        //  indices, and we can't read the List offsets here. For now, we use a naive fixed-size
        //  split for scanning ListLayout.
        for idx in (start..=end).step_by(LIST_STEP_SIZE).skip(1) {
            splits.insert(idx);
        }

        splits.insert(end);

        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // TODO(aduffy): support scalar functions over Lists once we have a MapList expression.
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask_fut: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        // TODO(aduffy): pushdown IsNull / IsNotNull over validity
        Ok(mask_fut)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let row_range = row_range.clone();
        let row_count = self.layout.row_count;
        let elements_count = self.layout.elements_count;

        let offsets_range = 0..row_count + 1;
        let elements_range = 0..elements_count;
        let offsets_mask = MaskFuture::new_true(row_count as usize + 1);
        let elements_mask = MaskFuture::new_true(elements_count as usize);

        let offsets = self.offsets.clone();
        let elements = self.elements.clone();
        let validity = self.validity.clone();

        let expr = expr.clone();

        // TODO(aduffy): this is awful. But it's hard to splat the mask onto the elements space.
        Ok(Box::pin(async move {
            let offsets_fut =
                offsets.projection_evaluation(&offsets_range, &root(), offsets_mask)?;
            let elements_fut =
                elements.projection_evaluation(&elements_range, &root(), elements_mask)?;

            let mut tasks = vec![offsets_fut, elements_fut];

            if let Some(validity_reader) = validity.as_ref() {
                tasks.push(validity_reader.projection_evaluation(
                    &row_range,
                    &root(),
                    MaskFuture::new_true(row_count as usize),
                )?);
            }

            let tasks = try_join_all(tasks).await?;
            let offsets = tasks[0].clone();
            let elements = tasks[1].clone();
            let validity = if tasks.len() == 3 {
                Validity::Array(tasks[2].clone())
            } else {
                Validity::NonNullable
            };

            let mask = mask.await?;

            // Rebuild the List and execute the Mask operation directly.
            let list = ListArray::try_new(elements, offsets, validity)?;
            let list = vortex_array::compute::filter(list.as_ref(), &mask)?;

            // apply the projection expression
            expr.evaluate(&list)
        }))
    }
}

#[cfg(test)]
mod tests {
    //! List Layouts give us the opportunity to have deeply nested data accesses with the benefits
    //! of full vectorization.
    //!
    //! Say that we have the following schema: `{a:list({b:i32?}?)}`
    //!
    //! This can be written with the following layout tree (some simple nodes elided):
    //!
    //! ```text
    //! StructLayout
    //! |__ a: ListLayout
    //!     |__ offsets: FlatLayout
    //!     |__ elements: StructLayout
    //!         |__ b: ChunkedLayout
    //!             |_ [0]: FlatLayout
    //!             |_ [1]: FlatLayout
    //!             |_ ...
    //!             |_ [N]: FlatLayout
    //! ```
    //!
    //! The ListLayout
}
