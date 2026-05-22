// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::try_join;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::List;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::SplitRange;
use crate::layouts::list::ListLayout;
use crate::segments::SegmentSource;

/// Reader for [`ListLayout`].
///
/// Reads the underlying `elements`, `offsets`, and (optional) `validity` children in parallel
/// and reassembles them into a [`ListArray`](vortex_array::arrays::ListArray) before applying
/// the requested projection.
pub struct ListReader {
    layout: ListLayout,
    name: Arc<str>,
    _session: VortexSession,
    elements: LayoutReaderRef,
    offsets: LayoutReaderRef,
    validity: Option<LayoutReaderRef>,
}

impl ListReader {
    pub(super) fn try_new(
        layout: ListLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> VortexResult<Self> {
        let elements = layout.elements().new_reader(
            format!("{name}.elements").into(),
            Arc::clone(&segment_source),
            &session,
        )?;
        let offsets = layout.offsets().new_reader(
            format!("{name}.offsets").into(),
            Arc::clone(&segment_source),
            &session,
        )?;
        let validity = layout
            .validity()
            .map(|v| {
                v.new_reader(
                    format!("{name}.validity").into(),
                    Arc::clone(&segment_source),
                    &session,
                )
            })
            .transpose()?;

        Ok(Self {
            layout,
            name,
            _session: session,
            elements,
            offsets,
            validity,
        })
    }
}

impl LayoutReader for ListReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        // Offsets has one more row than the list itself but shares the list's chunking
        // structure, so it's the appropriate child to drive scan splits.
        self.offsets.register_splits(field_mask, split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        Ok(mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let elements_len = self.layout.elements().row_count();
        let elements_range = 0..elements_len;
        let elements_mask = MaskFuture::new_true(usize::try_from(elements_len)?);

        let row_count = usize::try_from(row_range.end - row_range.start)?;
        // The offsets child has n+1 entries; reading row_range maps to offsets in
        // [row_range.start..row_range.end + 1).
        let offsets_range = row_range.start..row_range.end + 1;
        let offsets_count = usize::try_from(offsets_range.end - offsets_range.start)?;

        let elements_fut = self
            .elements
            .projection_evaluation(&elements_range, &root(), elements_mask)?;
        let offsets_fut = self.offsets.projection_evaluation(
            &offsets_range,
            &root(),
            MaskFuture::new_true(offsets_count),
        )?;
        let validity_fut = self
            .validity
            .as_ref()
            .map(|v| v.projection_evaluation(row_range, &root(), MaskFuture::new_true(row_count)))
            .transpose()?;

        let nullability = self.layout.dtype().nullability();
        let expr = expr.clone();

        Ok(async move {
            let (elements, offsets) = try_join!(elements_fut, offsets_fut)?;
            let validity = match validity_fut {
                Some(fut) => Validity::Array(fut.await?),
                None => match nullability {
                    Nullability::Nullable => Validity::AllValid,
                    Nullability::NonNullable => Validity::NonNullable,
                },
            };

            // SAFETY: the layout was validated at write time, so offsets are monotonic,
            // non-nullable, integer, of length n+1.
            let array: ArrayRef =
                unsafe { Array::<List>::new_unchecked(elements, offsets, validity) }.into_array();

            let mask = mask.await?;
            let array = if !mask.all_true() {
                array.filter(mask)?
            } else {
                array
            };

            array.apply(&expr)
        }
        .boxed())
    }
}
