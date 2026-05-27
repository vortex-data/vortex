// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::try_join;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ListArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
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
pub struct ListReader {
    layout: ListLayout,
    name: Arc<str>,
    session: VortexSession,
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
            session,
            elements,
            offsets,
            validity,
        })
    }

    fn fetch_offsets(&self, row_range: &Range<u64>) -> VortexResult<ArrayFuture> {
        // The offsets child has an extra entry, so reading `row_range` maps to offsets in
        // `[row_range.start..row_range.end + 1)`.
        let offsets_range = row_range.start..(row_range.end + 1);
        let offsets_count = usize::try_from(offsets_range.end - offsets_range.start)?;

        self.offsets.projection_evaluation(
            &offsets_range,
            &root(),
            MaskFuture::new_true(offsets_count),
        )
    }
}

/// Read `offsets[0]` and `offsets[-1]` and return the elements-buffer range they describe.
fn calculate_elements_range(
    offsets: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<Range<u64>> {
    if offsets.is_empty() {
        return Ok(0..0);
    }
    let mut exec_ctx = session.create_execution_ctx();
    let start = offsets
        .execute_scalar(0, &mut exec_ctx)?
        .as_primitive()
        .as_::<u64>()
        .vortex_expect("offset value fits in u64");
    let end = offsets
        .execute_scalar(offsets.len() - 1, &mut exec_ctx)?
        .as_primitive()
        .as_::<u64>()
        .vortex_expect("offset value fits in u64");
    Ok(start..end)
}

/// Subtract `first` from every offset so the resulting offsets index into a sliced
/// `elements[first..]` buffer starting at zero. The constant array is cast to the offsets' dtype.
fn rebase_offsets(offsets: ArrayRef, first: u64) -> VortexResult<ArrayRef> {
    let constant = ConstantArray::new(first, offsets.len())
        .into_array()
        .cast(offsets.dtype().clone())?;
    offsets.binary(constant, Operator::Sub)
}

fn create_validity(validity_array: Option<ArrayRef>, nullability: Nullability) -> Validity {
    match validity_array {
        Some(arr) => Validity::Array(arr),
        None => match nullability {
            Nullability::Nullable => Validity::AllValid,
            Nullability::NonNullable => Validity::NonNullable,
        },
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
        self.offsets
            .register_splits(field_mask, split_range, splits)?;
        if let Some(validity) = &self.validity {
            validity.register_splits(field_mask, split_range, splits)?;
        }
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        _mask: Mask,
    ) -> VortexResult<MaskFuture> {
        todo!()
    }

    fn filter_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        _mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        todo!()
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let offsets_fut = self.fetch_offsets(row_range)?;
        // Validity shares the list's row space, so the caller's mask is the right shape to
        // push down. Children get to fold it into their reads however they like.
        let validity_fut = self
            .validity
            .as_ref()
            .map(|v| v.projection_evaluation(row_range, &root(), mask.clone()))
            .transpose()?;

        let elements_reader = self.elements.clone();
        let session = self.session.clone();
        let nullability = self.layout.dtype().nullability();
        let expr = expr.clone();

        Ok(async move {
            // Fetch offsets and validity in parallel. Elements waits until we know
            // exactly which slice of the elements buffer it actually needs.
            let (offsets, validity_array) = try_join!(offsets_fut, async move {
                match validity_fut {
                    Some(fut) => fut.await.map(Some),
                    None => Ok(None),
                }
            },)?;

            // Bound the elements read using offsets[0] and offsets[-1]
            let elements_range = calculate_elements_range(&offsets, &session)?;

            // Rebase the offsets so they start at zero
            let rebased_offsets = rebase_offsets(offsets, elements_range.start)?;

            // Fetch only the elements we actually need.
            let elements_len = elements_range.end - elements_range.start;
            let elements = elements_reader
                .projection_evaluation(
                    &elements_range,
                    &root(),
                    MaskFuture::new_true(usize::try_from(elements_len)?),
                )?
                .await?;

            // Create ListArray
            let validity = create_validity(validity_array, nullability);
            let array = ListArray::try_new(elements, rebased_offsets, validity)?.into_array();

            // Apply mask and expression
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

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_buffer::buffer;

    use super::*;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::list::writer::ListLayoutStrategy;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;

    fn flat_list_strategy() -> ListLayoutStrategy {
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        ListLayoutStrategy::new(Arc::clone(&flat), Arc::clone(&flat), Arc::clone(&flat))
    }

    async fn write_layout<S: LayoutStrategy>(
        strategy: &S,
        array: ArrayRef,
    ) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef)> {
        let segments = Arc::new(TestSegments::default());
        let segments_ref: Arc<dyn SegmentSource> = Arc::<TestSegments>::clone(&segments);
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        let layout = strategy
            .write_stream(ArrayContext::empty(), segments, stream, eof, &SESSION)
            .await?;
        Ok((segments_ref, layout))
    }

    fn materialize_u32_array(array: ArrayRef) -> Vec<u32> {
        let mut ctx = SESSION.create_execution_ctx();
        array
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap()
            .as_slice::<u32>()
            .to_vec()
    }

    #[rstest]
    #[case::full(buffer![0u32, 2, 5, 5].into_array(), 0..5)]
    #[case::partial_slice(buffer![2u32, 5, 5, 8].into_array(), 2..8)]
    #[case::single_offset_is_empty(buffer![7u32].into_array(), 7..7)]
    #[case::u64_offsets(buffer![10u64, 12, 15, 15].into_array(), 10..15)]
    fn test_calculate_elements_range(
        #[case] offsets: ArrayRef,
        #[case] expected: Range<u64>,
    ) -> VortexResult<()> {
        assert_eq!(calculate_elements_range(&offsets, &SESSION)?, expected);
        Ok(())
    }

    #[test]
    fn calculate_elements_range_empty_offsets() -> VortexResult<()> {
        let offsets = PrimitiveArray::empty::<u32>(NonNullable).into_array();
        assert_eq!(calculate_elements_range(&offsets, &SESSION)?, 0..0);
        Ok(())
    }

    #[rstest]
    #[case::first_zero_is_identity(buffer![0u32, 2, 5, 5].into_array(), 0, vec![0, 2, 5, 5])]
    #[case::subtracts_first(buffer![3u32, 5, 8].into_array(), 3, vec![0, 2, 5])]
    fn test_rebase_offsets(
        #[case] offsets: ArrayRef,
        #[case] first: u64,
        #[case] expected: Vec<u32>,
    ) -> VortexResult<()> {
        let rebased = rebase_offsets(offsets, first)?;
        assert_eq!(materialize_u32_array(rebased), expected);
        Ok(())
    }

    #[tokio::test]
    async fn fetch_offsets_includes_extra_endpoint() -> VortexResult<()> {
        let list = ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5].into_array(),
            buffer![0u32, 2, 4, 5].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let (segments, layout) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION)?;
        let reader = reader
            .as_any()
            .downcast_ref::<ListReader>()
            .expect("ListReader");

        // row_range 1..3 should pull 3 offsets (indices 1, 2, 3) — the +1 endpoint matters.
        let offsets = reader.fetch_offsets(&(1..3))?.await?;
        assert_eq!(materialize_u32_array(offsets), vec![2, 4, 5]);

        // row_range 0..3 pulls all 4 offsets.
        let offsets = reader.fetch_offsets(&(0..3))?.await?;
        assert_eq!(materialize_u32_array(offsets), vec![0, 2, 4, 5]);
        Ok(())
    }

    fn create_basic_list_array(nullable: bool) -> ArrayRef {
        let validity = if nullable {
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array())
        } else {
            Validity::NonNullable
        };

        ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5].into_array(),
            buffer![0u32, 2, 4, 5].into_array(),
            validity,
        )
        .expect("array is valid")
        .into_array()
    }

    #[rstest]
    #[case::full_range(0..3, false)]
    #[case::partial_start(0..2, false)]
    #[case::partial_end(1..3, false)]
    #[case::middle_single(1..2, false)]
    #[case::empty_range(1..1, false)]
    #[case::full_range_null(0..3, true)]
    #[tokio::test]
    async fn projection_evaluation_round_trips(
        #[case] row_range: Range<u64>,
        #[case] nullable: bool,
    ) -> VortexResult<()> {
        let list = create_basic_list_array(nullable);

        let len = usize::try_from(row_range.end - row_range.start)?;
        let (segments, layout) = write_layout(&flat_list_strategy(), list.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION)?;

        let result = reader
            .projection_evaluation(&row_range, &root(), MaskFuture::new_true(len))?
            .await?;

        let expected =
            list.slice(usize::try_from(row_range.start)?..usize::try_from(row_range.end)?)?;
        assert_arrays_eq!(result, expected);
        Ok(())
    }
}
