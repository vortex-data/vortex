// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reader implementation for ListLayouts containing lists with reified offsets buffers.

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::try_join_all;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::ToCanonical;
use vortex_array::arrays::ListArray;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::DType;
use vortex_dtype::FieldMask;
use vortex_dtype::IntegerPType;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_mask::MaskMut;

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
}

const LIST_STEP_SIZE: usize = 1024;
const DENSE_THRESHOLD: f64 = 0.5;

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
        let expr = expr.clone();
        let layout = self.layout.clone();

        let row_range = row_range.clone();
        let offsets = self.offsets.clone();
        let elements = self.elements.clone();
        let validity = self.validity.clone();

        Ok(Box::pin(async move {
            let mask = mask.await?;

            let list = if mask.density() <= DENSE_THRESHOLD {
                read_sparse(&row_range, mask, offsets, elements, validity).await?
            } else {
                read_dense(&layout, &row_range, mask, offsets, elements, validity).await?
            };

            // apply the projection expression
            expr.evaluate(&list)
        }))
    }
}

async fn read_sparse(
    row_range: &Range<u64>,
    mask: Mask,
    offsets: Arc<dyn LayoutReader>,
    elements: Arc<dyn LayoutReader>,
    validity: Option<Arc<dyn LayoutReader>>,
) -> VortexResult<ArrayRef> {
    // Reading List with a sparse mask forces us to sequence IOs for the offsets and the elements,
    // but it reduces the # of IO requests necessary to read the data by first planning a mask
    // in the elements space so that we only load a subset of element segments.

    let offsets_range = row_range.start..row_range.end + 1;
    let offsets_mask = Mask::new_true(mask.len() + 1);

    let offsets = offsets
        .projection_evaluation(&offsets_range, &root(), MaskFuture::ready(offsets_mask))?
        .await?
        .to_primitive();

    let (elements_range, elements_mask, new_offsets) =
        match_each_integer_ptype!(offsets.ptype(), |P| {
            // Offsets are always in-bounds. Or at least they should be.
            let offsets = offsets.into_buffer::<P>();

            let (elements_range, elements_mask) =
                build_elements_mask_and_range(offsets.as_slice(), &mask);

            // rebuild the offsets to only cover the elements that were loaded.
            let new_offsets = build_offsets(offsets, &mask).into_array();

            (elements_range, elements_mask, new_offsets)
        });

    // Read the elements
    let elements_array = elements
        .projection_evaluation(&elements_range, &root(), MaskFuture::ready(elements_mask))?
        .await?;

    let validity = if let Some(reader) = validity {
        Validity::Array(
            reader
                .projection_evaluation(row_range, &root(), MaskFuture::ready(mask))?
                .await?,
        )
    } else {
        Validity::NonNullable
    };

    Ok(ListArray::try_new(elements_array, new_offsets, validity)?.into_array())
}

fn build_offsets<P: IntegerPType>(offsets: Buffer<P>, mask: &Mask) -> Buffer<P> {
    debug_assert_eq!(offsets.len(), mask.len() + 1);

    let mut buffer = BufferMut::with_capacity(mask.true_count() + 1);

    match mask.slices() {
        AllOr::All => {
            buffer.extend_trusted(offsets.into_iter());
        }
        AllOr::None => {
            buffer.push(P::zero());
        }
        AllOr::Some(slices) => {
            let mut last = P::zero();
            buffer.push(P::zero());

            for &(start, end) in slices {
                for i in start..end {
                    let list_start = offsets[i];
                    let list_end = offsets[i + 1];
                    last += list_end - list_start;
                    buffer.push(last);
                }
            }
        }
    }

    buffer.freeze()
}

fn build_elements_mask_and_range<P: IntegerPType>(
    offsets: &[P],
    mask: &Mask,
) -> (Range<u64>, Mask) {
    // If we have some elements, we need to expand to the row-range in question.
    let first_elem = offsets[0].as_();
    let last_elem = offsets[offsets.len() - 1].as_();

    let mut elements_mask = MaskMut::new_false(last_elem - first_elem);

    match mask.indices() {
        AllOr::All => {
            elements_mask = MaskMut::new_true(last_elem - first_elem);
        }
        AllOr::None => {
            // nothing to do, no elements
        }
        AllOr::Some(indices) => {
            for &index in indices {
                // Start needs to be referenced to first_elem.
                let start = offsets[index].as_() - first_elem;
                let end = offsets[index + 1].as_() - first_elem;
                for elem_index in start..end {
                    elements_mask.set(elem_index);
                }
            }
        }
    }

    let range = (first_elem as u64)..(last_elem as u64);
    let mask = elements_mask.freeze();

    (range, mask)
}

async fn read_dense(
    layout: &ListLayout,
    row_range: &Range<u64>,
    mask: Mask,
    offsets: Arc<dyn LayoutReader>,
    elements: Arc<dyn LayoutReader>,
    validity: Option<Arc<dyn LayoutReader>>,
) -> VortexResult<ArrayRef> {
    // Reading List with dense mask, we read ALL offsets for row_range + ALL elements, concurrently,
    // and filter the result.
    let row_count = layout.row_count;
    let elements_count = layout.elements_count;

    let offsets_range = row_range.start..(row_range.end + 1);
    let elements_range = 0..elements_count;
    let offsets_mask = MaskFuture::new_true(row_count as usize + 1);
    let elements_mask = MaskFuture::new_true(elements_count as usize);

    let offsets_fut = offsets.projection_evaluation(&offsets_range, &root(), offsets_mask)?;
    let elements_fut = elements.projection_evaluation(&elements_range, &root(), elements_mask)?;

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

    // Rebuild the List and execute the Mask operation directly.
    let list = ListArray::try_new(elements, offsets, validity)?;
    vortex_array::compute::filter(list.as_ref(), &mask)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::root;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_buffer::buffer_mut;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::compressed::CompressingStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::list::writer::ListStrategy;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        VortexSession::empty()
            .with::<ArraySession>()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_tokio()
    });

    async fn make_layout() -> (LayoutRef, Arc<dyn SegmentSource>) {
        let compress_then_flat = Arc::new(CompressingStrategy::new_btrblocks(
            FlatLayoutStrategy::default(),
            false,
        ));

        let writer = ListStrategy::new(
            compress_then_flat.clone(),
            compress_then_flat,
            Arc::new(FlatLayoutStrategy::default()),
        );

        let segments = Arc::new(TestSegments::default());
        let ctx = ArrayContext::empty();

        let (sequence_ptr, eof) = SequenceId::root().split();

        let elements = VarBinArray::from_iter(
            [
                Some("one"),
                Some("two"),
                Some("three"),
                None,
                Some("five"),
                Some("six"),
            ],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();

        let offsets = buffer![0, 0, 3, 4, 6].into_array();

        let list = ListArray::new(elements, offsets, Validity::NonNullable).to_array();

        let layout = writer
            .write_stream(
                ctx,
                segments.clone(),
                list.to_array_stream().sequenced(sequence_ptr),
                eof,
                SESSION.handle(),
            )
            .await
            .unwrap();

        (layout, segments as Arc<dyn SegmentSource>)
    }

    #[tokio::test]
    async fn test_full() {
        let (layout, segments) = make_layout().await;

        let reader = layout
            .new_reader(Arc::from("test_data"), segments, &*SESSION)
            .unwrap();

        let full = reader
            .projection_evaluation(&(0..4), &root(), MaskFuture::new_true(4))
            .unwrap()
            .await
            .unwrap();

        let expected = make_list(vec![
            vec![],
            vec![Some("one"), Some("two"), Some("three")],
            vec![None],
            vec![Some("five"), Some("six")],
        ]);

        assert_arrays_eq!(full, expected);
    }

    #[tokio::test]
    async fn test_dense() {
        let (layout, segments) = make_layout().await;

        let reader = layout
            .new_reader(Arc::from("test_data"), segments, &*SESSION)
            .unwrap();

        let dense = reader
            .projection_evaluation(
                &(1..4),
                &root(),
                MaskFuture::ready(Mask::from_iter([false, true, true])),
            )
            .unwrap()
            .await
            .unwrap();

        let expected = make_list(vec![vec![None], vec![Some("five"), Some("six")]]);

        assert_arrays_eq!(dense, expected);
    }

    #[tokio::test]
    async fn test_sparse() {
        let (layout, segments) = make_layout().await;

        let reader = layout
            .new_reader(Arc::from("test_data"), segments, &*SESSION)
            .unwrap();

        let sparse = reader
            .projection_evaluation(
                &(1..4),
                &root(),
                MaskFuture::ready(Mask::from_iter([false, true, false])),
            )
            .unwrap()
            .await
            .unwrap();

        let expected = make_list(vec![vec![None]]);

        assert_arrays_eq!(sparse, expected);
    }

    fn make_list(data: Vec<Vec<Option<&str>>>) -> ArrayRef {
        let mut offsets = buffer_mut![0u32];
        let mut prev = 0;
        for list in data.iter() {
            prev += list.len();
            offsets.push(prev as u32);
        }

        let elements = VarBinArray::from_iter(
            data.into_iter().flatten(),
            DType::Utf8(Nullability::Nullable),
        )
        .to_array();

        ListArray::new(
            elements,
            offsets.freeze().into_array(),
            Validity::NonNullable,
        )
        .into_array()
    }
}
