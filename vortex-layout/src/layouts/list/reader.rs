// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use futures::try_join;
use vortex_array::Array;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::expr::Expression;
use vortex_array::expr::get_item;
use vortex_array::expr::root;
use vortex_dtype::DType;
use vortex_dtype::FieldMask;
use vortex_dtype::FieldName;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::LazyReaderChildren;
use crate::layouts::list::ListLayout;
use crate::segments::SegmentSource;

pub struct ListReader {
    layout: ListLayout,
    name: Arc<str>,
    lazy_children: LazyReaderChildren,
    session: VortexSession,
}

impl ListReader {
    pub(super) fn try_new(
        layout: ListLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> VortexResult<Self> {
        let mut dtypes: Vec<DType> = Vec::new();
        let mut names: Vec<Arc<str>> = Vec::new();

        if layout.dtype().is_nullable() {
            dtypes.push(DType::Bool(Nullability::NonNullable));
            names.push(Arc::from("validity"));
        }

        match layout.dtype() {
            DType::List(element_dtype, _) => {
                dtypes.push(DType::Primitive(PType::U64, Nullability::NonNullable));
                names.push(Arc::from("offsets"));
                dtypes.push((**element_dtype).clone());
                names.push(Arc::from("elements"));
            }
            DType::FixedSizeList(element_dtype, ..) => {
                dtypes.push((**element_dtype).clone());
                names.push(Arc::from("elements"));
            }
            _ => vortex_bail!("Expected list dtype, got {}", layout.dtype()),
        }

        let lazy_children = LazyReaderChildren::new(
            layout.children().clone(),
            dtypes,
            names,
            segment_source,
            session.clone(),
        );

        Ok(Self {
            layout,
            name,
            lazy_children,
            session,
        })
    }

    fn validity(&self) -> VortexResult<Option<&LayoutReaderRef>> {
        self.dtype()
            .is_nullable()
            .then(|| self.lazy_children.get(0))
            .transpose()
    }

    fn offsets(&self) -> VortexResult<&LayoutReaderRef> {
        let idx = if self.dtype().is_nullable() { 1 } else { 0 };
        self.lazy_children.get(idx)
    }

    fn elements(&self) -> VortexResult<&LayoutReaderRef> {
        let idx = match self.dtype() {
            DType::List(..) => {
                if self.dtype().is_nullable() {
                    2
                } else {
                    1
                }
            }
            DType::FixedSizeList(..) => {
                if self.dtype().is_nullable() {
                    1
                } else {
                    0
                }
            }
            _ => return Err(vortex_err!("Expected list dtype, got {}", self.dtype())),
        };
        self.lazy_children.get(idx)
    }

    /// Creates a future that will produce a slice of this list array.
    ///
    /// The produced slice may have a projection applied to its elements.
    fn list_slice_future(
        &self,
        row_range: Range<u64>,
        element_expr: &Expression,
    ) -> VortexResult<ArrayFuture> {
        let dtype = self.dtype().clone();
        let validity_fut = self
            .validity()?
            .map(|reader| {
                let len = usize::try_from(row_range.end - row_range.start)
                    .vortex_expect("row range must fit in usize");
                reader.projection_evaluation(&row_range, &root(), MaskFuture::new_true(len))
            })
            .transpose()?;

        match dtype {
            DType::List(_, list_nullability) => {
                let offsets_reader = self.offsets()?.clone();
                let elements_reader = self.elements()?.clone();
                let row_range_clone = row_range.clone();
                let element_expr = element_expr.clone();

                Ok(Box::pin(async move {
                    let row_len = usize::try_from(row_range_clone.end - row_range_clone.start)
                        .vortex_expect("row range must fit in usize");

                    let offsets_row_range = row_range_clone.start..row_range_clone.end + 1;
                    let offsets_len = row_len + 1;
                    let offsets_fut = offsets_reader.projection_evaluation(
                        &offsets_row_range,
                        &root(),
                        MaskFuture::new_true(offsets_len),
                    )?;

                    let (offsets, validity) = try_join!(offsets_fut, async move {
                        match validity_fut {
                            Some(v) => v.await.map(Some),
                            None => Ok(None),
                        }
                    })?;

                    let offsets = offsets.to_primitive();
                    let offsets_slice = offsets.as_slice::<u64>();
                    let base = *offsets_slice.first().unwrap_or(&0u64);
                    let end = *offsets_slice.last().unwrap_or(&base);

                    let elements_row_range = base..end;
                    let elements_len = usize::try_from(end - base)
                        .vortex_expect("element range must fit in usize");
                    let elements = elements_reader.projection_evaluation(
                        &elements_row_range,
                        &element_expr,
                        MaskFuture::new_true(elements_len),
                    )?;

                    let elements = elements.await?;

                    let normalized_offsets = vortex_array::arrays::PrimitiveArray::from_iter(
                        offsets_slice.iter().map(|v| *v - base),
                    )
                    .into_array();

                    let validity = match (list_nullability, validity) {
                        (Nullability::NonNullable, _) => {
                            vortex_array::validity::Validity::NonNullable
                        }
                        (Nullability::Nullable, Some(v)) => {
                            vortex_array::validity::Validity::Array(v)
                        }
                        (Nullability::Nullable, None) => vortex_array::validity::Validity::AllValid,
                    };

                    Ok(ListArray::try_new(elements, normalized_offsets, validity)?.into_array())
                }))
            }
            DType::FixedSizeList(_, list_size, list_nullability) => {
                let elements_reader = self.elements()?.clone();
                let row_range_clone = row_range.clone();
                let element_expr = element_expr.clone();

                Ok(Box::pin(async move {
                    let row_len_u64 = row_range_clone.end - row_range_clone.start;
                    let row_len =
                        usize::try_from(row_len_u64).vortex_expect("row range must fit in usize");

                    let list_size_u64 = u64::from(list_size);
                    let element_start = row_range_clone
                        .start
                        .checked_mul(list_size_u64)
                        .ok_or_else(|| vortex_err!("FixedSizeList element start overflow"))?;
                    let element_end = row_range_clone
                        .end
                        .checked_mul(list_size_u64)
                        .ok_or_else(|| vortex_err!("FixedSizeList element end overflow"))?;

                    let elements_row_range = element_start..element_end;
                    let elements_len = usize::try_from(element_end - element_start)
                        .vortex_expect("element range must fit in usize");
                    let elements_fut = elements_reader.projection_evaluation(
                        &elements_row_range,
                        &element_expr,
                        MaskFuture::new_true(elements_len),
                    )?;

                    let (elements, validity) = try_join!(elements_fut, async move {
                        match validity_fut {
                            Some(v) => v.await.map(Some),
                            None => Ok(None),
                        }
                    })?;

                    let validity = match (list_nullability, validity) {
                        (Nullability::NonNullable, _) => {
                            vortex_array::validity::Validity::NonNullable
                        }
                        (Nullability::Nullable, Some(v)) => {
                            vortex_array::validity::Validity::Array(v)
                        }
                        (Nullability::Nullable, None) => vortex_array::validity::Validity::AllValid,
                    };

                    Ok(
                        FixedSizeListArray::try_new(elements, list_size, validity, row_len)?
                            .into_array(),
                    )
                }))
            }
            _ => Err(vortex_err!("Expected list dtype, got {}", dtype)),
        }
    }
}

impl LayoutReader for ListReader {
    fn name(&self) -> &Arc<str> {
        &self.name
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
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        splits.insert(row_range.end);

        match self.dtype() {
            DType::FixedSizeList(_, list_size, _) => {
                let list_size_u64 = u64::from(*list_size);

                let element_start = row_range
                    .start
                    .checked_mul(list_size_u64)
                    .ok_or_else(|| vortex_err!("FixedSizeList element start overflow"))?;
                let element_end = row_range
                    .end
                    .checked_mul(list_size_u64)
                    .ok_or_else(|| vortex_err!("FixedSizeList element end overflow"))?;

                let element_range = element_start..element_end;
                let mut element_splits = BTreeSet::new();
                self.elements()?.register_splits(
                    field_mask,
                    &element_range,
                    &mut element_splits,
                )?;

                // Convert element splits back to row splits, but only when the element split
                // is aligned to a row boundary.
                for element_split in element_splits {
                    if element_split % list_size_u64 != 0 {
                        continue;
                    }

                    let row_split = element_split / list_size_u64;
                    if row_split > row_range.start && row_split < row_range.end {
                        splits.insert(row_split);
                    }
                }
            }
            DType::List(..) => {
                let offsets_end = row_range
                    .end
                    .checked_add(1)
                    .ok_or_else(|| vortex_err!("List offsets end overflow"))?;
                let offsets_range = row_range.start..offsets_end;

                let mut offsets_splits = BTreeSet::new();
                self.offsets()?
                    .register_splits(field_mask, &offsets_range, &mut offsets_splits)?;

                // Convert splits in the offsets array back to row splits.
                //
                // The offsets array has length = rows + 1, so a split at offset index `i`
                // corresponds to a split in rows at `i - 1`.
                for offsets_split in offsets_splits {
                    if offsets_split <= row_range.start {
                        continue;
                    }

                    let row_split = offsets_split - 1;
                    if row_split > row_range.start && row_split < row_range.end {
                        splits.insert(row_split);
                    }
                }
            }
            _ => {}
        }

        Ok(())
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
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let row_range = row_range.clone();
        let expr = expr.clone();
        let session = self.session.clone();

        let list_fut = self.list_slice_future(row_range.clone(), &root())?;

        Ok(MaskFuture::new(
            usize::try_from(row_range.end - row_range.start)
                .vortex_expect("row range must fit in usize"),
            async move {
                let (array, mask) = try_join!(list_fut, mask)?;
                if mask.all_false() {
                    return Ok(mask);
                }

                let array = array.apply(&expr)?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.execute::<Mask>(&mut ctx)?;

                Ok(mask.bitand(&array_mask))
            },
        ))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        // If the expression is a simple element projection, we can push it down to the elements.
        //
        // NOTE: `vortex.get_item_list` is a temporary list-of-struct projection expression;
        // when pushing down we construct the element projection and pass it into the elements reader.
        let (is_pushdown, element_expr) = if expr.id().as_ref() == "vortex.get_item_list"
            && expr.child(0).id().as_ref() == "vortex.root"
        {
            let field_name = expr
                .options()
                .as_any()
                .downcast_ref::<FieldName>()
                .vortex_expect("vortex.get_item_list options must be a FieldName");
            (true, get_item(field_name.clone(), root()))
        } else if expr.id().as_ref() == "vortex.select" {
            (true, expr.clone())
        } else {
            (false, root())
        };

        let row_range = row_range.clone();
        let expr = expr.clone();
        let list_fut = self.list_slice_future(row_range, &element_expr)?;

        Ok(Box::pin(async move {
            let (mut array, mask) = try_join!(list_fut, mask)?;

            // Apply the selection mask before applying the expression, matching `FlatReader`.
            if !mask.all_true() {
                array = array.filter(mask)?;
            }

            if is_pushdown {
                Ok(array)
            } else {
                array.apply(&expr)
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use futures::stream;
    use vortex_array::Array;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::ListArray;
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType;
    use vortex_io::runtime::single::block_on;

    use crate::LayoutStrategy;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::list::writer::ListStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialStreamAdapter;
    use crate::sequence::SequentialStreamExt as _;
    use crate::test::SESSION;

    #[test]
    fn register_splits_fixed_size_list_maps_element_splits_to_rows() {
        let ctx = ArrayContext::empty();

        let segments = Arc::new(TestSegments::default());

        let list_size: u32 = 2;

        let chunk1_elements = buffer![1i32, 2, 3, 4].into_array();
        let chunk1 = FixedSizeListArray::try_new(
            chunk1_elements,
            list_size,
            vortex_array::validity::Validity::NonNullable,
            2,
        )
        .unwrap()
        .into_array();

        let chunk2_elements = buffer![5i32, 6, 7, 8].into_array();
        let chunk2 = FixedSizeListArray::try_new(
            chunk2_elements,
            list_size,
            vortex_array::validity::Validity::NonNullable,
            2,
        )
        .unwrap()
        .into_array();

        let list_dtype = chunk1.dtype().clone();

        let elements_strategy = Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()));
        let strategy = ListStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
            elements_strategy,
        );

        let (mut sequence_id, eof) = SequenceId::root().split();
        let layout = block_on(|handle| {
            strategy.write_stream(
                ctx,
                segments.clone(),
                SequentialStreamAdapter::new(
                    vortex_dtype::DType::FixedSizeList(
                        Arc::new(vortex_dtype::DType::Primitive(PType::I32, NonNullable)),
                        list_size,
                        NonNullable,
                    ),
                    stream::iter([
                        Ok((sequence_id.advance(), chunk1)),
                        Ok((sequence_id.advance(), chunk2)),
                    ]),
                )
                .sendable(),
                eof,
                handle,
            )
        })
        .unwrap();

        // Sanity check we produced the expected fixed-size list shape.
        assert_eq!(layout.row_count(), 4);
        assert_eq!(layout.dtype(), &list_dtype);

        // The elements child is chunked with a split at element index 4, which should map to row 2.
        let reader = layout.new_reader("".into(), segments, &SESSION).unwrap();
        let mut splits = BTreeSet::new();
        reader
            .register_splits(&[], &(0..layout.row_count()), &mut splits)
            .unwrap();

        assert!(splits.contains(&2), "splits = {splits:?}");
        assert!(splits.contains(&layout.row_count()));
    }

    #[test]
    fn register_splits_list_maps_offset_splits_to_rows() {
        let ctx = ArrayContext::empty();

        let segments = Arc::new(TestSegments::default());

        let chunk1_elements = buffer![1i32, 2, 3, 4].into_array();
        let chunk1_offsets = buffer![0u64, 2, 4].into_array();
        let chunk1 = ListArray::try_new(
            chunk1_elements,
            chunk1_offsets,
            vortex_array::validity::Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        let chunk2_elements = buffer![5i32, 6, 7, 8].into_array();
        let chunk2_offsets = buffer![0u64, 2, 4].into_array();
        let chunk2 = ListArray::try_new(
            chunk2_elements,
            chunk2_offsets,
            vortex_array::validity::Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        let list_dtype = chunk1.dtype().clone();

        let offsets_strategy = Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()));
        let elements_strategy = Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()));
        let strategy = ListStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            offsets_strategy,
            elements_strategy,
        );

        let (mut sequence_id, eof) = SequenceId::root().split();
        let layout = block_on(|handle| {
            strategy.write_stream(
                ctx,
                segments.clone(),
                SequentialStreamAdapter::new(
                    vortex_dtype::DType::List(
                        Arc::new(vortex_dtype::DType::Primitive(PType::I32, NonNullable)),
                        NonNullable,
                    ),
                    stream::iter([
                        Ok((sequence_id.advance(), chunk1)),
                        Ok((sequence_id.advance(), chunk2)),
                    ]),
                )
                .sendable(),
                eof,
                handle,
            )
        })
        .unwrap();

        // Sanity check we produced the expected list shape.
        assert_eq!(layout.row_count(), 4);
        assert_eq!(layout.dtype(), &list_dtype);

        // The offsets child is chunked with a split at offsets index 3, which maps to row 2.
        let reader = layout.new_reader("".into(), segments, &SESSION).unwrap();
        let mut splits = BTreeSet::new();
        reader
            .register_splits(&[], &(0..layout.row_count()), &mut splits)
            .unwrap();

        assert!(splits.contains(&2), "splits = {splits:?}");
        assert!(splits.contains(&layout.row_count()));
    }
}
