use std::sync::Arc;

use bytes::Bytes;
use flatbuffers::{ForwardsUOffset, Vector};
use itertools::Itertools;
use vortex_dtype::field::Field;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_expr::{Column, Select};
use vortex_flatbuffers::footer;

use crate::layouts::read::batch::ColumnBatchReader;
use crate::layouts::read::cache::{LazyDeserializedDType, RelativeLayoutCache};
use crate::layouts::read::filter_project::filter_project;
use crate::layouts::read::selection::RowSelector;
use crate::layouts::{
    LayoutDeserializer, LayoutId, LayoutReader, LayoutSpec, Message, RangeResult, ReadResult,
    RowFilter, Scan, COLUMN_LAYOUT_ID,
};

#[derive(Debug)]
pub struct ColumnLayoutSpec;

impl LayoutSpec for ColumnLayoutSpec {
    fn id(&self) -> LayoutId {
        COLUMN_LAYOUT_ID
    }

    fn layout(
        &self,
        fb_bytes: Bytes,
        fb_loc: usize,
        length: u64,
        scan: Scan,
        layout_serde: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> Box<dyn LayoutReader> {
        Box::new(ColumnLayout::new(
            fb_bytes,
            fb_loc,
            length,
            scan,
            layout_serde,
            message_cache,
        ))
    }
}

/// In memory representation of Columnar NestedLayout.
///
/// Each child represents a column
#[derive(Debug)]
pub struct ColumnLayout {
    fb_bytes: Bytes,
    fb_loc: usize,
    length: u64,
    offset: usize,
    scan: Scan,
    layout_serde: LayoutDeserializer,
    message_cache: RelativeLayoutCache,
    reader: Option<ColumnBatchReader>,
}

impl ColumnLayout {
    pub fn new(
        fb_bytes: Bytes,
        fb_loc: usize,
        length: u64,
        scan: Scan,
        layout_serde: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> Self {
        Self {
            fb_bytes,
            fb_loc,
            scan,
            length,
            layout_serde,
            message_cache,
            reader: None,
            offset: 0,
        }
    }

    pub fn flatbuffer(&self) -> footer::Layout {
        unsafe {
            let tab = flatbuffers::Table::new(&self.fb_bytes, self.fb_loc);
            footer::Layout::init_from_table(tab)
        }
    }

    fn read_child(
        &self,
        idx: usize,
        children: Vector<ForwardsUOffset<footer::Layout>>,
        dtype: DType,
    ) -> VortexResult<Box<dyn LayoutReader>> {
        let mut layout = self.layout_serde.read_layout(
            self.fb_bytes.clone(),
            children.get(idx)._tab.loc(),
            self.length,
            // TODO(robert): Changes this once we support nested projections
            Scan::new(None),
            self.message_cache.relative(
                idx as u16,
                Arc::new(LazyDeserializedDType::from_dtype(dtype)),
            ),
        )?;
        if self.offset != 0 {
            layout.advance(self.offset)?;
        }
        Ok(layout)
    }

    fn filter_reader(&mut self) -> VortexResult<ColumnBatchReader> {
        let Some(ref rf) = self.scan.expr else {
            vortex_bail!("Must have scan expression");
        };

        let filter_refs = self
            .scan_fields()?
            .vortex_expect("Can't be an empty filter");
        let lazy_dtype = self.message_cache.dtype().project(&filter_refs)?;

        let fb_children = self
            .flatbuffer()
            .children()
            .ok_or_else(|| vortex_err!("Missing children"))?;
        let filter_dtype = lazy_dtype.value()?;
        let DType::Struct(s, ..) = filter_dtype else {
            vortex_bail!("Column layout must have struct dtype")
        };

        let mut unhandled_children_names = Vec::new();
        let mut unhandled_children = Vec::new();
        let mut handled_children = Vec::new();
        let mut handled_names = Vec::new();

        for (idx, field) in filter_refs.into_iter().enumerate() {
            let resolved_child = lazy_dtype.resolve_field(&field)?;
            let child_loc = fb_children.get(resolved_child)._tab.loc();
            let filter = filter_project(rf, &[field]);

            let has_filter = filter.is_some();

            let mut child = self.layout_serde.read_layout(
                self.fb_bytes.clone(),
                child_loc,
                self.length,
                Scan::new(filter),
                self.message_cache.relative(
                    resolved_child as u16,
                    Arc::new(LazyDeserializedDType::from_dtype(s.dtypes()[idx].clone())),
                ),
            )?;

            if self.offset != 0 {
                child.advance(self.offset)?;
            }

            if has_filter {
                handled_children.push(child);
                handled_names.push(s.names()[idx].clone());
            } else {
                unhandled_children.push(child);
                unhandled_children_names.push(s.names()[idx].clone());
            }
        }

        if !unhandled_children_names.is_empty() {
            let Some(prf) = filter_project(
                rf,
                &unhandled_children_names
                    .iter()
                    .map(|n| Field::from(n.as_ref()))
                    .collect::<Vec<_>>(),
            ) else {
                vortex_bail!("Must be able to project filter into unhandled space")
            };

            handled_children.push(Box::new(ColumnBatchReader::new(
                unhandled_children_names.into(),
                unhandled_children,
                Some(prf),
                true,
            )));
            handled_names.push("unhandled".into());
        }

        let filter = Some(Arc::new(RowFilter::from_conjunction(
            handled_names
                .iter()
                .map(|f| Arc::new(Column::new(Field::from(&**f))) as _)
                .collect(),
        )) as _);
        Ok(ColumnBatchReader::new(
            handled_names.into(),
            handled_children,
            filter,
            true,
        ))
    }

    fn read_children(&mut self) -> VortexResult<ColumnBatchReader> {
        let lazy_dtype = self
            .scan_fields()?
            .map(|e| self.message_cache.dtype().project(&e))
            .unwrap_or_else(|| Ok(self.message_cache.dtype().clone()))?;
        let DType::Struct(s, _) = lazy_dtype.value()? else {
            vortex_bail!("DType was not a struct")
        };

        let fb_children = self
            .flatbuffer()
            .children()
            .ok_or_else(|| vortex_err!("Missing children"))?;

        let expr_fields = self.scan_fields()?;

        let child_layouts = match expr_fields {
            None => (0..fb_children.len())
                .zip_eq(s.dtypes().iter())
                .map(|(index, dtype)| self.read_child(index, fb_children, dtype.clone()))
                .collect::<VortexResult<Vec<_>>>()?,
            Some(e) => e
                .into_iter()
                .map(|f| lazy_dtype.resolve_field(&f))
                .zip(s.dtypes().iter().cloned())
                .map(|(child_idx, dtype)| self.read_child(child_idx?, fb_children, dtype))
                .collect::<VortexResult<Vec<_>>>()?,
        };

        Ok(ColumnBatchReader::new(
            s.names().clone(),
            child_layouts,
            None,
            false,
        ))
    }

    fn scan_fields(&self) -> VortexResult<Option<Vec<Field>>> {
        self.scan
            .expr
            .as_ref()
            .map(|e| {
                if let Some(se) = e.as_any().downcast_ref::<Select>() {
                    match se {
                        Select::Include(i) => Ok(i.clone()),
                        Select::Exclude(_) => vortex_bail!("Select::Exclude not supported"),
                    }
                } else {
                    Ok(e.references().into_iter().cloned().collect::<Vec<_>>())
                }
            })
            .transpose()
    }

    fn read_init(&mut self) -> VortexResult<()> {
        if let Some(expr) = self.scan.expr.as_ref() {
            if expr.as_any().is::<RowFilter>() {
                self.reader = Some(self.filter_reader()?);
            } else {
                self.reader = Some(self.read_children()?);
            }
        } else {
            self.reader = Some(self.read_children()?);
        }
        Ok(())
    }
}

impl LayoutReader for ColumnLayout {
    fn next_range(&mut self) -> VortexResult<RangeResult> {
        if let Some(r) = &mut self.reader {
            r.next_range()
        } else {
            self.read_init()?;
            self.next_range()
        }
    }

    fn read_next(&mut self, selector: RowSelector) -> VortexResult<Option<ReadResult>> {
        if let Some(r) = &mut self.reader {
            r.read_next(selector)
        } else {
            self.read_init()?;
            self.read_next(selector)
        }
    }

    fn advance(&mut self, up_to_row: usize) -> VortexResult<Vec<Message>> {
        if let Some(r) = &mut self.reader {
            r.advance(up_to_row)
        } else {
            self.offset = up_to_row;
            Ok(Vec::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::iter;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, RwLock};

    use bytes::Bytes;
    use vortex::accessor::ArrayAccessor;
    use vortex::array::{ChunkedArray, PrimitiveArray, StructArray, VarBinArray};
    use vortex::validity::Validity;
    use vortex::{ArrayDType, IntoArray, IntoArrayVariant};
    use vortex_dtype::field::Field;
    use vortex_dtype::{DType, Nullability};
    use vortex_expr::{BinaryExpr, Column, Literal, Operator};
    use vortex_schema::projection::Projection;

    use crate::layouts::read::cache::{LazyDeserializedDType, RelativeLayoutCache};
    use crate::layouts::read::layouts::test_read::{
        filter_read_layout, read_filters, read_layout, read_layout_data, read_layout_ranges,
    };
    use crate::layouts::{
        LayoutDescriptorReader, LayoutDeserializer, LayoutMessageCache, LayoutReader, LayoutWriter,
        RowFilter, Scan,
    };

    async fn layout_and_bytes(
        cache: Arc<RwLock<LayoutMessageCache>>,
        scan: Scan,
    ) -> (Box<dyn LayoutReader>, Box<dyn LayoutReader>, Bytes) {
        let int_array = PrimitiveArray::from((0..100).collect::<Vec<_>>()).into_array();
        let int_dtype = int_array.dtype().clone();
        let chunked = ChunkedArray::try_new(iter::repeat(int_array).take(5).collect(), int_dtype)
            .unwrap()
            .into_array();
        let str_array = VarBinArray::from_vec(
            iter::repeat("test text").take(500).collect(),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let len = chunked.len();
        let struct_arr = StructArray::try_new(
            vec!["ints".into(), "strs".into()].into(),
            vec![chunked, str_array],
            len,
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        let mut writer = LayoutWriter::new(Vec::new());
        writer = writer.write_array_columns(struct_arr).await.unwrap();
        let written = writer.finalize().await.unwrap();

        let footer = LayoutDescriptorReader::new(LayoutDeserializer::default())
            .read_footer(&written, written.len() as u64)
            .await
            .unwrap();

        let dtype = Arc::new(LazyDeserializedDType::from_bytes(
            footer.dtype_bytes().unwrap(),
            Projection::All,
        ));
        let len = len as u64;
        (
            footer
                .layout(
                    len,
                    scan,
                    RelativeLayoutCache::new(cache.clone(), dtype.clone()),
                )
                .unwrap(),
            footer
                .layout(len, Scan::new(None), RelativeLayoutCache::new(cache, dtype))
                .unwrap(),
            Bytes::from(written),
        )
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_range() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (mut filter_layout, mut project_layout, buf) = layout_and_bytes(
            cache.clone(),
            Scan::new(Some(Arc::new(RowFilter::new(Arc::new(BinaryExpr::new(
                Arc::new(Column::new(Field::from("ints"))),
                Operator::Gt,
                Arc::new(Literal::new(10.into())),
            )))))),
        )
        .await;
        let arr = filter_read_layout(filter_layout.as_mut(), project_layout.as_mut(), cache, &buf)
            .pop_front();

        assert!(arr.is_some());
        let prim_arr = arr
            .as_ref()
            .unwrap()
            .with_dyn(|a| a.as_struct_array_unchecked().field(0))
            .unwrap()
            .into_primitive()
            .unwrap();
        let str_arr = arr
            .as_ref()
            .unwrap()
            .with_dyn(|a| a.as_struct_array_unchecked().field(1))
            .unwrap()
            .into_varbinview()
            .unwrap();
        assert_eq!(
            prim_arr.maybe_null_slice::<i32>(),
            &(11..100).collect::<Vec<_>>()
        );
        assert_eq!(
            str_arr
                .with_iterator(|iter| iter
                    .flatten()
                    .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                    .collect::<Vec<_>>())
                .unwrap(),
            iter::repeat("test text").take(89).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_range_no_filter() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (_, mut project_layout, buf) = layout_and_bytes(cache.clone(), Scan::new(None)).await;
        let arr = read_layout(project_layout.as_mut(), cache, &buf).pop_front();

        assert!(arr.is_some());
        let prim_arr = arr
            .as_ref()
            .unwrap()
            .with_dyn(|a| a.as_struct_array_unchecked().field(0))
            .unwrap()
            .into_primitive()
            .unwrap();
        let str_arr = arr
            .as_ref()
            .unwrap()
            .with_dyn(|a| a.as_struct_array_unchecked().field(1))
            .unwrap()
            .into_varbinview()
            .unwrap();
        assert_eq!(
            prim_arr.maybe_null_slice::<i32>(),
            (0..100).collect::<Vec<_>>()
        );
        assert_eq!(
            str_arr
                .with_iterator(|iter| iter
                    .flatten()
                    .map(|s| unsafe { String::from_utf8_unchecked(s.to_vec()) })
                    .collect::<Vec<_>>())
                .unwrap(),
            iter::repeat("test text").take(100).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn advance_read_range() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (mut filter_layout, mut project_layout, buf) = layout_and_bytes(
            cache.clone(),
            Scan::new(Some(Arc::new(RowFilter::new(Arc::new(BinaryExpr::new(
                Arc::new(Column::new(Field::from("ints"))),
                Operator::Gt,
                Arc::new(Literal::new(10.into())),
            )))))),
        )
        .await;
        filter_layout.advance(50).unwrap();
        let arr = filter_read_layout(filter_layout.as_mut(), project_layout.as_mut(), cache, &buf)
            .pop_front();

        assert!(arr.is_some());
        let arr = arr
            .unwrap()
            .with_dyn(|a| a.as_struct_array_unchecked().field(0))
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(
            arr.into_primitive().unwrap().maybe_null_slice::<i32>(),
            &(50..100).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn advance_skipped() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (mut filter_layout, mut project_layout, buf) = layout_and_bytes(
            cache.clone(),
            Scan::new(Some(Arc::new(RowFilter::new(Arc::new(BinaryExpr::new(
                Arc::new(Column::new(Field::from("ints"))),
                Operator::Gt,
                Arc::new(Literal::new(10.into())),
            )))))),
        )
        .await;
        filter_layout.advance(500).unwrap();
        let arr = filter_read_layout(filter_layout.as_mut(), project_layout.as_mut(), cache, &buf);

        assert!(arr.is_empty());
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn advance_after_filter() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (mut filter_layout, mut project_layout, buf) = layout_and_bytes(
            cache.clone(),
            Scan::new(Some(Arc::new(RowFilter::new(Arc::new(BinaryExpr::new(
                Arc::new(Column::new(Field::from("ints"))),
                Operator::Gt,
                Arc::new(Literal::new(10.into())),
            )))))),
        )
        .await;
        let selectors = read_layout_ranges(filter_layout.as_mut(), cache.clone(), &buf)
            .into_iter()
            .flat_map(|s| read_filters(filter_layout.as_mut(), cache.clone(), &buf, s))
            .collect::<Vec<_>>();

        project_layout.advance(50).unwrap();
        let arr = selectors
            .into_iter()
            .flat_map(|s| read_layout_data(project_layout.as_mut(), cache.clone(), &buf, s))
            .collect::<Vec<_>>();

        assert_eq!(
            arr.first()
                .unwrap()
                .with_dyn(|a| a.as_struct_array_unchecked().field(0))
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(50..100).collect::<Vec<_>>()
        );
        assert_eq!(
            arr[4]
                .with_dyn(|a| a.as_struct_array_unchecked().field(0))
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(11..100).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn advance_mid_read() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (mut filter_layout, mut project_layout, buf) = layout_and_bytes(
            cache.clone(),
            Scan::new(Some(Arc::new(RowFilter::new(Arc::new(BinaryExpr::new(
                Arc::new(Column::new(Field::from("ints"))),
                Operator::Gt,
                Arc::new(Literal::new(10.into())),
            )))))),
        )
        .await;
        let selectors = read_layout_ranges(filter_layout.as_mut(), cache.clone(), &buf)
            .into_iter()
            .flat_map(|s| read_filters(filter_layout.as_mut(), cache.clone(), &buf, s))
            .collect::<Vec<_>>();
        let advanced = AtomicBool::new(false);
        let mut arr = selectors
            .into_iter()
            .flat_map(|s| {
                let a = read_layout_data(project_layout.as_mut(), cache.clone(), &buf, s);
                if advanced
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    project_layout.advance(321).unwrap();
                }
                a
            })
            .collect::<VecDeque<_>>();

        assert_eq!(arr.len(), 3);
        assert_eq!(
            arr.pop_front()
                .unwrap()
                .with_dyn(|a| a.as_struct_array_unchecked().field(0))
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(11..100).collect::<Vec<_>>()
        );
        assert_eq!(
            arr.pop_front()
                .unwrap()
                .with_dyn(|a| a.as_struct_array_unchecked().field(0))
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(21..100).collect::<Vec<_>>()
        );
        assert_eq!(
            arr.pop_front()
                .unwrap()
                .with_dyn(|a| a.as_struct_array_unchecked().field(0))
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(11..100).collect::<Vec<_>>()
        );
    }
}
