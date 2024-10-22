use std::sync::Arc;

use bytes::Bytes;
use flatbuffers::{ForwardsUOffset, Vector};
use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_flatbuffers::footer as fb;
use vortex_schema::projection::Projection;

use crate::layouts::read::batch::BatchReader;
use crate::layouts::read::cache::{LazyDeserializedDType, RelativeLayoutCache};
use crate::layouts::read::context::{LayoutDeserializer, LayoutId, LayoutSpec};
use crate::layouts::read::{LayoutReader, ReadResult, Scan};
use crate::layouts::COLUMN_LAYOUT_ID;

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
        scan: Scan,
        layout_builder: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> Box<dyn LayoutReader> {
        Box::new(ColumnLayout::new(
            fb_bytes,
            fb_loc,
            scan,
            layout_builder,
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
    scan: Scan,
    layout_builder: LayoutDeserializer,
    message_cache: RelativeLayoutCache,
    reader: Option<BatchReader>,
}

impl ColumnLayout {
    pub fn new(
        fb_bytes: Bytes,
        fb_loc: usize,
        scan: Scan,
        layout_builder: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> Self {
        Self {
            fb_bytes,
            fb_loc,
            scan,
            layout_builder,
            message_cache,
            reader: None,
        }
    }

    pub fn flatbuffer(&self) -> fb::Layout {
        unsafe {
            let tab = flatbuffers::Table::new(&self.fb_bytes, self.fb_loc);
            fb::Layout::init_from_table(tab)
        }
    }

    fn read_child(
        &self,
        idx: usize,
        children: Vector<ForwardsUOffset<fb::Layout>>,
        dtype: DType,
    ) -> VortexResult<Box<dyn LayoutReader>> {
        let layout = children.get(idx);

        // TODO: Figure out complex nested schema projections
        let mut child_scan = self.scan.clone();
        child_scan.projection = Projection::All;

        self.layout_builder.read_layout(
            self.fb_bytes.clone(),
            layout._tab.loc(),
            child_scan,
            self.message_cache.relative(
                idx as u16,
                Arc::new(LazyDeserializedDType::from_dtype(dtype)),
            ),
        )
    }

    pub fn lazy_dtype(&self) -> VortexResult<Arc<LazyDeserializedDType>> {
        match &self.scan.projection {
            Projection::All => Ok(self.message_cache.dtype().clone()),
            Projection::Flat(p) => self.message_cache.dtype().project(p),
        }
    }
}

impl LayoutReader for ColumnLayout {
    fn read_next(&mut self) -> VortexResult<Option<ReadResult>> {
        if let Some(br) = &mut self.reader {
            br.read()
        } else {
            let result_lazy_dtype = self.lazy_dtype()?;
            let DType::Struct(s, _) = result_lazy_dtype.value()? else {
                vortex_bail!("DType was not a struct")
            };

            let fb_children = self
                .flatbuffer()
                .children()
                .ok_or_else(|| vortex_err!("Missing children"))?;

            let child_layouts = match &self.scan.projection {
                Projection::All => (0..fb_children.len())
                    .zip_eq(s.dtypes().iter())
                    .map(|(index, dtype)| self.read_child(index, fb_children, dtype.clone()))
                    .collect::<VortexResult<Vec<_>>>()?,
                Projection::Flat(proj) => proj
                    .iter()
                    .map(|f| result_lazy_dtype.resolve_field(f))
                    .zip(s.dtypes().iter().cloned())
                    .map(|(child_idx, dtype)| self.read_child(child_idx?, fb_children, dtype))
                    .collect::<VortexResult<Vec<_>>>()?,
            };

            self.reader = Some(BatchReader::new(s.names().clone(), child_layouts));
            self.read_next()
        }
    }
}
