use itertools::Itertools;
use vortex_array::iter::ArrayIteratorArrayExt;
use vortex_array::ArrayRef;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexExpect, VortexResult};

use crate::data::Layout;
use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentWriter;
use crate::strategy::LayoutStrategy;
use crate::writer::LayoutWriter;
use crate::LayoutVTableRef;

/// A [`LayoutWriter`] that splits a StructArray batch into child layout writers
pub struct StructLayoutWriter {
    column_strategies: Vec<Box<dyn LayoutWriter>>,
    dtype: DType,
    row_count: u64,
}

impl StructLayoutWriter {
    pub fn new(dtype: DType, column_layout_writers: Vec<Box<dyn LayoutWriter>>) -> Self {
        let struct_dtype = dtype.as_struct().vortex_expect("dtype is not a struct");
        if struct_dtype.fields().len() != column_layout_writers.len() {
            vortex_panic!(
                "number of fields in struct dtype does not match number of column layout writers"
            );
        }
        Self {
            column_strategies: column_layout_writers,
            dtype,
            row_count: 0,
        }
    }

    pub fn try_new_with_factory<F: LayoutStrategy>(
        dtype: &DType,
        factory: F,
    ) -> VortexResult<Self> {
        let struct_dtype = dtype.as_struct().vortex_expect("dtype is not a struct");
        Ok(Self::new(
            dtype.clone(),
            struct_dtype
                .fields()
                .map(|dtype| factory.new_writer(&dtype))
                .try_collect()?,
        ))
    }
}

impl LayoutWriter for StructLayoutWriter {
    fn push_chunk(
        &mut self,
        segments: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        let struct_array = chunk
            .as_struct_typed()
            .ok_or_else(|| vortex_err!("batch is not a struct array"))?;

        if struct_array.nfields() != self.column_strategies.len() {
            vortex_bail!(
                "number of fields in struct array does not match number of column layout writers"
            );
        }
        self.row_count += struct_array.len() as u64;

        for i in 0..struct_array.nfields() {
            // TODO(joe): handle struct validity
            let column = chunk
                .as_struct_typed()
                .vortex_expect("batch is a struct array")
                .maybe_null_field_by_idx(i)
                .vortex_expect("bounds already checked");

            for column_chunk in column.to_array_iterator() {
                self.column_strategies[i].push_chunk(segments, column_chunk?)?;
            }
        }

        Ok(())
    }

    fn finish(&mut self, segments: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        let mut column_layouts = vec![];
        for writer in self.column_strategies.iter_mut() {
            column_layouts.push(writer.finish(segments)?);
        }
        Ok(Layout::new_owned(
            "struct".into(),
            LayoutVTableRef::from_static(&StructLayout),
            self.dtype.clone(),
            self.row_count,
            vec![],
            column_layouts,
            None,
        ))
    }
}
