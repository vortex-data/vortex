use itertools::Itertools;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::iter::ArrayIteratorArrayExt;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};

use crate::LayoutVTableRef;
use crate::data::Layout;
use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentWriter;
use crate::strategy::LayoutStrategy;
use crate::writer::LayoutWriter;

/// A [`LayoutWriter`] that splits a StructArray batch into child layout writers
pub struct StructLayoutWriter {
    column_strategies: Vec<Box<dyn LayoutWriter>>,
    dtype: DType,
    row_count: u64,
}

impl StructLayoutWriter {
    pub fn try_new(
        dtype: DType,
        column_layout_writers: Vec<Box<dyn LayoutWriter>>,
    ) -> VortexResult<Self> {
        let struct_dtype = dtype
            .as_struct()
            .ok_or_else(|| vortex_err!("expected StructDType"))?;
        if HashSet::from_iter(struct_dtype.names().iter()).len() != struct_dtype.names().len() {
            vortex_bail!("StructLayout must have unique field names")
        }
        if struct_dtype.fields().len() != column_layout_writers.len() {
            vortex_bail!(
                "number of fields in struct dtype does not match number of column layout writers"
            );
        }
        Ok(Self {
            column_strategies: column_layout_writers,
            dtype,
            row_count: 0,
        })
    }

    pub fn try_new_with_factory<F: LayoutStrategy>(
        ctx: &ArrayContext,
        dtype: &DType,
        factory: F,
    ) -> VortexResult<Self> {
        let struct_dtype = dtype
            .as_struct()
            .ok_or_else(|| vortex_err!("expected StructDType"))?;
        Self::try_new(
            dtype.clone(),
            struct_dtype
                .fields()
                .map(|dtype| factory.new_writer(ctx, &dtype))
                .try_collect()?,
        )
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
            LayoutVTableRef::new_ref(&StructLayout),
            self.dtype.clone(),
            self.row_count,
            vec![],
            column_layouts,
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::LayoutWriterExt;
    use crate::layouts::flat::writer::{FlatLayoutOptions, FlatLayoutWriter};
    use crate::layouts::struct_::writer::StructLayoutWriter;

    #[test]
    #[should_panic]
    fn fails_on_duplicate_field() {
        StructLayoutWriter::try_new(
            DType::Struct(
                Arc::new(
                    [
                        ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                        ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                    ]
                    .into_iter()
                    .collect(),
                ),
                Nullability::NonNullable,
            ),
            vec![
                FlatLayoutWriter::new(
                    ArrayContext::empty(),
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    FlatLayoutOptions::default(),
                )
                .boxed(),
                FlatLayoutWriter::new(
                    ArrayContext::empty(),
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    FlatLayoutOptions::default(),
                )
                .boxed(),
            ],
        )
        .unwrap();
    }
}
