mod eval_expr;
mod reader;
pub mod writer;

use std::collections::BTreeSet;
use std::sync::Arc;

use reader::StructReader;
use vortex_array::ArrayContext;
use vortex_dtype::{DType, Field, FieldMask};
use vortex_error::{VortexResult, vortex_bail};

use crate::data::Layout;
use crate::reader::{LayoutReader, LayoutReaderExt};
use crate::segments::AsyncSegmentReader;
use crate::vtable::LayoutVTable;
use crate::{LayoutId, STRUCT_LAYOUT_ID};

#[derive(Debug)]
pub struct StructLayout;

impl LayoutVTable for StructLayout {
    fn id(&self) -> LayoutId {
        STRUCT_LAYOUT_ID
    }

    fn reader(
        &self,
        layout: Layout,
        ctx: ArrayContext,
        segment_reader: Arc<dyn AsyncSegmentReader>,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(StructReader::try_new(layout, ctx, segment_reader)?.into_arc())
    }

    fn register_splits(
        &self,
        layout: &Layout,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        let DType::Struct(dtype, _) = layout.dtype() else {
            vortex_bail!("Mismatched dtype {} for struct layout", layout.dtype());
        };

        // If the field mask contains an `All` fields, then register splits for all fields.
        if field_mask.iter().any(|mask| mask.matches_all()) {
            for (idx, field_dtype) in dtype.fields().enumerate() {
                let child = layout.child(idx, field_dtype, dtype.field_name(idx)?)?;
                child.register_splits(&[FieldMask::All], row_offset, splits)?;
            }
            return Ok(());
        }

        // Register the splits for each field in the mask
        for path in field_mask {
            let Some(field) = path.starting_field()? else {
                // skip fields not in mask
                continue;
            };
            let Field::Name(field_name) = field else {
                vortex_bail!("Expected field name, got {:?}", field);
            };

            let idx = dtype.find(field_name)?;
            let child = layout.child(idx, dtype.field_by_index(idx)?, field_name)?;
            child.register_splits(&[path.clone().step_into()?], row_offset, splits)?;
        }

        Ok(())
    }
}
