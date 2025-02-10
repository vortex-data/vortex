// mod eval_expr;
mod range_reader;
mod reader;
pub mod writer;

use std::collections::BTreeSet;
use std::sync::Arc;

use reader::StructReader;
use vortex_array::ContextRef;
use vortex_dtype::{DType, Field, FieldMask};
use vortex_error::{vortex_bail, VortexResult};

use crate::data::Layout;
use crate::reader::{LayoutReader, LayoutReaderExt};
use crate::segments::AsyncSegmentReader;
use crate::vtable::LayoutVTable;
use crate::{LayoutId, COLUMNAR_LAYOUT_ID};

#[derive(Debug)]
pub struct StructLayout;

impl LayoutVTable for StructLayout {
    fn id(&self) -> LayoutId {
        COLUMNAR_LAYOUT_ID
    }

    fn reader(
        &self,
        layout: Layout,
        ctx: ContextRef,
        segments: Arc<dyn AsyncSegmentReader>,
        field_mask: &[FieldMask],
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(StructReader::try_new(layout, segments, ctx, field_mask)?.into_arc())
    }

    fn register_splits(
        &self,
        layout: &Layout,
        field_mask: &[FieldMask],
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        let DType::Struct(dtype, _) = layout.dtype() else {
            vortex_bail!("Mismatched dtype {} for struct layout", layout.dtype());
        };

        // If the field mask contains an `All` fields, then register splits for all fields.
        if field_mask.iter().any(|mask| mask.matches_all()) {
            for (idx, field_dtype) in dtype.fields().enumerate() {
                let child = layout.child(idx, field_dtype, layout.row_offset())?;
                child.register_splits(&[FieldMask::All], splits)?;
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
            let child = layout.child(idx, dtype.field_by_index(idx)?, layout.row_offset())?;
            child.register_splits(&[path.clone().step_into()?], splits)?;
        }

        Ok(())
    }
}
