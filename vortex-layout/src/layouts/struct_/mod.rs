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
use crate::segments::SegmentSource;
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
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(StructReader::try_new(layout, segment_source.clone(), ctx.clone())?.into_arc())
    }

    fn register_splits(
        &self,
        layout: &Layout,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        for_all_matching_children(layout, field_mask, |mask, child| {
            child.register_splits(&[mask], row_offset, splits)
        })?;
        Ok(())
    }
}

fn for_all_matching_children<F>(
    layout: &Layout,
    field_mask: &[FieldMask],
    mut per_child: F,
) -> VortexResult<()>
where
    F: FnMut(FieldMask, Layout) -> VortexResult<()>,
{
    let DType::Struct(dtype, _) = layout.dtype() else {
        vortex_bail!("Mismatched dtype {} for struct layout", layout.dtype());
    };

    // If the field mask contains an `All` fields, then enumerate all fields.
    if field_mask.iter().any(|mask| mask.matches_all()) {
        for (idx, field_dtype) in dtype.fields().enumerate() {
            let child = layout.child(idx, field_dtype, dtype.field_name(idx)?)?;
            per_child(FieldMask::All, child)?;
        }
        return Ok(());
    }

    // Enumerate each field in the mask
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
        per_child(path.clone().step_into()?, child)?;
    }

    Ok(())
}
