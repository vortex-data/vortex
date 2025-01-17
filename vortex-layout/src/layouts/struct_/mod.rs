mod eval_expr;
mod eval_stats;
mod reader;
pub mod writer;

use std::collections::BTreeSet;
use std::sync::Arc;

use reader::StructReader;
use vortex_array::ContextRef;
use vortex_dtype::{DType, FieldPath};
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::data::LayoutData;
use crate::encoding::{LayoutEncoding, LayoutId};
use crate::reader::{LayoutReader, LayoutReaderExt};
use crate::segments::AsyncSegmentReader;
use crate::COLUMNAR_LAYOUT_ID;

#[derive(Debug)]
pub struct StructLayout;

impl LayoutEncoding for StructLayout {
    fn id(&self) -> LayoutId {
        COLUMNAR_LAYOUT_ID
    }

    fn reader(
        &self,
        layout: LayoutData,
        ctx: ContextRef,
        segments: Arc<dyn AsyncSegmentReader>,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        Ok(StructReader::try_new(layout, segments, ctx)?.into_arc())
    }

    fn register_splits(
        &self,
        layout: &LayoutData,
        field_mask: &[FieldPath],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        let DType::Struct(dtype, _) = layout.dtype() else {
            vortex_bail!("Mismatched dtype {} for struct layout", layout.dtype());
        };

        // Register the splits for each field in the mask
        for path in field_mask {
            if path.is_root() {
                continue;
            }
            let idx = dtype
                .find(&path.path()[0])
                .ok_or_else(|| vortex_err!("Field not found: {}", path))?;

            let child = layout.child(idx, dtype.field_dtype(idx)?)?;
            child.register_splits(&[path.clone().step_into()?], row_offset, splits)?;
        }

        Ok(())
    }
}
