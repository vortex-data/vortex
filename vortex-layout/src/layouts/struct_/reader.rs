use std::sync::{Arc, OnceLock};

use vortex_array::aliases::hash_map::HashMap;
use vortex_array::ContextRef;
use vortex_dtype::{DType, Field, FieldName, StructDType};
use vortex_error::{vortex_err, vortex_panic, VortexExpect, VortexResult};

use crate::layouts::struct_::StructLayout;
use crate::segments::AsyncSegmentReader;
use crate::{LayoutData, LayoutEncoding, LayoutReader, LayoutReaderExt};

#[derive(Clone)]
pub struct StructReader {
    layout: LayoutData,
    ctx: ContextRef,

    segments: Arc<dyn AsyncSegmentReader>,

    field_readers: Arc<[OnceLock<Arc<dyn LayoutReader>>]>,
    field_lookup: HashMap<FieldName, usize>,
}

impl StructReader {
    pub(super) fn try_new(
        layout: LayoutData,
        segments: Arc<dyn AsyncSegmentReader>,
        ctx: ContextRef,
    ) -> VortexResult<Self> {
        if layout.encoding().id() != StructLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        let dtype = layout.dtype();
        let DType::Struct(struct_dt, _) = dtype else {
            vortex_panic!("Mismatched dtype {} for struct layout", dtype);
        };

        let field_readers = struct_dt.names().iter().map(|_| OnceLock::new()).collect();

        let field_lookup = struct_dt
            .names()
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // This is where we need to do some complex things with the scan in order to split it into
        // different scans for different fields.
        Ok(Self {
            layout,
            ctx,
            segments,
            field_readers,
            field_lookup,
        })
    }

    /// Return the [`StructDType`] of this layout.
    pub(crate) fn struct_dtype(&self) -> &StructDType {
        self.dtype()
            .as_struct()
            .vortex_expect("Struct layout must have a struct DType, verified at construction")
    }

    /// Return the child reader for the chunk.
    pub(crate) fn child(&self, field: &Field) -> VortexResult<&Arc<dyn LayoutReader>> {
        let idx = match field {
            Field::Name(n) => *self
                .field_lookup
                .get(n)
                .ok_or_else(|| vortex_err!("Field {} not found in struct layout", n))?,
            Field::Index(idx) => *idx,
        };
        self.field_readers[idx].get_or_try_init(|| {
            let child_layout = self
                .layout
                .child(idx, self.struct_dtype().field_dtype(idx)?)?;
            child_layout.reader(self.segments.clone(), self.ctx.clone())
        })
    }
}

impl LayoutReader for StructReader {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }
}
