use std::hash::Hash;
use std::sync::{Arc, OnceLock, RwLock};

use vortex_array::aliases::hash_map::{Entry, HashMap};
use vortex_array::ContextRef;
use vortex_dtype::{DType, FieldName, StructDType};
use vortex_error::{vortex_err, vortex_panic, VortexExpect, VortexResult};
use vortex_expr::transform::partition::{partition, PartitionedExpr};
use vortex_expr::ExprRef;

use crate::layouts::struct_::StructLayout;
use crate::segments::AsyncSegmentReader;
use crate::{Layout, LayoutReader, LayoutReaderExt, LayoutVTable};

#[derive(Clone)]
pub struct StructReader {
    layout: Layout,
    ctx: ContextRef,

    segments: Arc<dyn AsyncSegmentReader>,

    field_readers: Arc<[OnceLock<Arc<dyn LayoutReader>>]>,
    field_lookup: Option<HashMap<FieldName, usize>>,
    expr_cache: Arc<RwLock<HashMap<ExactExpr, Arc<PartitionedExpr>>>>,
}

impl StructReader {
    pub(super) fn try_new(
        layout: Layout,
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

        let field_readers = (0..layout.nchildren()).map(|_| OnceLock::new()).collect();

        // NOTE: This number is arbitrary and likely depends on the longest common prefix of field names
        let field_lookup = (layout.nchildren() > 80).then(|| {
            struct_dt
                .names()
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), i))
                .collect()
        });

        // This is where we need to do some complex things with the scan in order to split it into
        // different scans for different fields.
        Ok(Self {
            layout,
            ctx,
            segments,
            field_readers,
            field_lookup,
            expr_cache: Arc::new(Default::default()),
        })
    }

    /// Return the [`StructDType`] of this layout.
    pub(crate) fn struct_dtype(&self) -> &StructDType {
        self.dtype()
            .as_struct()
            .vortex_expect("Struct layout must have a struct DType, verified at construction")
    }

    /// Return the child reader for the chunk.
    pub(crate) fn child(&self, name: &FieldName) -> VortexResult<&Arc<dyn LayoutReader>> {
        let idx = self
            .field_lookup
            .as_ref()
            .and_then(|lookup| lookup.get(name).copied())
            .or_else(|| self.struct_dtype().find_name(name))
            .ok_or_else(|| vortex_err!("Field {} not found in struct layout", name))?;

        // TODO: think about a Hashmap<FieldName, OnceLock<Arc<dyn LayoutReader>>> for large |fields|.
        self.field_readers[idx].get_or_try_init(|| {
            let child_layout = self
                .layout
                .child(idx, self.struct_dtype().field_dtype(idx)?)?;
            child_layout.reader(self.segments.clone(), self.ctx.clone())
        })
    }

    /// Utility for partitioning an expression over the fields of a struct.
    pub(crate) fn partition_expr(&self, expr: ExprRef) -> VortexResult<Arc<PartitionedExpr>> {
        Ok(
            match self
                .expr_cache
                .write()
                .map_err(|_| vortex_err!("poisoned lock"))?
                .entry(ExactExpr(expr.clone()))
            {
                Entry::Occupied(entry) => entry.get().clone(),
                Entry::Vacant(entry) => entry
                    .insert(Arc::new(partition(expr, self.dtype())?))
                    .clone(),
            },
        )
    }
}

impl LayoutReader for StructReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }
}

/// An expression wrapper that performs pointer equality.
/// NOTE(ngates): we should consider if this shoud live in vortex-expr crate?
#[derive(Clone)]
struct ExactExpr(ExprRef);

impl PartialEq for ExactExpr {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ExactExpr {}

impl Hash for ExactExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state)
    }
}
