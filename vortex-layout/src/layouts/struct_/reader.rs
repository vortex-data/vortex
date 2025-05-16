use std::hash::Hash;
use std::sync::{Arc, OnceLock};

use itertools::Itertools;
use parking_lot::RwLock;
use vortex_array::ArrayContext;
use vortex_array::aliases::hash_map::{Entry, HashMap};
use vortex_dtype::{DType, FieldName, StructDType};
use vortex_error::{VortexExpect, VortexResult, vortex_err, vortex_panic};
use vortex_expr::ExprRef;
use vortex_expr::transform::partition::{PartitionedExpr, partition};

use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentSource;
use crate::{Layout, LayoutReader, LayoutVTable};

pub struct StructReader {
    layout: Layout,
    segment_source: Arc<dyn SegmentSource>,
    ctx: ArrayContext,

    field_readers: Vec<OnceLock<Arc<dyn LayoutReader>>>,
    field_lookup: Option<HashMap<FieldName, usize>>,
    partitioned_expr_cache: RwLock<HashMap<ExactExpr, Arc<PartitionedExpr>>>,
}

impl StructReader {
    pub(super) fn try_new(
        layout: Layout,
        segment_source: Arc<dyn SegmentSource>,
        ctx: ArrayContext,
    ) -> VortexResult<Self> {
        if layout.vtable().id() != StructLayout.id() {
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
            segment_source,
            ctx,
            field_readers,
            field_lookup,
            partitioned_expr_cache: Default::default(),
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
            .or_else(|| self.struct_dtype().find(name).ok())
            .ok_or_else(|| vortex_err!("Field {} not found in struct layout", name))?;

        // TODO: think about a Hashmap<FieldName, OnceLock<Arc<dyn LayoutReader>>> for large |fields|.
        self.field_readers[idx].get_or_try_init(|| {
            let child_layout =
                self.layout
                    .child(idx, self.struct_dtype().field_by_index(idx)?, name)?;
            child_layout.reader(&self.segment_source, &self.ctx)
        })
    }

    /// Utility for partitioning an expression over the fields of a struct.
    pub(crate) fn partition_expr(&self, expr: ExprRef) -> Arc<PartitionedExpr> {
        match self
            .partitioned_expr_cache
            .write()
            .entry(ExactExpr(expr.clone()))
        {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => entry
                .insert(Arc::new(partition(expr, self.dtype()).vortex_expect(
                    "We should not fail to partition expression over struct fields",
                )))
                .clone(),
        }
    }
}

impl LayoutReader for StructReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn children(&self) -> VortexResult<Vec<Arc<dyn LayoutReader>>> {
        self.struct_dtype()
            .names()
            .iter()
            .map(|name| self.child(name).cloned())
            .try_collect()
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
