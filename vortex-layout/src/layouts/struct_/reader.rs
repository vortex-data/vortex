use std::hash::Hash;
use std::ops::Range;
use std::sync::{Arc, RwLock};

use vortex_array::aliases::hash_map::{Entry, HashMap};
use vortex_array::ContextRef;
use vortex_dtype::{DType, Field, FieldMask, FieldName, FieldNames};
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexExpect, VortexResult};
use vortex_expr::transform::partition::{partition, PartitionedExpr};
use vortex_expr::ExprRef;

use crate::layouts::struct_::range_reader::StructRangeReader;
use crate::layouts::struct_::StructLayout;
use crate::segments::AsyncSegmentReader;
use crate::{Layout, LayoutRangeReader, LayoutReader, LayoutVTable};

pub struct StructReader {
    layout: Layout,
    /// Field readers corresponding to each field name.
    field_readers: Vec<Arc<dyn LayoutReader>>,
    /// Shared state among all range readers.
    shared_state: Arc<SharedState>,
}

impl StructReader {
    pub(super) fn try_new(
        layout: Layout,
        segments: Arc<dyn AsyncSegmentReader>,
        ctx: ContextRef,
        field_mask: &[FieldMask],
    ) -> VortexResult<Self> {
        if layout.encoding().id() != StructLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        let dtype = layout.dtype();
        let DType::Struct(struct_dt, _) = dtype else {
            vortex_panic!("Mismatched dtype {} for struct layout", dtype);
        };

        let mut field_names = vec![];
        let mut field_readers = vec![];

        // If the field mask contains an `All` fields, then register splits for all fields.
        if field_mask.iter().any(|mask| mask.matches_all()) {
            for (idx, field_dtype) in struct_dt.fields().enumerate() {
                let child_layout = layout.child(idx, field_dtype, layout.row_offset())?;
                let child_reader =
                    child_layout.reader(segments.clone(), ctx.clone(), &[FieldMask::All])?;

                field_names.push(
                    struct_dt
                        .field_name(idx)
                        .vortex_expect("missing field")
                        .clone(),
                );
                field_readers.push(child_reader);
            }
        } else {
            for path in field_mask {
                let Some(field) = path.starting_field()? else {
                    // skip fields not in mask
                    continue;
                };
                let Field::Name(field_name) = field else {
                    vortex_bail!("Expected field name, got {:?}", field);
                };

                let idx = struct_dt.find(field_name)?;
                let child_layout =
                    layout.child(idx, struct_dt.field_by_index(idx)?, layout.row_offset())?;
                let child_reader = child_layout.reader(
                    segments.clone(),
                    ctx.clone(),
                    &[path.clone().step_into()?],
                )?;

                field_names.push(field_name.clone());
                field_readers.push(child_reader);
            }
        }

        let dtype = layout.dtype().clone();

        Ok(Self {
            layout,
            field_readers,
            shared_state: Arc::new(SharedState {
                expr_cache: Default::default(),
                field_names: field_names.into(),
                dtype,
            }),
        })
    }
}

impl LayoutReader for StructReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn range_reader(&self, row_range: Range<u64>) -> Arc<dyn LayoutRangeReader> {
        let fields = self
            .field_readers
            .iter()
            .map(|r| r.range_reader(row_range.clone()))
            .collect();

        Arc::new(StructRangeReader {
            row_range,
            fields,
            shared_state: self.shared_state.clone(),
        }) as _
    }
}

pub(crate) struct SharedState {
    /// Cache of partitioned expressions.
    expr_cache: RwLock<HashMap<ExactExpr, Arc<PartitionedExpr>>>,
    /// Field names of the selected fields.
    field_names: FieldNames,
    /// The dtype of the struct.
    dtype: DType,
}

impl SharedState {
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
                    .insert(Arc::new(partition(expr, &self.dtype)?))
                    .clone(),
            },
        )
    }

    /// Return the child idx for the given field name.
    pub(crate) fn child_idx(&self, name: &FieldName) -> VortexResult<usize> {
        self.field_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| vortex_err!("Field {} not found in struct layout", name))
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
