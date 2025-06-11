use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_expr::{ExprRef, ScopeDType};

use crate::{ArrayEvaluation, LayoutReaderRef, MaskEvaluation, PruningEvaluation};

pub type VirtualLayoutReaderRef = Arc<dyn VirtualLayoutReader>;

/// A [`crate::VirtualLayoutReader`] is a used to adapt (and read) a [`crate::Layout`].
/// virtual layouts add virtual columns or other reader functionality on top of a layout
pub trait VirtualLayoutReader: 'static + Send + Sync {
    /// Returns the name of the layout reader for debugging.
    fn name(&self) -> &Arc<str>;

    /// Returns the un-projected dtype of the layout reader.
    fn scope_dtype(&self) -> &ScopeDType;

    /// Performs an approximate evaluation of the expression against the layout reader.
    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>>;

    /// Performs an exact evaluation of the expression against the layout reader.
    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>>;

    /// Evaluates the expression against the layout.
    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>>;
}

pub struct PhysicalToVirtualLayoutAdaptor {
    child: LayoutReaderRef,
    scope_dtype: ScopeDType,
}

impl PhysicalToVirtualLayoutAdaptor {
    pub fn new(child: LayoutReaderRef) -> Self {
        let dtype = child.dtype().clone();
        Self {
            child,
            scope_dtype: ScopeDType::new(dtype),
        }
    }
}

impl VirtualLayoutReader for PhysicalToVirtualLayoutAdaptor {
    fn name(&self) -> &Arc<str> {
        self.child.name()
    }

    fn scope_dtype(&self) -> &ScopeDType {
        &self.scope_dtype
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        self.child.pruning_evaluation(row_range, expr)
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        self.child.filter_evaluation(row_range, expr)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        self.child.projection_evaluation(row_range, expr)
    }
}
