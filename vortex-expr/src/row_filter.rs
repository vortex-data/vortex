use std::any::Any;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use itertools::Itertools;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::ConstantArray;
use vortex_array::compute::{and_kleene, fill_null};
use vortex_array::stats::ArrayStatistics;
use vortex_array::{ArrayData, IntoArrayData};
use vortex_dtype::Field;
use vortex_error::{VortexExpect, VortexResult};

use crate::{expr_project, split_conjunction, ExprRef, VortexExpr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowFilter {
    pub(crate) conjunction: Vec<ExprRef>,
}

impl RowFilter {
    pub fn new(expr: ExprRef) -> Self {
        let conjunction = split_conjunction(&expr);
        Self { conjunction }
    }

    pub fn new_expr(expr: ExprRef) -> ExprRef {
        Arc::new(Self::new(expr))
    }

    /// Create a new row filter from a conjunction. The conjunction **must** have length > 0.
    pub fn from_conjunction(conjunction: Vec<ExprRef>) -> Self {
        assert!(!conjunction.is_empty());
        Self { conjunction }
    }

    /// Create a new row filter from a conjunction. The conjunction **must** have length > 0.
    pub fn from_conjunction_expr(conjunction: Vec<ExprRef>) -> ExprRef {
        Arc::new(Self::from_conjunction(conjunction))
    }

    pub fn only_fields(&self, fields: &[Field]) -> Option<ExprRef> {
        let conj = self
            .conjunction
            .iter()
            .filter_map(|c| expr_project(c, fields))
            .collect::<Vec<_>>();

        if conj.is_empty() {
            None
        } else {
            Some(Self::from_conjunction_expr(conj))
        }
    }
}

impl Display for RowFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RowFilter({})", self.conjunction.iter().format(","))
    }
}

impl VortexExpr for RowFilter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        let mut filter_iter = self.conjunction.iter();
        let mut mask = filter_iter
            .next()
            .vortex_expect("must have at least one predicate")
            .evaluate(batch)?;
        for expr in filter_iter {
            let n_true = mask.statistics().compute_true_count().unwrap_or_default();
            let n_null = mask.statistics().compute_null_count().unwrap_or_default();

            if n_true == 0 && n_null == 0 {
                // false AND x = false
                return Ok(ConstantArray::new(false, batch.len()).into_array());
            }

            let new_mask = expr.evaluate(batch)?;
            // Either `and` or `and_kleene` is fine. They only differ on `false AND null`, but
            // fill_null only cares which values are true.
            mask = and_kleene(new_mask, mask)?;
        }

        fill_null(mask, false.into())
    }

    fn collect_references<'a>(&'a self, references: &mut HashSet<&'a Field>) {
        for expr in self.conjunction.iter() {
            expr.collect_references(references);
        }
    }
}
