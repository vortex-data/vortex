use std::fmt::Debug;
use std::sync::Arc;

use vortex::aliases::hash_set::HashSet;
use vortex::array::ConstantArray;
use vortex::compute::and;
use vortex::stats::ArrayStatistics;
use vortex::{Array, IntoArray, IntoArrayVariant};
use vortex_dtype::field::Field;
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::{split_conjunction, VortexExpr};

use crate::layouts::null_as_false;

#[derive(Debug, Clone)]
pub struct RowFilter {
    conjunction: Vec<Arc<dyn VortexExpr>>,
}

impl RowFilter {
    pub fn new(expr: Arc<dyn VortexExpr>) -> Self {
        let conjunction = split_conjunction(&expr);
        Self { conjunction }
    }

    /// Evaluate the underlying filter against a target array, returning a boolean mask
    pub fn evaluate(&self, target: &Array) -> VortexResult<Array> {
        let mut filter_iter = self.conjunction.iter();
        let mut mask = filter_iter
            .next()
            .vortex_expect("must have at least one predicate")
            .evaluate(target)?;
        for expr in filter_iter {
            if mask.statistics().compute_true_count().unwrap_or_default() == 0 {
                return Ok(ConstantArray::new(false, target.len()).into_array());
            }

            let new_mask = expr.evaluate(target)?;
            mask = and(new_mask, mask)?;
        }

        null_as_false(mask.into_bool()?)
    }

    /// Returns a set of all referenced fields in the underlying filter
    pub fn references(&self) -> HashSet<&Field> {
        let mut set = HashSet::new();
        for expr in self.conjunction.iter() {
            expr.collect_references(&mut set);
        }

        set
    }
}
