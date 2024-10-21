use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use vortex::aliases::hash_set::HashSet;
use vortex::array::ConstantArray;
use vortex::compute::and;
use vortex::stats::ArrayStatistics;
use vortex::{Array, IntoArray, IntoArrayVariant};
use vortex_dtype::field::Field;
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::{split_conjunction, unbox_any, VortexExpr};

use crate::layouts::null_as_false;
use crate::layouts::read::filter_project::filter_project;

#[derive(Debug, Clone)]
pub struct RowFilter {
    conjunction: Vec<Arc<dyn VortexExpr>>,
}

impl RowFilter {
    pub fn new(expr: Arc<dyn VortexExpr>) -> Self {
        let conjunction = split_conjunction(&expr);
        Self { conjunction }
    }

    pub(crate) fn from_conjunction(conjunction: Vec<Arc<dyn VortexExpr>>) -> Self {
        Self { conjunction }
    }

    pub fn only_fields(&self, fields: &[Field]) -> Option<Self> {
        let conj = self
            .conjunction
            .iter()
            .filter_map(|c| filter_project(c, fields))
            .collect::<Vec<_>>();

        if conj.is_empty() {
            None
        } else {
            Some(Self::from_conjunction(conj))
        }
    }
}

impl VortexExpr for RowFilter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &Array) -> VortexResult<Array> {
        let mut filter_iter = self.conjunction.iter();
        let mut mask = filter_iter
            .next()
            .vortex_expect("must have at least one predicate")
            .evaluate(batch)?;
        for expr in filter_iter {
            if mask.statistics().compute_true_count().unwrap_or_default() == 0 {
                return Ok(ConstantArray::new(false, batch.len()).into_array());
            }

            let new_mask = expr.evaluate(batch)?;
            mask = and(new_mask, mask)?;
        }

        null_as_false(mask.into_bool()?)
    }

    fn collect_references<'a>(&'a self, references: &mut HashSet<&'a Field>) {
        for expr in self.conjunction.iter() {
            expr.collect_references(references);
        }
    }
}

impl PartialEq<dyn Any> for RowFilter {
    fn eq(&self, other: &dyn Any) -> bool {
        unbox_any(other)
            .downcast_ref::<Self>()
            .map(|x| x == other)
            .unwrap_or(false)
    }
}
