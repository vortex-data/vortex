use std::any::Any;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use arrow_buffer::BooleanBuffer;
use itertools::Itertools;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::{BoolArray, ConstantArray};
use vortex_array::compute::and_kleene;
use vortex_array::stats::ArrayStatistics;
use vortex_array::validity::Validity;
use vortex_array::{ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};
use vortex_dtype::field::Field;
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::{split_conjunction, unbox_any, ExprRef, VortexExpr};

use crate::read::expr_project::expr_project;

#[derive(Debug, Clone)]
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
    pub fn from_conjunction_expr(conjunction: Vec<ExprRef>) -> Arc<Self> {
        Arc::new(Self::from_conjunction(conjunction))
    }

    pub fn only_fields(&self, fields: &[Field]) -> Option<Self> {
        let conj = self
            .conjunction
            .iter()
            .filter_map(|c| expr_project(c, fields))
            .collect::<Vec<_>>();

        if conj.is_empty() {
            None
        } else {
            Some(Self::from_conjunction(conj))
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
            if mask.statistics().compute_true_count().unwrap_or_default() == 0 {
                return Ok(ConstantArray::new(false, batch.len()).into_array());
            }

            let new_mask = expr.evaluate(batch)?;
            // Either `and` or `and_kleene` is fine. They only differ on `false AND null`, but
            // null_as_false only cares which values are true.
            mask = and_kleene(new_mask, mask)?;
        }

        null_as_false(mask.into_bool()?)
    }

    fn collect_references<'a>(&'a self, references: &mut HashSet<&'a Field>) {
        for expr in self.conjunction.iter() {
            expr.collect_references(references);
        }
    }
}

impl PartialEq for RowFilter {
    fn eq(&self, other: &Self) -> bool {
        self.conjunction
            .iter()
            .all(|c| other.conjunction.iter().any(|o| **o == *c.as_any()))
            && other
                .conjunction
                .iter()
                .all(|c| self.conjunction.iter().any(|o| **o == *c.as_any()))
    }
}

impl PartialEq<dyn Any> for RowFilter {
    fn eq(&self, other: &dyn Any) -> bool {
        unbox_any(other)
            .downcast_ref::<Self>()
            .map(|x| x == self)
            .unwrap_or(false)
    }
}

pub fn null_as_false(array: BoolArray) -> VortexResult<ArrayData> {
    Ok(match array.validity() {
        Validity::NonNullable => array.into_array(),
        Validity::AllValid => BoolArray::from(array.boolean_buffer()).into_array(),
        Validity::AllInvalid => BoolArray::from(BooleanBuffer::new_unset(array.len())).into_array(),
        Validity::Array(v) => {
            let bool_buffer = &array.boolean_buffer() & &v.into_bool()?.boolean_buffer();
            BoolArray::from(bool_buffer).into_array()
        }
    })
}

#[cfg(test)]
mod tests {
    use vortex_array::array::BoolArray;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArrayVariant;

    use super::*;

    #[test]
    fn coerces_nulls() {
        let bool_array = BoolArray::try_new(
            BooleanBuffer::from_iter([true, true, false, false]),
            Validity::from_iter([true, false, true, false]),
        )
        .unwrap();
        let non_null_array = null_as_false(bool_array).unwrap().into_bool().unwrap();
        assert_eq!(
            non_null_array.boolean_buffer().iter().collect::<Vec<_>>(),
            vec![true, false, false, false]
        );
    }
}
