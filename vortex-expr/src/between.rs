use std::any::Any;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use vortex_array::compute::between;
use vortex_array::Array;
use vortex_dtype::DType;
use vortex_dtype::DType::Bool;
use vortex_error::{vortex_err, VortexResult};

use crate::{ExprRef, Operator, VortexExpr};

#[derive(Debug, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct Between {
    arr: ExprRef,
    lower: ExprRef,
    lower_op: Operator,
    upper: ExprRef,
    upper_op: Operator,
}

impl Between {
    pub fn between(
        arr: ExprRef,
        lower: ExprRef,
        lower_op: Operator,
        upper: ExprRef,
        upper_op: Operator,
    ) -> ExprRef {
        Arc::new(Self {
            arr,
            lower,
            lower_op,
            upper,
            upper_op,
        })
    }
}

impl Display for Between {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({} {} {} {} {})",
            self.lower, self.lower_op, self.arr, self.upper_op, self.upper
        )
    }
}

impl PartialEq for Between {
    fn eq(&self, other: &Between) -> bool {
        self.arr.eq(&other.arr)
            && other.lower.eq(&self.lower)
            && self.lower_op == other.lower_op
            && other.upper.eq(&self.upper)
            && self.upper_op == other.upper_op
    }
}

impl VortexExpr for Between {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &Array) -> VortexResult<Array> {
        let arr_val = self.arr.evaluate(batch)?;
        let lower_val = self.lower.evaluate(batch)?;
        let upper_arr_val = self.upper.evaluate(batch)?;

        between(
            &arr_val,
            &lower_val,
            self.lower_op
                .maybe_cmp_operator()
                .ok_or_else(|| vortex_err!("must be a cmp operator"))?,
            &upper_arr_val,
            self.upper_op
                .maybe_cmp_operator()
                .ok_or_else(|| vortex_err!("must be a cmp operator"))?,
        )
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![&self.arr, &self.lower, &self.upper]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        Arc::new(Self {
            arr: children[0].clone(),
            lower: children[1].clone(),
            lower_op: self.lower_op,
            upper: children[2].clone(),
            upper_op: self.upper_op,
        })
    }

    fn return_dtype(&self, scope_dtype: &DType) -> VortexResult<DType> {
        let arr_dt = self.arr.return_dtype(scope_dtype)?;
        let lower_dt = self.lower.return_dtype(scope_dtype)?;
        let upper_dt = self.upper.return_dtype(scope_dtype)?;

        assert!(arr_dt.eq_ignore_nullability(&lower_dt));
        assert!(arr_dt.eq_ignore_nullability(&upper_dt));

        Ok(Bool(
            arr_dt.nullability() | lower_dt.nullability() | upper_dt.nullability(),
        ))
    }
}
