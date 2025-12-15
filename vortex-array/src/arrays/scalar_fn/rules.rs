// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_scalar::Scalar;
use vortex_vector::Datum;
use vortex_vector::VectorOps;
use vortex_vector::datum_matches_dtype;

use crate::Array;
use crate::ArrayRef;
use crate::ArrayVisitor;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::ScalarFnArray;
use crate::arrays::ScalarFnVTable;
use crate::expr::ExecutionArgs;
use crate::expr::ReduceCtx;
use crate::expr::ReduceNode;
use crate::expr::ReduceNodeRef;
use crate::expr::ScalarFn;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::optimizer::rules::ReduceRuleSet;

pub(super) const RULES: ReduceRuleSet<ScalarFnVTable> =
    ReduceRuleSet::new(&[&ScalarFnConstantRule, &ScalarFnAbstractReduceRule]);

pub(super) const PARENT_RULES: ParentRuleSet<ScalarFnVTable> = ParentRuleSet::new(&[]);

#[derive(Debug)]
struct ScalarFnConstantRule;
impl ArrayReduceRule<ScalarFnVTable> for ScalarFnConstantRule {
    fn reduce(&self, array: &ScalarFnArray) -> VortexResult<Option<ArrayRef>> {
        if !array.children.iter().all(|c| c.is::<ConstantVTable>()) {
            return Ok(None);
        }

        let input_datums: Vec<_> = array
            .children
            .iter()
            .map(|c| c.as_::<ConstantVTable>().scalar().to_vector_scalar())
            .map(Datum::Scalar)
            .collect();
        let input_dtypes = array.children.iter().map(|c| c.dtype().clone()).collect();

        let result = array.scalar_fn.execute(ExecutionArgs {
            datums: input_datums,
            dtypes: input_dtypes,
            row_count: array.len,
            return_dtype: array.dtype.clone(),
        })?;
        vortex_ensure!(
            datum_matches_dtype(&result, &array.dtype),
            "Scalar function {} result does not match expected dtype",
            array.scalar_fn
        );

        let result = match result {
            Datum::Scalar(s) => s,
            Datum::Vector(v) => {
                tracing::warn!(
                    "Scalar function {} returned vector from execution over all scalar inputs",
                    array.scalar_fn
                );
                v.scalar_at(0)
            }
        };

        Ok(Some(
            ConstantArray::new(Scalar::from_vector_scalar(result, &array.dtype)?, array.len)
                .into_array(),
        ))
    }
}

#[derive(Debug)]
struct ScalarFnAbstractReduceRule;
impl ArrayReduceRule<ScalarFnVTable> for ScalarFnAbstractReduceRule {
    fn reduce(&self, array: &ScalarFnArray) -> VortexResult<Option<ArrayRef>> {
        if let Some(reduced) = array.scalar_fn.reduce(
            // Blergh, re-boxing
            &array.to_array(),
            &ArrayReduceCtx { len: array.len },
        )? {
            return Ok(Some(
                reduced
                    .as_any()
                    .downcast_ref::<ArrayRef>()
                    .vortex_expect("ReduceNode is not an ArrayRef")
                    .clone(),
            ));
        }
        Ok(None)
    }
}

impl ReduceNode for ArrayRef {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn node_dtype(&self) -> VortexResult<DType> {
        Ok(self.as_ref().dtype().clone())
    }

    fn scalar_fn(&self) -> Option<&ScalarFn> {
        self.as_opt::<ScalarFnVTable>().map(|a| a.scalar_fn())
    }

    fn child(&self, idx: usize) -> ReduceNodeRef {
        Arc::new(<dyn Array>::children(self)[idx].clone())
    }

    fn child_count(&self) -> usize {
        self.nchildren()
    }
}

struct ArrayReduceCtx {
    // The length of the array being reduced
    len: usize,
}
impl ReduceCtx for ArrayReduceCtx {
    fn new_node(
        &self,
        scalar_fn: ScalarFn,
        children: &[ReduceNodeRef],
    ) -> VortexResult<ReduceNodeRef> {
        Ok(Arc::new(
            ScalarFnArray::try_new(
                scalar_fn,
                children
                    .iter()
                    .map(|c| {
                        c.as_any()
                            .downcast_ref::<ArrayRef>()
                            .vortex_expect("ReduceNode is not an ArrayRef")
                            .clone()
                    })
                    .collect(),
                self.len,
            )?
            .into_array(),
        ))
    }
}
