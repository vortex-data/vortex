use std::ops::{BitAnd, Range};

use async_trait::async_trait;
use futures::join;
use vortex_array::arrays::StructArray;
use vortex_array::compute::filter;
use vortex_array::{Array, ArrayRef};
use vortex_dict::DictArray;
use vortex_error::{VortexResult, vortex_err};
use vortex_expr::{ExprRef, Identity};
use vortex_mask::Mask;

use super::reader::DictReader;
use crate::layouts::SharedArrayFuture;
use crate::{
    ArrayEvaluation, ExprEvaluator, MaskEvaluation, NoOpPruningEvaluation, PruningEvaluation,
};

impl ExprEvaluator for DictReader {
    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        // NOTE: we can get the values here, convert expression to the codes domain, and push down
        // to the codes child. We don't do that here because:
        // - Reading values only for an approx filter is expensive
        // - In practice, all stats based pruning evaluation should be already done upstream of this dict reader
        Ok(Box::new(NoOpPruningEvaluation))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        let values_eval = self.values_eval(expr.clone());

        // We register interest on the entire codes row_range for now, there
        // is no straightforward shift into the codes domain we can do to the expression
        // without reading values.
        let codes_eval = self
            .codes
            .projection_evaluation(row_range, &Identity::new_expr())?;

        Ok(Box::new(DictMaskEvaluation {
            values_eval,
            codes_eval,
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        let values_eval = self.values_eval(expr.clone());
        let codes_eval = self
            .codes
            .projection_evaluation(row_range, &Identity::new_expr())?;
        Ok(Box::new(DictArrayEvaluation {
            values_eval,
            codes_eval,
        }))
    }
}

struct DictMaskEvaluation {
    values_eval: SharedArrayFuture,
    codes_eval: Box<dyn ArrayEvaluation>,
}

#[async_trait]
impl MaskEvaluation for DictMaskEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        if mask.all_false() {
            return Ok(mask);
        }

        let values_result = self.values_eval.clone().await?;
        match values_result
            .as_bool_typed()
            .ok_or_else(|| vortex_err!("expr must return bool"))?
            .true_count()?
        {
            0 => return Ok(Mask::AllFalse(mask.len())),
            count if count == values_result.len() => {
                return Ok(mask);
            }
            _ => (),
        }

        let codes = self.codes_eval.invoke(Mask::new_true(mask.len())).await?;
        // TODO(os): remove the low density code path, does not really improve perf
        if mask.density() < 0.1 {
            let codes = filter(&codes, &mask)?;
            let dict_mask = &Mask::try_from(
                DictArray::try_new(codes, values_result)?
                    .to_array()
                    .as_ref(),
            )?;
            Ok(mask.intersect_by_rank(dict_mask))
        } else {
            // Creating a mask from the dict array would canonicalise it,
            // it should be fine for now as long as values is already canonical,
            // so different row ranges do not canonicalise the same array
            // multiple times.
            let dict_mask = &Mask::try_from(
                DictArray::try_new(codes, values_result)?
                    .to_array()
                    .as_ref(),
            )?;
            Ok(mask.bitand(dict_mask))
        }
    }
}

struct DictArrayEvaluation {
    values_eval: SharedArrayFuture,
    codes_eval: Box<dyn ArrayEvaluation>,
}

#[async_trait]
impl ArrayEvaluation for DictArrayEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef> {
        let (values_result, codes) = join!(self.values_eval.clone(), self.codes_eval.invoke(mask));
        let (values_result, codes) = (values_result?, codes?);

        match values_result.as_struct_typed() {
            // If the expression returns a struct push down the dict creation,
            // return a struct of dicts
            Some(struct_typed) => Ok(StructArray::try_new(
                struct_typed.names().clone(),
                struct_typed
                    .fields()
                    .map(|field| Ok(DictArray::try_new(codes.clone(), field)?.to_array()))
                    .collect::<VortexResult<Vec<_>>>()?,
                codes.len(),
                struct_typed.dtype().nullability().into(),
            )?
            .to_array()),
            None => Ok(DictArray::try_new(codes, values_result)?.to_array()),
        }
    }
}
