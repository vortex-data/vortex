use std::ops::{BitAnd, Range};

use async_trait::async_trait;
use vortex_array::{Array, ArrayRef};
use vortex_dict::DictArray;
use vortex_error::VortexResult;
use vortex_expr::{ExprRef, Identity};
use vortex_mask::Mask;

use super::reader::{DictReader, SharedArrayFuture};
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
        let codes_eval = self.codes.projection_evaluation(row_range, expr)?;
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

        // TODO(os): if mask density is low, we should run take on codes first?
        let values_result = self.values_eval.clone().await?;
        let codes = self.codes_eval.invoke(Mask::new_true(mask.len())).await?;

        // creating a mask from the dict array would canonicalise it,
        // it should be fine for now as long as values is already canonical,
        // so different row ranges do not double canonicalise it
        let dict_mask = &Mask::try_from(
            DictArray::try_new(codes, values_result)?
                .to_array()
                .as_ref(),
        )?;
        Ok(mask.bitand(dict_mask))
    }
}

struct DictArrayEvaluation {
    values_eval: SharedArrayFuture,
    codes_eval: Box<dyn ArrayEvaluation>,
}

#[async_trait]
impl ArrayEvaluation for DictArrayEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef> {
        let values_result = self.values_eval.clone().await?;
        let codes = self.codes_eval.invoke(mask).await?;
        Ok(DictArray::try_new(codes, values_result)?.to_array())
    }
}
