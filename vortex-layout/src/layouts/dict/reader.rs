use std::ops::{BitAnd, Deref, Range};
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use dashmap::DashMap;
use futures::{FutureExt, join};
use vortex_array::arrays::StructArray;
use vortex_array::compute::{MinMaxResult, filter, min_max};
use vortex_array::{Array, ArrayContext, ArrayRef, ToCanonical};
use vortex_dict::DictArray;
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::{ExprRef, Identity};
use vortex_mask::Mask;

use super::DictLayout;
use crate::layouts::SharedArrayFuture;
use crate::segments::SegmentSource;
use crate::{
    ArrayEvaluation, Layout, LayoutReader, LayoutReaderRef, MaskEvaluation, NoOpPruningEvaluation,
    PruningEvaluation,
};

pub struct DictReader {
    layout: DictLayout,
    #[allow(dead_code)] // Typically used for logging
    name: Arc<str>,

    /// Cached dict values array
    values_array: OnceLock<SharedArrayFuture>,
    /// Cache of expression evaluation results on the values array by expression
    values_evals: DashMap<ExprRef, SharedArrayFuture>,

    values: LayoutReaderRef,
    codes: LayoutReaderRef,
}

impl Deref for DictReader {
    type Target = dyn Layout;

    fn deref(&self) -> &Self::Target {
        self.layout.deref()
    }
}

impl DictReader {
    pub(super) fn try_new(
        layout: DictLayout,
        name: Arc<str>,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<Self> {
        let values =
            layout
                .values
                .new_reader(&format!("{}.values", name).into(), segment_source, ctx)?;
        let codes =
            layout
                .codes
                .new_reader(&format!("{}.codes", name).into(), segment_source, ctx)?;

        Ok(Self {
            layout,
            name,
            values_array: Default::default(),
            values_evals: Default::default(),
            values,
            codes,
        })
    }

    fn values_array(&self) -> SharedArrayFuture {
        // We capture the name, so it may be wrong if we re-use the same reader within multiple
        // different parent readers. But that's rare...
        self.values_array
            .get_or_init(move || {
                let values_len = self.values.row_count();
                let eval = self
                    .values
                    .projection_evaluation(&(0..values_len), &Identity::new_expr())
                    .vortex_expect("must construct dict values array evaluation");

                async move {
                    eval.invoke(Mask::new_true(
                        usize::try_from(values_len)
                            .vortex_expect("dict values length must fit in u32"),
                    ))
                    .await
                    .map_err(Arc::new)
                }
                .boxed()
                .shared()
            })
            .clone()
    }

    fn values_eval(&self, expr: ExprRef) -> SharedArrayFuture {
        self.values_evals
            .entry(expr.clone())
            .or_insert_with(|| {
                self.values_array()
                    .map(move |array| expr.evaluate(&array?).map_err(Arc::new))
                    .boxed()
                    .shared()
            })
            .clone()
    }
}

impl LayoutReader for DictReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

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

        // Short-circuit when the values are all true/false.
        if let Some(MinMaxResult { min, max }) = min_max(&values_result)? {
            if !max.as_bool().value().unwrap_or(true) {
                // All values are false
                return Ok(Mask::AllFalse(mask.len()));
            }
            if min.as_bool().value().unwrap_or(false) {
                // All values are true
                return Ok(mask);
            }
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

        if values_result.dtype().is_struct() {
            // If the expression returns a struct push down the dict creation,
            // return a struct of dicts
            let values_result = values_result.to_struct()?;
            Ok(StructArray::try_new(
                values_result.names().clone(),
                values_result
                    .fields()
                    .iter()
                    .map(|field| Ok(DictArray::try_new(codes.clone(), field.clone())?.to_array()))
                    .collect::<VortexResult<Vec<_>>>()?,
                codes.len(),
                values_result.dtype().nullability().into(),
            )?
            .to_array())
        } else {
            Ok(DictArray::try_new(codes, values_result)?.to_array())
        }
    }
}
