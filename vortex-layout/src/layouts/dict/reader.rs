use std::sync::{Arc, OnceLock, RwLock};

use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::{ArrayContext, ArrayRef, DeserializeMetadata, IntoArray, ProstMetadata};
use vortex_dtype::{DType, PType};
use vortex_error::{SharedVortexResult, VortexExpect, VortexResult, vortex_panic};
use vortex_expr::{ExprRef, Identity};
use vortex_mask::Mask;

use super::DictLayout;
use super::writer::DictLayoutMetadata;
use crate::segments::SegmentSource;
use crate::{Layout, LayoutReader, LayoutVTable};

pub(crate) type SharedArrayFuture = Shared<BoxFuture<'static, SharedVortexResult<ArrayRef>>>;

pub struct DictReader {
    layout: Layout,
    /// Cached dict values array
    values_array: OnceLock<SharedArrayFuture>,
    /// Cache of expression evaluation results on the values array by expression
    values_evals: RwLock<HashMap<ExprRef, SharedArrayFuture>>,
    pub(crate) values: Arc<dyn LayoutReader>,
    pub(crate) codes: Arc<dyn LayoutReader>,
}

impl DictReader {
    pub(super) fn try_new(
        layout: Layout,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<Self> {
        if layout.vtable().id() != DictLayout.id() {
            vortex_panic!("Mitmatched layout ID")
        }
        let metadata = ProstMetadata::<DictLayoutMetadata>::deserialize(
            layout.metadata().as_ref().map(|b| b.as_ref()),
        )?;

        let values = layout
            .child(0, layout.dtype().clone(), "values")?
            .reader(segment_source, ctx)?;

        let codes_dtype = DType::from(PType::from(metadata.codes_ptype()))
            .with_nullability(values.dtype().nullability());

        let codes = layout
            .child(1, codes_dtype, "codes")?
            .reader(segment_source, ctx)?;
        Ok(Self {
            layout,
            values_array: Default::default(),
            values_evals: Default::default(),
            values,
            codes,
        })
    }

    pub(crate) fn values_array(&self) -> SharedArrayFuture {
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

    pub(crate) fn values_eval(&self, expr: ExprRef) -> SharedArrayFuture {
        self.values_evals
            .write()
            .vortex_expect("poisoned lock")
            .entry(expr.clone())
            .or_insert_with(|| {
                self.values_array()
                    .map(move |array| {
                        expr.evaluate(&array?)
                            .and_then(|result| result.to_canonical())
                            // TODO(os): not all expressions would benefit from a canonical array
                            .map(|canonical| canonical.into_array())
                            .map_err(Arc::new)
                    })
                    .boxed()
                    .shared()
            })
            .clone()
    }
}

impl LayoutReader for DictReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn children(&self) -> VortexResult<Vec<Arc<dyn LayoutReader>>> {
        Ok(vec![self.values.clone(), self.codes.clone()])
    }
}
