// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::{BitAnd, Range};
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use dashmap::DashMap;
use futures::{FutureExt, join};
use vortex_array::ArrayRef;
use vortex_array::compute::{MinMaxResult, min_max};
use vortex_array::stats::Precision;
use vortex_dict::DictArray;
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::{ExprRef, Scope, root};
use vortex_mask::Mask;

use super::DictLayout;
use crate::layouts::SharedArrayFuture;
use crate::segments::SegmentSource;
use crate::{
    ArrayEvaluation, LayoutReader, LayoutReaderRef, MaskEvaluation, NoOpPruningEvaluation,
    PruningEvaluation,
};

pub struct DictReader {
    layout: DictLayout,
    #[allow(dead_code)] // Typically used for logging
    name: Arc<str>,

    /// Length of the values array
    values_len: usize,
    /// Cached dict values array
    values_array: OnceLock<SharedArrayFuture>,
    /// Cache of expression evaluation results on the values array by expression
    values_evals: DashMap<ExprRef, SharedArrayFuture>,

    values: LayoutReaderRef,
    codes: LayoutReaderRef,
}

impl DictReader {
    pub(super) fn try_new(
        layout: DictLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
    ) -> VortexResult<Self> {
        let values_len = usize::try_from(layout.values.row_count())?;
        let values = layout
            .values
            .new_reader(format!("{name}.values").into(), segment_source.clone())?;
        let codes = layout
            .codes
            .new_reader(format!("{name}.codes").into(), segment_source)?;

        Ok(Self {
            layout,
            name,
            values_len,
            values_array: Default::default(),
            values_evals: Default::default(),
            values,
            codes,
        })
    }

    fn values_array(&self) -> SharedArrayFuture {
        // We capture the name, so it may be wrong if we re-use the same reader within multiple
        // different parent readers. But that's rare...
        let values_len = self.values_len;
        self.values_array
            .get_or_init(move || {
                let eval = self
                    .values
                    .projection_evaluation(&(0..values_len as u64), &root())
                    .vortex_expect("must construct dict values array evaluation");

                async move {
                    eval.invoke(Mask::new_true(values_len))
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
                    .map(move |array| expr.evaluate(&Scope::new(array?)).map_err(Arc::new))
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

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::Exact(self.layout.row_count())
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.codes.register_splits(field_mask, row_offset, splits)
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
        let codes_eval = self.codes.projection_evaluation(row_range, &root())?;

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
        let values_eval = self.values_eval(root());
        let codes_eval = self.codes.projection_evaluation(row_range, &root())?;
        Ok(Box::new(DictArrayEvaluation {
            values_eval,
            codes_eval,
            expr: expr.clone(),
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
                // All values are true, but we still need to respect codes validity
                let codes = self.codes_eval.invoke(Mask::new_true(mask.len())).await?;
                return Ok(mask.bitand(&codes.validity_mask()?));
            }
        }

        let codes = self.codes_eval.invoke(Mask::new_true(mask.len())).await?;
        // Creating a mask from the dict array would canonicalize it,
        // it should be fine for now as long as values is already canonical,
        // so different row ranges do not canonicalize to the same array
        // multiple times.
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
    expr: ExprRef,
}

#[async_trait]
impl ArrayEvaluation for DictArrayEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef> {
        let (values_result, codes) = join!(self.values_eval.clone(), self.codes_eval.invoke(mask));
        let (values_result, codes) = (values_result?, codes?);

        // Validate that codes are valid for the values
        let array = DictArray::try_new(codes, values_result)?.to_array();
        self.expr.evaluate(&Scope::new(array))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;
    use futures::stream;
    use rstest::rstest;
    use vortex_array::arrays::{StructArray, VarBinArray};
    use vortex_array::arrow::IntoArrowArray;
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayContext, IntoArray as _};
    use vortex_dtype::{DType, FieldName, FieldNames, Nullability};
    use vortex_expr::{is_null, not, pack, root};
    use vortex_mask::Mask;

    use crate::layouts::dict::writer::{DictLayoutOptions, DictStrategy};
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::{SequenceWriter, TestSegments};
    use crate::sequence::SequenceId;
    use crate::{
        LayoutId, LayoutRef, LayoutStrategy, LocalExecutor, SequentialStreamAdapter,
        SequentialStreamExt,
    };

    #[tokio::test]
    async fn reading_nested_packs_works() {
        let strategy = DictStrategy::new(
            FlatLayoutStrategy::default(),
            FlatLayoutStrategy::default(),
            FlatLayoutStrategy::default(),
            DictLayoutOptions::default(),
            Arc::new(LocalExecutor),
        );

        let array = VarBinArray::from_iter(
            [
                Some("abc"),
                Some("def"),
                None,
                Some("abc"),
                Some("def"),
                None,
                Some("abc"),
                Some("def"),
                None,
            ],
            DType::Utf8(Nullability::Nullable),
        )
        .to_array();
        let array_to_write = array.clone();
        let ctx = ArrayContext::empty();
        let segments = TestSegments::default();
        let layout: LayoutRef = block_on(
            strategy.write_stream(
                &ctx,
                SequenceWriter::new(Box::new(segments.clone())),
                SequentialStreamAdapter::new(
                    DType::Utf8(Nullability::Nullable),
                    stream::once(
                        async move { Ok((SequenceId::root().downgrade(), array_to_write)) },
                    ),
                )
                .sendable(),
            ),
        )
        .unwrap();

        let expression = pack(
            [(
                "top",
                pack([("one", root()), ("two", root())], Nullability::NonNullable),
            )],
            Nullability::NonNullable,
        );
        assert!(layout.encoding_id() == LayoutId::new_ref("vortex.dict"));
        let actual = layout
            .new_reader("".into(), Arc::from(segments))
            .unwrap()
            .projection_evaluation(&(0..layout.row_count()), &expression)
            .unwrap()
            .invoke(Mask::new_true(layout.row_count().try_into().unwrap()))
            .await
            .unwrap();
        let expected = StructArray::try_new(
            FieldNames::from([FieldName::from("top")]),
            vec![
                StructArray::try_new(
                    FieldNames::from([FieldName::from("one"), FieldName::from("two")]),
                    vec![array.clone(), array],
                    9,
                    Validity::NonNullable,
                )
                .unwrap()
                .into_array(),
            ],
            9,
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();
        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();
        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
    }

    #[rstest]
    #[case::all_true_case(
        vec![Some(""), None, Some("")], // Dict values: [""]
        "", // Filter for empty string
        vec![true, false, true], // Expected: nulls excluded, all dict values match
    )]
    #[case::all_false_case(
        vec![Some("x"), None, Some("x")], // Dict values: ["x"] 
        "", // Filter for empty string
        vec![false, false, false], // Expected: all false, no dict values match
    )]
    #[tokio::test]
    async fn shortpathes_filtering(
        #[case] data: Vec<Option<&str>>,
        #[case] filter_value: &str,
        #[case] expected: Vec<bool>,
    ) {
        let strategy = DictStrategy::new(
            FlatLayoutStrategy::default(),
            FlatLayoutStrategy::default(),
            FlatLayoutStrategy::default(),
            DictLayoutOptions::default(),
            Arc::new(LocalExecutor),
        );

        let array = VarBinArray::from_iter(data, DType::Utf8(Nullability::Nullable)).to_array();

        let ctx = ArrayContext::empty();
        let segments = TestSegments::default();
        let layout: LayoutRef = block_on(
            strategy.write_stream(
                &ctx,
                SequenceWriter::new(Box::new(segments.clone())),
                SequentialStreamAdapter::new(
                    DType::Utf8(Nullability::Nullable),
                    stream::once(async move { Ok((SequenceId::root().downgrade(), array)) }),
                )
                .sendable(),
            ),
        )
        .unwrap();

        let filter = vortex_expr::eq(
            root(),
            vortex_expr::lit(vortex_scalar::Scalar::utf8(
                filter_value,
                Nullability::Nullable,
            )),
        );
        let mask = layout
            .new_reader("".into(), Arc::from(segments))
            .unwrap()
            .filter_evaluation(&(0..3), &filter)
            .unwrap()
            .invoke(Mask::new_true(3))
            .await
            .unwrap();

        assert_eq!(
            mask.to_boolean_buffer().iter().collect::<Vec<_>>(),
            expected
        );
    }

    #[tokio::test]
    async fn reading_is_null_works() {
        let strategy = DictStrategy::new(
            FlatLayoutStrategy::default(),
            FlatLayoutStrategy::default(),
            FlatLayoutStrategy::default(),
            DictLayoutOptions::default(),
            Arc::new(LocalExecutor),
        );

        let array = VarBinArray::from_iter(
            [
                Some("abc"),
                Some("def"),
                None,
                Some("abc"),
                Some("def"),
                None,
                Some("abc"),
                Some("def"),
                None,
            ],
            DType::Utf8(Nullability::Nullable),
        )
        .to_array();
        let array_to_write = array.clone();
        let ctx = ArrayContext::empty();
        let segments = TestSegments::default();
        let layout: LayoutRef = block_on(
            strategy.write_stream(
                &ctx,
                SequenceWriter::new(Box::new(segments.clone())),
                SequentialStreamAdapter::new(
                    DType::Utf8(Nullability::Nullable),
                    stream::once(
                        async move { Ok((SequenceId::root().downgrade(), array_to_write)) },
                    ),
                )
                .sendable(),
            ),
        )
        .unwrap();

        let expression = not(is_null(root())); // easier to test not_is_null b/c that's the validity array
        assert!(layout.encoding_id() == LayoutId::new_ref("vortex.dict"));
        let actual = layout
            .new_reader("".into(), Arc::from(segments))
            .unwrap()
            .projection_evaluation(&(0..layout.row_count()), &expression)
            .unwrap()
            .invoke(Mask::new_true(layout.row_count().try_into().unwrap()))
            .await
            .unwrap();
        let expected = array.validity_mask().unwrap().into_array();
        let actual = actual.into_arrow_preferred().unwrap();
        let expected = expected.into_arrow_preferred().unwrap();
        assert_eq!(actual.data_type(), expected.data_type());
        assert_eq!(&actual, &expected);
    }
}
