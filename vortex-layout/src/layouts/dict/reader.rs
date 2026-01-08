// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::try_join;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VectorExecutor;
use vortex_array::arrays::DictArray;
use vortex_array::compute::MinMaxResult;
use vortex_array::compute::min_max;
use vortex_array::compute::take;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::mask::MaskExecutor;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::vectors::VectorIntoArray;
use vortex_dtype::DType;
use vortex_dtype::FieldMask;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use super::DictLayout;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::layouts::SharedArrayFuture;
use crate::layouts::USE_VORTEX_OPERATORS;
use crate::segments::SegmentSource;

pub struct DictReader {
    layout: DictLayout,
    #[allow(dead_code)] // Typically used for logging
    name: Arc<str>,
    session: VortexSession,

    /// Length of the values array
    values_len: usize,
    /// Cached dict values array
    values_array: OnceLock<SharedArrayFuture>,
    /// Cache of expression evaluation results on the values array by expression
    values_evals: DashMap<Expression, SharedArrayFuture>,

    values: LayoutReaderRef,
    codes: LayoutReaderRef,
}

impl DictReader {
    pub(super) fn try_new(
        layout: DictLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> VortexResult<Self> {
        let values_len = usize::try_from(layout.values.row_count())?;
        let values = layout.values.new_reader(
            format!("{name}.values").into(),
            segment_source.clone(),
            &session,
        )?;
        let codes =
            layout
                .codes
                .new_reader(format!("{name}.codes").into(), segment_source, &session)?;

        Ok(Self {
            layout,
            name,
            session,
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
        let session = self.session.clone();
        self.values_array
            .get_or_init(move || {
                self.values
                    .projection_evaluation(
                        &(0..values_len as u64),
                        &root(),
                        MaskFuture::new_true(values_len),
                    )
                    .vortex_expect("must construct dict values array evaluation")
                    .map_err(Arc::new)
                    .map(move |array| {
                        if *USE_VORTEX_OPERATORS {
                            // We execute the array to avoid re-evaluating for every split.
                            let array = array?;
                            Ok(array.execute_vector(&session)?.into_array(array.dtype()))
                        } else {
                            Ok(array?.to_canonical().into_array())
                        }
                    })
                    .boxed()
                    .shared()
            })
            .clone()
    }

    fn values_eval(&self, expr: Expression) -> SharedArrayFuture {
        // This is unsound since we cannot be sure that all the values are referenced in the query
        // after applying the filter, so if the expression is fallible this might fail when it
        // shouldn't.
        // TODO(joe): fixme
        let session = self.session.clone();
        self.values_evals
            .entry(expr.clone())
            .or_insert_with(|| {
                self.values_array()
                    .map(move |array| {
                        if *USE_VORTEX_OPERATORS {
                            let array = array?.apply(&expr)?;
                            // We execute the array to avoid re-evaluating for every split.
                            Ok(array.execute_vector(&session)?.into_array(array.dtype()))
                        } else {
                            expr.evaluate(&array?).map_err(Arc::new)
                        }
                    })
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

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.codes.register_splits(field_mask, row_range, splits)
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // NOTE: we can get the values here, convert expression to the codes domain, and push down
        // to the codes child. We don't do that here because:
        // - Reading values only for an approx filter is expensive
        // - In practice, all stats based pruning evaluation should be already done upstream of this dict reader
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        // TODO(joe): fix up expr partitioning with fallible & null sensitive annotations
        let values_eval = self.values_eval(expr.clone());

        // We register interest on the entire codes row_range for now, there
        // is no straightforward shift into the codes domain we can do to the expression
        // without reading values.
        let codes_eval = self.codes.projection_evaluation(
            row_range,
            &root(),
            MaskFuture::new_true(mask.len()),
        )?;

        let session = self.session.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            // Join on the I/O futures first, before the mask.
            let (codes, values) = try_join!(codes_eval, values_eval.map_err(VortexError::from))?;
            let mask = mask.await?;

            let dict_mask = if *USE_VORTEX_OPERATORS {
                values.take(codes)?.execute_mask(&session)?
            } else {
                // Short-circuit when the values are all true/false.
                if values.all_valid()
                    && let Some(MinMaxResult { min, max }) = min_max(&values)?
                {
                    #[expect(clippy::bool_comparison, reason = "easy to follow")]
                    if max.as_bool().value().vortex_expect("non null") == false {
                        // All values are false
                        return Ok(Mask::AllFalse(mask.len()));
                    }
                    #[expect(clippy::bool_comparison, reason = "easy to follow")]
                    if min.as_bool().value().vortex_expect("not null") == true {
                        // All values are true, but we still need to respect codes validity
                        return Ok(mask.bitand(&codes.validity_mask()));
                    }
                }

                // Creating a mask from the dict array would canonicalize it,
                // it should be fine for now as long as values is already canonical,
                // so different row ranges do not canonicalize to the same array
                // multiple times.
                // TODO(joe): fixme casting null to false is *VERY* unsound, if the expression in the filter
                // can inspect nulls (e.g. `is_null`).
                // See `FlatEvaluation` for more details.
                take(&values, &codes)?.try_to_mask_fill_null_false()?
            };

            Ok(mask.bitand(&dict_mask))
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        // TODO: fix up expr partitioning with fallible & null sensitive annotations
        let values_eval = self.values_eval(root());
        let codes_eval = self
            .codes
            .projection_evaluation(row_range, &root(), mask)
            .map_err(|err| err.with_context("While evaluating projection on codes"))?;
        let expr = expr.clone();

        let all_values_referenced = self.layout.has_all_values_referenced();
        Ok(async move {
            let (values, codes) = try_join!(values_eval.map_err(VortexError::from), codes_eval)?;

            // SAFETY: Layout was validated at write time.
            //  * The codes dtype is guaranteed to be an unsigned integer type from the layout
            //  * The codes child reader ensures the correct dtype.
            //  * The layout stores `all_values_referenced` and if this is malicious then it must
            //    only affect correctness not memory safety.
            let array = unsafe {
                DictArray::new_unchecked(codes, values)
                    .set_all_values_referenced(all_values_referenced)
            }
            .to_array()
            .optimize()?;

            if *USE_VORTEX_OPERATORS {
                array.apply(&expr)
            } else {
                expr.evaluate(&array)
            }
        }
        .boxed())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray as _;
    use vortex_array::MaskFuture;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::eq;
    use vortex_array::expr::is_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::not;
    use vortex_array::expr::pack;
    use vortex_array::expr::root;
    use vortex_array::validity::Validity;
    use vortex_dtype::DType;
    use vortex_dtype::FieldName;
    use vortex_dtype::FieldNames;
    use vortex_dtype::Nullability;
    use vortex_io::runtime::single::block_on;

    use crate::LayoutId;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::dict::writer::DictLayoutOptions;
    use crate::layouts::dict::writer::DictStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::sequence::SequentialStreamAdapter;
    use crate::sequence::SequentialStreamExt;
    use crate::test::SESSION;

    #[test]
    fn reading_nested_packs_works() {
        block_on(|handle| async move {
            let strategy = DictStrategy::new(
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
                DictLayoutOptions::default(),
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
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let layout: LayoutRef = strategy
                .write_stream(
                    ctx,
                    segments.clone(),
                    SequentialStreamAdapter::new(
                        DType::Utf8(Nullability::Nullable),
                        array_to_write.to_array_stream().sequenced(ptr),
                    )
                    .sendable(),
                    eof,
                    handle,
                )
                .await
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
                .new_reader("".into(), segments, &SESSION)
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &expression,
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
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
            assert_arrays_eq!(actual, expected);
        })
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
    fn shortpathes_filtering(
        #[case] data: Vec<Option<&str>>,
        #[case] filter_value: &str,
        #[case] expected: Vec<bool>,
    ) {
        block_on(|handle| async move {
            let strategy = DictStrategy::new(
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
                DictLayoutOptions::default(),
            );

            let array = VarBinArray::from_iter(data, DType::Utf8(Nullability::Nullable)).to_array();
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let layout: LayoutRef = strategy
                .write_stream(
                    ctx,
                    segments.clone(),
                    SequentialStreamAdapter::new(
                        DType::Utf8(Nullability::Nullable),
                        array.to_array_stream().sequenced(ptr),
                    )
                    .sendable(),
                    eof,
                    handle,
                )
                .await
                .unwrap();

            let filter = eq(
                root(),
                lit(vortex_scalar::Scalar::utf8(
                    filter_value,
                    Nullability::Nullable,
                )),
            );
            let mask = layout
                .new_reader("".into(), segments, &SESSION)
                .unwrap()
                .filter_evaluation(&(0..3), &filter, MaskFuture::new_true(3))
                .unwrap()
                .await
                .unwrap();

            assert_eq!(mask.to_bit_buffer().iter().collect::<Vec<_>>(), expected);
        })
    }

    #[test]
    fn reading_is_null_works() {
        block_on(|handle| async move {
            let strategy = DictStrategy::new(
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
                DictLayoutOptions::default(),
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

            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let layout: LayoutRef = strategy
                .write_stream(
                    ctx,
                    segments.clone(),
                    SequentialStreamAdapter::new(
                        DType::Utf8(Nullability::Nullable),
                        array_to_write.to_array_stream().sequenced(ptr),
                    )
                    .sendable(),
                    eof,
                    handle,
                )
                .await
                .unwrap();

            let expression = not(is_null(root())); // easier to test not_is_null b/c that's the validity array
            assert!(layout.encoding_id() == LayoutId::new_ref("vortex.dict"));
            let actual = layout
                .new_reader("".into(), segments, &SESSION)
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &expression,
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();
            let expected = array.validity_mask().into_array();
            assert_arrays_eq!(actual.to_canonical().into_array(), expected);
        })
    }
}
