// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::{BitAnd, Range};
use std::sync::{Arc, OnceLock};

use futures::future::BoxFuture;
use futures::{FutureExt, TryFutureExt, try_join};
use vortex_array::compute::{MinMaxResult, min_max, take};
use vortex_array::{ArrayRef, MaskFuture};
use vortex_dict::DictArray;
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexError, VortexExpect, VortexResult};
use vortex_expr::{ExprRef, Scope, root};
use vortex_mask::Mask;
use vortex_utils::aliases::dash_map::DashMap;

use super::DictLayout;
use crate::layouts::SharedArrayFuture;
use crate::segments::SegmentSource;
use crate::{LayoutReader, LayoutReaderRef};

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
                self.values
                    .projection_evaluation(
                        &(0..values_len as u64),
                        &root(),
                        MaskFuture::new_true(values_len),
                    )
                    .vortex_expect("must construct dict values array evaluation")
                    .map_err(Arc::new)
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

    fn row_count(&self) -> u64 {
        self.layout.row_count()
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
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let values_eval = self.values_eval(expr.clone());

        // We register interest on the entire codes row_range for now, there
        // is no straightforward shift into the codes domain we can do to the expression
        // without reading values.
        let codes_eval = self.codes.projection_evaluation(
            row_range,
            &root(),
            MaskFuture::new_true(mask.len()),
        )?;

        Ok(MaskFuture::new(mask.len(), async move {
            // Join on the I/O futures first, before the mask.
            let (codes, values) = try_join!(codes_eval, values_eval.map_err(VortexError::from))?;
            let mask = mask.await?;

            // Short-circuit when the values are all true/false.
            if let Some(MinMaxResult { min, max }) = min_max(&values)? {
                if !max.as_bool().value().unwrap_or(true) {
                    // All values are false
                    return Ok(Mask::AllFalse(mask.len()));
                }
                if min.as_bool().value().unwrap_or(false) {
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
            let dict_mask = take(&values, &codes)?.try_to_mask_fill_null_false()?;

            Ok(mask.bitand(&dict_mask))
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let values_eval = self.values_eval(root());
        let codes_eval = self.codes.projection_evaluation(row_range, &root(), mask)?;
        let expr = expr.clone();

        Ok(async move {
            let (values, codes) = try_join!(values_eval.map_err(VortexError::from), codes_eval)?;

            // Validate that codes are valid for the values
            let array = DictArray::try_new(codes, values)?.to_array();
            expr.evaluate(&Scope::new(array))
        }
        .boxed())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::arrays::{StructArray, VarBinArray};
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayContext, IntoArray as _, MaskFuture, assert_arrays_eq};
    use vortex_dtype::{DType, FieldName, FieldNames, Nullability};
    use vortex_expr::{is_null, not, pack, root};
    use vortex_io::runtime::single::block_on;

    use crate::layouts::dict::writer::{DictLayoutOptions, DictStrategy};
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::{
        SequenceId, SequentialArrayStreamExt, SequentialStreamAdapter, SequentialStreamExt,
    };
    use crate::{LayoutId, LayoutRef, LayoutStrategy};

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
                .new_reader("".into(), segments)
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

            let filter = vortex_expr::eq(
                root(),
                vortex_expr::lit(vortex_scalar::Scalar::utf8(
                    filter_value,
                    Nullability::Nullable,
                )),
            );
            let mask = layout
                .new_reader("".into(), segments)
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
                .new_reader("".into(), segments)
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
            assert_arrays_eq!(actual, expected);
        })
    }
}
