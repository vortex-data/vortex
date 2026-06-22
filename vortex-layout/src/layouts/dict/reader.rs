// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::try_join;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::SharedArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::direct_annotations;
use vortex_array::expr::is_root;
use vortex_array::expr::label_tree;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::transform::partition_annotations;
use vortex_array::scalar_fn::is_negative_cost;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use super::DictLayout;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::SharedArrayFuture;
use crate::segments::SegmentSource;

pub struct DictReader {
    layout: DictLayout,
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
        ctx: crate::LayoutReaderContext,
    ) -> VortexResult<Self> {
        let values_len = usize::try_from(layout.values.row_count())?;
        let values = layout.values.new_reader(
            format!("{name}.values").into(),
            Arc::clone(&segment_source),
            &session,
            &ctx,
        )?;
        let codes = layout.codes.new_reader(
            format!("{name}.codes").into(),
            segment_source,
            &session,
            &ctx,
        )?;

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
                    .map(move |array| Ok(SharedArray::new(array?).into_array()))
                    .boxed()
                    .shared()
            })
            .clone()
    }

    // This is the dict values array without canonicalization, if not already canonical
    fn values_array_uncanonical(&self) -> SharedArrayFuture {
        // We capture the name, so it may be wrong if we re-use the same reader within multiple
        // different parent readers. But that's rare...
        let values_len = self.values_len;
        self.values_array.get().cloned().unwrap_or_else(|| {
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
    }

    fn values_eval(&self, expr: Expression) -> SharedArrayFuture {
        // This is unsound since we cannot be sure that all the values are referenced in the query
        // after applying the filter, so if the expression is fallible this might fail when it
        // shouldn't.
        // TODO(joe): fixme

        // Check cache first with read-only lock
        if let Some(fut) = self.values_evals.get(&expr) {
            return fut.clone();
        }

        self.values_evals
            .entry(expr.clone())
            .or_insert_with(|| {
                self.values_array_uncanonical()
                    .map(move |array| {
                        let array = array?.apply(&expr)?;
                        Ok(SharedArray::new(array).into_array())
                    })
                    .boxed()
                    .shared()
            })
            .clone()
    }
}

// On expression pushdown, "inner" is packed as field with this name.
// "outer" expects this name as input.
const PUSHDOWN_ANNOTATION: &str = "";

/// Split expression into two parts:
///
/// left is the outer part that we want to apply to array after canonicalizing.
/// right is the optional inner part that we want to apply to array before
/// canonicalizing.
///
/// We want to push to array only if expression has a negative cost, is
/// infallible and null-insensitive.
fn split_expression_for_pushdown(
    expr: Expression,
    dtype: &DType,
) -> VortexResult<(Expression, Option<Expression>)> {
    let references_root = label_tree(&expr, is_root, |acc, &child| acc | child);
    let annotations = direct_annotations(&expr, |expr| {
        let signature = expr.signature();
        if !signature.is_fallible()
            && !signature.is_null_sensitive()
            && is_negative_cost(expr.id())
            && references_root.get(&expr).copied().unwrap_or(true)
        {
            vec![PUSHDOWN_ANNOTATION]
        } else {
            vec![]
        }
    });
    let partition = partition_annotations(expr.clone(), dtype, annotations)?;
    if partition.partitions.is_empty() {
        Ok((partition.root, None))
    } else {
        debug_assert_eq!(1, partition.partitions.len());
        Ok((partition.root, Some(partition.partitions[0].clone())))
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
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        self.codes.register_splits(field_mask, split_range, splits)
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

            let mut ctx = session.create_execution_ctx();
            let dict_mask = values.take(codes)?.null_as_false().execute(&mut ctx)?;

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
        let codes_eval = self
            .codes
            .projection_evaluation(row_range, &root(), mask)
            .map_err(|err| err.with_context("While evaluating projection on codes"))?;

        let (expr_outer, expr_inner) = split_expression_for_pushdown(expr.clone(), self.dtype())?;

        let values_eval = if let Some(inner) = expr_inner {
            // "outer" takes a struct field with PUSHDOWN_ANNOTATION name, so
            // pack inner with this name as well
            let inner = pack([(PUSHDOWN_ANNOTATION, inner)], Nullability::NonNullable);

            // We can't use values_eval as it uses values_array_uncanonical
            // which in turn gets populated from self.values. If
            // self.values_array() is called first, it will populate
            // self.values with uncompressed data. Supply uncached data
            let values_len = self.values_len;
            self.values
                .projection_evaluation(
                    &(0..values_len as u64),
                    &root(),
                    MaskFuture::new_true(values_len),
                )
                .vortex_expect("must construct dict values array evaluation")
                .map_err(Arc::new)
                .map(move |array| Ok(SharedArray::new(array?.apply(&inner)?).into_array()))
                .boxed()
                .shared()
        } else {
            self.values_array()
        };
        let all_values_referenced = self.layout.has_all_values_referenced();
        Ok(async move {
            let (values, codes) = try_join!(values_eval.map_err(VortexError::from), codes_eval)?;

            // SAFETY: Layout was validated at write time.
            //  * The codes dtype is guaranteed to be an integer type from the layout
            //  * The codes child reader ensures the correct dtype.
            //  * The layout stores `all_values_referenced` and if this is malicious then it must
            //    only affect correctness not memory safety.
            let parts =
                unsafe { DictArray::new_unchecked_parts(codes, values, all_values_referenced) };
            let array = parts.optimize()?;

            array.apply(&expr_outer)
        }
        .boxed())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray as _;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::Expression;
    use vortex_array::expr::byte_length;
    use vortex_array::expr::cast;
    use vortex_array::expr::eq;
    use vortex_array::expr::get_item;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::like;
    use vortex_array::expr::lit;
    use vortex_array::expr::pack;
    use vortex_array::expr::root;
    use vortex_array::validity::Validity;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_io::runtime::Handle;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_session::VortexSession;

    use crate::LayoutId;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::dict::reader::PUSHDOWN_ANNOTATION;
    use crate::layouts::dict::reader::split_expression_for_pushdown;
    use crate::layouts::dict::writer::DictLayoutOptions;
    use crate::layouts::dict::writer::DictStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::sequence::SequentialStreamAdapter;
    use crate::sequence::SequentialStreamExt;
    use crate::session::LayoutSession;

    // FIXME(ngates): Deprecate the global `runtime::single::block_on` helper and require tests
    // to call `block_on` on an explicit runtime instance.
    fn session_with_handle(handle: Handle) -> VortexSession {
        vortex_array::array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle)
    }

    async fn write_dict_layout(
        array: ArrayRef,
        session: &VortexSession,
    ) -> (LayoutRef, Arc<TestSegments>) {
        let strategy = DictStrategy::new(
            FlatLayoutStrategy::default(),
            FlatLayoutStrategy::default(),
            FlatLayoutStrategy::default(),
            DictLayoutOptions::default(),
        );
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let dtype = array.dtype().clone();
        let layout = strategy
            .write_stream(
                ArrayContext::empty(),
                Arc::<TestSegments>::clone(&segments),
                SequentialStreamAdapter::new(dtype, array.to_array_stream().sequenced(ptr))
                    .sendable(),
                eof,
                session,
            )
            .await
            .unwrap();
        (layout, segments)
    }

    #[test]
    fn reading_nested_packs_works() {
        block_on(|handle| async move {
            let session = session_with_handle(handle);
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
            .into_array();
            let array_to_write = array.clone();
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let layout: LayoutRef = strategy
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    SequentialStreamAdapter::new(
                        DType::Utf8(Nullability::Nullable),
                        array_to_write.to_array_stream().sequenced(ptr),
                    )
                    .sendable(),
                    eof,
                    &session,
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
            assert!(layout.encoding_id() == LayoutId::new("vortex.dict"));
            let actual = layout
                .new_reader("".into(), segments, &session, &Default::default())
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
            let session = session_with_handle(handle);
            let strategy = DictStrategy::new(
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
                DictLayoutOptions::default(),
            );

            let array =
                VarBinArray::from_iter(data, DType::Utf8(Nullability::Nullable)).into_array();
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let layout: LayoutRef = strategy
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    SequentialStreamAdapter::new(
                        DType::Utf8(Nullability::Nullable),
                        array.to_array_stream().sequenced(ptr),
                    )
                    .sendable(),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let filter = eq(
                root(),
                lit(vortex_array::scalar::Scalar::utf8(
                    filter_value,
                    Nullability::Nullable,
                )),
            );
            let mask = layout
                .new_reader("".into(), segments, &session, &Default::default())
                .unwrap()
                .filter_evaluation(&(0..3), &filter, MaskFuture::new_true(3))
                .unwrap()
                .await
                .unwrap();

            assert_arrays_eq!(mask.into_array(), BoolArray::from_iter(expected));
        })
    }

    #[test]
    fn reading_is_null_works() {
        block_on(|handle| async move {
            let mut ctx_exec = LEGACY_SESSION.create_execution_ctx();
            let session = session_with_handle(handle);
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
            .into_array();
            let array_to_write = array.clone();
            let ctx = ArrayContext::empty();

            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let layout: LayoutRef = strategy
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    SequentialStreamAdapter::new(
                        DType::Utf8(Nullability::Nullable),
                        array_to_write.to_array_stream().sequenced(ptr),
                    )
                    .sendable(),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let expression = is_not_null(root());
            assert_eq!(layout.encoding_id(), LayoutId::new("vortex.dict"));
            let actual = layout
                .new_reader("".into(), segments, &session, &Default::default())
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &expression,
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();
            let expected = array
                .validity()
                .unwrap()
                .execute_mask(array.len(), &mut ctx_exec)
                .unwrap()
                .into_array();
            let actual_canonical = actual
                .execute::<Canonical>(&mut ctx_exec)
                .vortex_expect("to_canonical failed")
                .into_array();
            assert_arrays_eq!(actual_canonical, expected);
        })
    }

    #[test]
    fn reading_byte_length_pushdown_works() {
        let array = VarBinArray::from_iter(
            [
                Some("abc"),
                Some("defg"),
                None,
                Some("abc"),
                Some("defg"),
                None,
                Some("abc"),
                Some("defg"),
                None,
            ],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();

        let expected = array
            .clone()
            .apply(&byte_length(root()))
            .unwrap()
            .into_array();

        block_on(|handle| async move {
            let session = session_with_handle(handle);
            let (layout, segments) = write_dict_layout(array, &session).await;
            assert_eq!(layout.encoding_id(), LayoutId::new("vortex.dict"));
            let actual = layout
                .new_reader("".into(), segments, &session, &Default::default())
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &byte_length(root()),
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap()
                .into_array();
            assert_arrays_eq!(actual, expected);
        })
    }

    fn pushed_inner(exprs: impl IntoIterator<Item = Expression>) -> Expression {
        pack(
            exprs
                .into_iter()
                .enumerate()
                .map(|(idx, e)| (format!("_{idx}"), e)),
            Nullability::NonNullable,
        )
    }

    fn pushed_ref(idx: usize) -> Expression {
        get_item(format!("_{idx}"), get_item("", root()))
    }

    fn test_apply(original: Expression, outer: Expression, inner: Expression) -> VortexResult<()> {
        let array = VarBinArray::from_iter(
            [Some("abc"), Some("def"), None],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();

        let pushed = array.clone().apply(&pack(
            [(PUSHDOWN_ANNOTATION, inner)],
            Nullability::NonNullable,
        ))?;
        let actual = pushed.apply(&outer)?;
        let expected = array.apply(&original)?;
        assert_arrays_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn split_expr_root() {
        let (outer, inner) = split_expression_for_pushdown(root(), &DType::Null).unwrap();
        assert_eq!(outer, root());
        assert_eq!(inner, None);
    }

    #[test]
    fn split_expr_partial_pushdown() -> VortexResult<()> {
        // cast is fallible, thus not pushed
        let target = DType::Primitive(PType::I64, Nullability::Nullable);
        let expr = cast(byte_length(root()), target.clone());
        let (outer, inner) =
            split_expression_for_pushdown(expr.clone(), &DType::Utf8(false.into()))?;
        let inner = inner.unwrap();
        // [0] = cast([1], dtype)
        // [1] = byte_length(root)
        assert_eq!(outer, cast(pushed_ref(0), target));
        assert_eq!(inner, pushed_inner([byte_length(root())]));
        test_apply(expr, outer, inner)
    }

    #[test]
    fn split_expr_full_pushdown() -> VortexResult<()> {
        let expr = byte_length(root());
        let (outer, inner) =
            split_expression_for_pushdown(expr.clone(), &DType::Utf8(false.into()))?;
        let inner = inner.unwrap();
        assert_eq!(outer, pushed_ref(0));
        assert_eq!(inner, pushed_inner([byte_length(root())]));
        test_apply(expr, outer, inner)
    }

    #[test]
    fn split_expr_no_pushdown() {
        // like is fallible, thus not pushed. lit() does not reference root()
        let expr = like(root(), lit(1u64));
        let (outer, inner) =
            split_expression_for_pushdown(expr.clone(), &DType::Utf8(true.into())).unwrap();
        assert_eq!(outer, expr);
        assert_eq!(inner, None);
    }
}
