// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [ScalarFnVTable] implementation for Bloom-filter zone pruning.
//!
//! The bloom filter has two main access points,
//! `insert` and `contains`. Insert happens during writes,
//! while for pruning `contains` exposes if a zone contains
//! a value or not.
//!
//! This file also contains a simple equality [StatsRewriteRule] that
//! uses `falsify` for pruning.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::arrays::varbinview::VarBinViewArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::not;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::stats::expr::bound::stat as bound_stat;
use vortex_array::stats::rewrite::StatsRewriteCtx;
use vortex_array::stats::rewrite::StatsRewriteRule;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::BloomFilter;
use super::BloomOptions;
use crate::layouts::zoned::aggregates::bloom_filter::BloomPartial;
use crate::layouts::zoned::aggregates::bloom_filter::is_bloom_valid_dtype;

#[derive(Clone, Debug)]
pub(in crate::layouts::zoned) struct BloomContains;

impl ScalarFnVTable for BloomContains {
    type Options = BloomOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.bloom_filter.sbbf.contains");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        BloomFilter.serialize(options)
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        BloomFilter.deserialize(metadata, session)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("filter"),
            1 => ChildName::from("needle"),
            _ => unreachable!("bloom_contains has exactly two children"),
        }
    }

    // The bloom filter supports only a subset of scalars,
    // so this function will return an error if the type is still unsupported.
    fn return_dtype(&self, _options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            matches!(args[0], DType::Binary(_)),
            "bloom filter must be Binary"
        );

        vortex_ensure!(
            is_bloom_valid_dtype(&args[1]),
            "bloom filter needle value type is unsupported"
        );

        Ok(DType::Bool(args[0].nullability() | args[1].nullability()))
    }

    // Execution can only fail for one of the following three reasons:
    // 1. The needle is not a constant/literal.
    // 2. The mask is invalid.
    // 3. A `BloomFilter` fails to deserialize.
    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let filters = args.get(0)?.execute::<VarBinViewArray>(ctx)?;

        // For now, a bloom filter can execute only when the needle is a literal,
        // so anything other than a constant here is unexpected.
        let needle_array = args.get(1)?;
        let needle = needle_array
            .as_constant()
            .ok_or_else(|| vortex_err!("bloom filter needle must be a constant"))?;

        // The bloom filter doesn't consider nulls as values
        // whose presence it should detect. During writes, the filter
        // skips null values, so the answer for null values is
        // already known.
        //
        // This case is also checked by the only available rule,
        // so it shouldn't happen. Anyways, if it does, rule out early.
        if needle.is_null() {
            return Ok(ConstantArray::new(
                Scalar::null(DType::Bool(Nullability::Nullable)),
                args.row_count(),
            )
            .into_array());
        }

        let validity = filters
            .varbinview_validity()
            // Match the expected nullability defined by `BloomContains::return_dtype`.
            .union_nullability(needle.dtype().nullability());

        let valid = validity.execute_mask(filters.len(), ctx)?;

        // The bloom filter is a probabilistic structure,
        // and false positives are possible.
        //
        // Choosing `new_unset` over `new_set` because I expect
        // that most zones will be pruned.
        let mut possibles = BitBufferMut::new_unset(filters.len());

        // Utility function to process views in both validity cases.
        //
        // Very similar approach as on how the bloom filter processes
        // binary views.
        let process_view = |view: &BinaryView, buffers: &[&Buffer<u8>]| -> VortexResult<bool> {
            // The max inlined size is very small (currently 12B), and a bloom filter will always
            // be at least 32B, so an inlined bloom filter should be impossible.
            //
            // So, the following if is a candidate for removal with a debug assertion and a test
            // over `BinaryView::MAX_INLINED_SIZE` to spot changes. I'm keeping it this way just to
            // keep this function "robust" for now.
            let bytes = if view.is_inlined() {
                view.as_inlined().value()
            } else {
                let view_ref = view.as_view();
                &buffers[view_ref.buffer_index as usize][view_ref.as_range()]
            };

            // One possible performance optimization here would be to hash the needle and
            // calculate the target block before deserializing the whole Bloom filter.
            // Then we could calculate the target block's offset, deserialize those 32 bytes,
            // and perform the check on that small subset.
            let partial = BloomPartial::deserialize(bytes)?;

            // [`BloomPartial`] validates that the byte length is valid, but it doesn't know
            // what the initial options were. This assertion compares the partial block count
            // with the options' block count as an extra guard.
            // Still, this could conflict with a future `BloomPartial` optimization (folding),
            // where the partial's block count can be smaller than the initial options.
            vortex_ensure_eq!(
                // Bloom filter length is never larger than a `u32`. This is intentional
                // and a property of the implementation.
                u32::try_from(partial.len()).vortex_expect("valid u32 size"),
                options.blocks_count().get(),
                "expected equal blocks count"
            );
            partial.contains_scalar(&needle)
        };

        match valid {
            vortex_mask::Mask::AllTrue(_) => {
                let buffers = filters
                    .data_buffers()
                    .iter()
                    .map(|b| b.as_host())
                    .collect::<Vec<_>>();

                for (idx, view) in filters.views().iter().enumerate() {
                    possibles.set_to(idx, process_view(view, &buffers)?);
                }
            }
            // It is ok to skip `AllFalse` because every bit in `possibles` is already `false`.
            vortex_mask::Mask::AllFalse(_) => {}
            vortex_mask::Mask::Values(mask_values) => {
                let buffers = filters
                    .data_buffers()
                    .iter()
                    .map(|b| b.as_host())
                    .collect::<Vec<_>>();

                for (idx, (view, valid)) in filters
                    .views()
                    .iter()
                    .zip(mask_values.bit_buffer())
                    .enumerate()
                {
                    // Invalid zones are skipped, `possibles` has all bits set to `false`
                    // during init.
                    if valid {
                        possibles.set_to(idx, process_view(view, &buffers)?);
                    }
                }
            }
        }

        Ok(BoolArray::new(possibles.freeze(), validity).into_array())
    }
}

/// Equality rewrite that turns a Bloom miss into a zone falsifier.
#[derive(Clone, Debug)]
pub(in crate::layouts::zoned) struct BloomEqRewrite;

impl StatsRewriteRule for BloomEqRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Binary.id()
    }

    /// Returns a falsifier for equality expression between the root column
    /// and a non-null literal.
    ///
    /// For example, `root == 42` and `42 == root` can be falsified,
    /// while `root == <expr>` or `root == NULL` are inconclusive.
    fn falsify(
        &self,
        expr: &BoundExpression,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<BoundExpression>> {
        let Some(aggregate_fn) = ctx
            .aggregate_fns()
            .iter()
            .find(|aggregate_fn| aggregate_fn.is::<BloomFilter>())
        else {
            return Ok(None);
        };

        let options = aggregate_fn.as_::<BloomFilter>().clone();

        if *expr.as_::<Binary>() != Operator::Eq {
            return Ok(None);
        }

        // Case 1: root == <lit>
        let (column, literal) = if expr.child(0).is_root() && expr.child(1).is::<Literal>() {
            (expr.child(0), expr.child(1))
        // Case 2: <lit> == root
        } else if expr.child(1).is_root() && expr.child(0).is::<Literal>() {
            (expr.child(1), expr.child(0))
        } else {
            return Ok(None);
        };

        // Nulls are not stored in Bloom filters, so it is not possible to determine
        // if it is present or not, so the answer is inconclusive.
        if !is_bloom_valid_dtype(&ctx.return_dtype(column)?) || literal.as_::<Literal>().is_null() {
            return Ok(None);
        }

        let filter = bound_stat(column.clone(), BloomFilter.bind(options.clone()));
        let contains = BloomContains.try_new_bound_expr(options, [filter, literal.clone()])?;

        Ok(Some(not(contains)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ScalarFnArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::bound::eq;
    use vortex_array::expr::bound::lit;
    use vortex_array::expr::bound::root;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::scalar_fn::ScalarFnVTableExt;
    use vortex_array::scalar_fn::VecExecutionArgs;
    use vortex_array::scalar_fn::session::ScalarFnSessionExt;
    use vortex_array::stats::StatsSessionExt;
    use vortex_array::stats::rewrite::StatsRewriteCtx;
    use vortex_array::stats::rewrite::StatsRewriteRule;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::BloomContains;
    use super::BloomOptions;
    use crate::layouts::zoned::aggregates::bloom_filter::BloomFilter;
    use crate::layouts::zoned::aggregates::bloom_filter::BloomPartial;
    use crate::layouts::zoned::aggregates::bloom_filter::scalar_fn::BloomEqRewrite;
    use crate::layouts::zoned::zone_map::ZoneMap;

    fn register(session: &VortexSession) {
        session.aggregate_fns().register(BloomFilter);
        session.scalar_fns().register(BloomContains);
        session.stats().register_rewrite(BloomEqRewrite);
    }

    fn serialized_filter(options: &BloomOptions, values: &[Scalar]) -> VortexResult<Vec<u8>> {
        let mut partial = BloomPartial::from(options);
        for value in values {
            partial.insert_scalar(value)?;
        }
        Ok(partial.serialize())
    }

    #[test]
    fn bloom_rule_prunes_ok() -> VortexResult<()> {
        let options = BloomOptions::default();
        let aggregate_fn = BloomFilter.bind(options.clone());
        let session = array_session();
        register(&session);

        let filters = VarBinViewArray::from_iter_nullable_bin([
            Some(serialized_filter(
                &options,
                &[Scalar::primitive(42_i64, Nullability::NonNullable)],
            )?),
            Some(serialized_filter(&options, &[])?),
        ])
        .into_array();

        let zone_map = ZoneMap::try_new(
            DType::Primitive(PType::I64, Nullability::NonNullable),
            StructArray::from_fields(&[(aggregate_fn.to_string(), filters)])?,
            Arc::new([aggregate_fn.clone()]),
            1,
            2,
        )?;

        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let predicate = eq(root(dtype), lit(42i64));
        let available_aggregate_fns = [aggregate_fn];
        let rewrite_ctx =
            StatsRewriteCtx::new(&session).with_aggregate_fns(&available_aggregate_fns);
        let proof = rewrite_ctx
            .falsify(&predicate)?
            .ok_or_else(|| vortex_error::vortex_err!("equality should have a bloom falsifier"))?;

        let mut ctx = session.create_execution_ctx();
        assert_arrays_eq!(
            zone_map.prune(&proof, &session)?.into_array(),
            BoolArray::from_iter([false, true]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn bloom_rule_is_inconclusive_for_nulls() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::Nullable);
        let session = array_session();
        let available_aggregate_fns = [BloomFilter.bind(BloomOptions::default())];
        let ctx = StatsRewriteCtx::new(&session).with_aggregate_fns(&available_aggregate_fns);
        let rule = BloomEqRewrite;

        let non_literal = eq(root(dtype.clone()), root(dtype.clone()));
        assert!(rule.falsify(&non_literal, &ctx)?.is_none());

        let null_literal = eq(root(dtype.clone()), lit(Scalar::null(dtype)));
        assert!(rule.falsify(&null_literal, &ctx)?.is_none());
        Ok(())
    }

    #[test]
    fn missing_bloom_stat_stays_inconclusive() -> VortexResult<()> {
        let options = BloomOptions::default();
        let aggregate_fn = BloomFilter.bind(options);
        let session = array_session();
        register(&session);

        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let predicate = eq(root(dtype.clone()), lit(42i64));
        let available_aggregate_fns = [aggregate_fn];
        let rewrite_ctx =
            StatsRewriteCtx::new(&session).with_aggregate_fns(&available_aggregate_fns);
        let proof = rewrite_ctx
            .falsify(&predicate)?
            .ok_or_else(|| vortex_error::vortex_err!("equality should have a bloom falsifier"))?;

        let zone_map = ZoneMap::try_new(
            dtype,
            StructArray::try_new(Vec::<&str>::new().into(), vec![], 2, Validity::NonNullable)?,
            Arc::new([]),
            8,
            16,
        )?;

        assert!(zone_map.prune(&proof, &session)?.all_false());
        Ok(())
    }

    #[test]
    fn malformed_filter_is_an_error() {
        let options = BloomOptions::default();
        let filters =
            VarBinViewArray::from_iter_bin([b"not a bloom filter".as_slice()]).into_array();
        let needle = ConstantArray::new(42i64, 1).into_array();
        let args = VecExecutionArgs::new(vec![filters, needle], 1);
        let mut ctx = array_session().create_execution_ctx();

        let error = BloomContains
            .execute(&options, &args, &mut ctx)
            .expect_err("the Bloom filter byte length should be invalid");
        assert!(
            error
                .to_string()
                .contains("invalid bloom filter byte length"),
            "unexpected error: {error}"
        );
    }

    // The following test is about having one set of options with an expected size (the default)
    // and a zone with a smaller size (unexpected). This scenario shouldn't be possible
    // with the current implementation, but it might become possible in the future
    // with optimizations (folding).
    #[test]
    fn mismatched_filter_block_count_is_an_error() {
        let options = BloomOptions::default();
        // Smaller filter and empty
        let filters = VarBinViewArray::from_iter_bin([vec![0; 32]]).into_array();
        let needle = ConstantArray::new(42i64, 1).into_array();
        let args = VecExecutionArgs::new(vec![filters, needle], 1);
        let mut ctx = array_session().create_execution_ctx();

        let error = BloomContains
            .execute(&options, &args, &mut ctx)
            .expect_err("the Bloom filter block count should not match the options");
        assert!(
            error.to_string().contains("expected equal blocks count"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn null_needle_is_inconclusive() -> VortexResult<()> {
        let options = BloomOptions::default();
        let filters =
            VarBinViewArray::from_iter_bin([serialized_filter(&options, &[])?]).into_array();
        let needle = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
            1,
        )
        .into_array();
        let args = VecExecutionArgs::new(vec![filters, needle], 1);
        let mut ctx = array_session().create_execution_ctx();

        let actual = BloomContains.execute(&options, &args, &mut ctx)?;
        assert_arrays_eq!(actual, BoolArray::from_iter([None::<bool>]), &mut ctx);
        Ok(())
    }

    #[test]
    fn check_nullability_validity_is_correct() -> VortexResult<()> {
        let options = BloomOptions::default();
        let filters = VarBinViewArray::from_iter_bin([serialized_filter(
            &options,
            &[Scalar::primitive(42_i64, Nullability::NonNullable)],
        )?])
        .into_array();
        let needle =
            ConstantArray::new(Scalar::primitive(42_i64, Nullability::Nullable), 1).into_array();
        let mut ctx = array_session().create_execution_ctx();

        let planned = ScalarFnArray::try_new(BloomContains.bind(options), vec![filters, needle])?;
        let expected_dtype = planned.dtype().clone();

        // If the validity types are wrong (e.g. `bool? != bool`), `execute` will raise an error.
        // The asserts that follow are just extra guards.
        let actual = planned.into_array().execute::<BoolArray>(&mut ctx)?;

        assert_eq!(actual.dtype(), &expected_dtype);
        assert_arrays_eq!(actual, BoolArray::from_iter([Some(true)]), &mut ctx);
        Ok(())
    }

    #[test]
    fn invalid_needle_is_an_error() -> VortexResult<()> {
        let options = BloomOptions::default();
        let filters =
            VarBinViewArray::from_iter_bin([serialized_filter(&options, &[])?]).into_array();

        // Decimlas are currently not supported.
        let needle = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I128(42),
                DecimalDType::new(10, 2),
                Nullability::NonNullable,
            ),
            1,
        )
        .into_array();
        let args = VecExecutionArgs::new(vec![filters, needle], 1);
        let mut ctx = array_session().create_execution_ctx();

        let error = BloomContains
            .execute(&options, &args, &mut ctx)
            .expect_err("decimal needles should be unsupported");
        assert!(
            error
                .to_string()
                .contains("Unsupported scalar type for bloom filter"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    // The following tests are just for validity cases.
    // Nothing in particular but just to test all the branches, even if a case is rare.
    #[test]
    fn validity_all_false() -> VortexResult<()> {
        let options = BloomOptions::default();

        // Two invalid filters
        let filters = VarBinViewArray::from_iter_nullable_bin([None::<Vec<u8>>, None]).into_array();

        let needle = ConstantArray::new(42i64, 2).into_array();
        let args = VecExecutionArgs::new(vec![filters, needle], 2);
        let mut ctx = array_session().create_execution_ctx();

        let actual = BloomContains.execute(&options, &args, &mut ctx)?;
        assert_arrays_eq!(actual, BoolArray::from_iter([None::<bool>, None]), &mut ctx);
        Ok(())
    }

    #[test]
    fn validity_mixed() -> VortexResult<()> {
        let options = BloomOptions::default();

        // [Valid, Invalid, Valid] filters
        let filters = VarBinViewArray::from_iter_nullable_bin([
            Some(serialized_filter(
                &options,
                &[Scalar::primitive(42_i64, Nullability::NonNullable)],
            )?),
            None,
            Some(serialized_filter(&options, &[])?),
        ])
        .into_array();
        let needle = ConstantArray::new(42i64, 3).into_array();
        let args = VecExecutionArgs::new(vec![filters, needle], 3);
        let mut ctx = array_session().create_execution_ctx();

        let actual = BloomContains.execute(&options, &args, &mut ctx)?;
        assert_arrays_eq!(
            actual,
            BoolArray::from_iter([Some(true), None, Some(false)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn validity_all_true() -> VortexResult<()> {
        let options = BloomOptions::default();

        // Two valid filters, but the second one is empty
        let filters = VarBinViewArray::from_iter_nullable_bin([
            Some(serialized_filter(
                &options,
                &[Scalar::primitive(42_i64, Nullability::NonNullable)],
            )?),
            Some(serialized_filter(&options, &[])?),
        ])
        .into_array();

        let needle = ConstantArray::new(42i64, 2).into_array();
        let args = VecExecutionArgs::new(vec![filters, needle], 2);
        let mut ctx = array_session().create_execution_ctx();

        let actual = BloomContains.execute(&options, &args, &mut ctx)?;
        assert_arrays_eq!(
            actual,
            BoolArray::from_iter([Some(true), Some(false)]),
            &mut ctx
        );
        Ok(())
    }
}
