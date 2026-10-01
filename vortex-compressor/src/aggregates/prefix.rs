// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Distinct physical fingerprints of canonical variable-length views.
//!
//! This private aggregate preserves the compressor's existing string-cardinality heuristic. It
//! counts the low 64 bits of every supplied view, including invalid slots. Cache entries therefore
//! belong to the exact canonical representation and cannot propagate through slices or compaction.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateArgs;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::EmptyOptions as ScalarOptions;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::list_length::ListLength;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_set::HashSet;

/// Count distinct low-64-bit fingerprints of every physical `VarBinView` slot.
///
/// This requires canonical [`VarBinViewArray`](vortex_array::arrays::VarBinViewArray) input and includes invalid views. Its result is an
/// exact fingerprint count, used only as an approximate compression heuristic for logical values.
/// It is not persisted or included among default zone aggregates.
#[derive(Clone, Debug)]
pub(crate) struct VarBinViewPrefixDistinct;

impl AggregateFnVTable for VarBinViewPrefixDistinct {
    type Options = EmptyOptions;
    type Partial = HashSet<u64>;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.compressor.varbinview_prefix_distinct");
        *ID
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        matches!(input_dtype, DType::Utf8(_) | DType::Binary(_))
            .then_some(DType::Primitive(PType::U32, NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)?;
        Some(DType::List(
            Arc::new(DType::Primitive(PType::U64, NonNullable)),
            NonNullable,
        ))
    }

    fn empty_partial(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
    ) -> VortexResult<Self::Partial> {
        Ok(HashSet::new())
    }

    fn partial_from_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        scalar: Scalar,
    ) -> VortexResult<Self::Partial> {
        scalar
            .as_list()
            .elements()
            .ok_or_else(|| vortex_err!("Physical prefix partial must be a non-null list"))?
            .into_iter()
            .map(|element| {
                element
                    .as_primitive()
                    .typed_value::<u64>()
                    .ok_or_else(|| vortex_err!("Physical prefix must be non-null"))
            })
            .collect()
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        mut first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        first.extend(second);
        Ok(first)
    }

    fn to_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        // Scalar lists use u32 lengths. Preserve the existing checked cardinality limit.
        _ = u32::try_from(partial.len())?;
        Ok(Scalar::list(
            DType::Primitive(PType::U64, NonNullable),
            partial
                .iter()
                .map(|&prefix| Scalar::primitive(prefix, NonNullable))
                .collect(),
            NonNullable,
        ))
    }

    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> bool {
        false
    }

    fn try_accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let Some(array) = batch.as_opt::<VarBinView>() else {
            vortex_bail!(
                "Physical prefix distinct requires canonical VarBinView input, got {}",
                batch.encoding_id()
            );
        };

        accumulate_views(state, array.views());
        Ok(true)
    }

    fn accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &Columnar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Columnar::Canonical(Canonical::VarBinView(array)) = batch else {
            vortex_bail!("Physical prefix distinct requires canonical VarBinView input");
        };

        accumulate_views(state, array.views());
        Ok(())
    }

    fn finalize(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        ScalarFnArray::try_new(ListLength.bind(ScalarOptions), vec![partials])?
            .into_array()
            .cast(args.return_dtype.clone())
    }

    fn finalize_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(
            u32::try_from(partial.len())?,
            NonNullable,
        ))
    }
}

/// Add every view without validity filtering or rebuilding its physical representation.
fn accumulate_views(state: &mut HashSet<u64>, views: &[BinaryView]) {
    if state.is_empty() {
        state.reserve(views.len() / 2);
    }

    for view in views {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "The fingerprint is the low 64 bits"
        )]
        let prefix = view.as_u128() as u64;
        state.insert(prefix);
    }
}
