// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use prost::Message;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateArgs;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::dtype::DType;
use vortex_array::proto::expr as pb;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::sum::Sum;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<AggregateFnSession>();
    session.allow_unknown();
    session
});

/// A minimal serializable aggregate function used solely to exercise the serde round-trip.
#[derive(Clone, Debug)]
struct TestAgg;

impl AggregateFnVTable for TestAgg {
    type Options = EmptyOptions;
    type Partial = ();

    #[expect(clippy::disallowed_methods, reason = "test-only id")]
    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.test.proto")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        Some(input_dtype.clone())
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
    ) -> VortexResult<Self::Partial> {
        Ok(())
    }

    fn partial_from_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        Ok(())
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _first: Self::Partial,
        _second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        Ok(())
    }

    fn to_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        vortex_panic!("TestAgg is for serde tests only");
    }

    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> bool {
        true
    }

    fn accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _state: &mut Self::Partial,
        _batch: &Columnar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        Ok(())
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        vortex_panic!("TestAgg is for serde tests only");
    }
}

#[test]
fn aggregate_fn_serde() {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session.aggregate_fns().register(TestAgg);

    let agg_fn = TestAgg.bind(EmptyOptions);

    let serialized = agg_fn.serialize_proto().unwrap();
    let buf = serialized.encode_to_vec();
    let deserialized_proto = pb::AggregateFn::decode(buf.as_slice()).unwrap();
    let deserialized = AggregateFnRef::from_proto(&deserialized_proto, &session).unwrap();

    assert_eq!(deserialized, agg_fn);
}

/// The `skip_nans` option must survive a protobuf serialize/deserialize round-trip for the
/// numeric aggregates, including the non-default NaN-including configuration.
#[rstest]
#[case(NumericalAggregateOpts::skip_nans())]
#[case(NumericalAggregateOpts::include_nans())]
fn numeric_aggregate_options_round_trip(
    #[case] options: NumericalAggregateOpts,
) -> VortexResult<()> {
    let session = VortexSession::empty();
    crate::initialize(&session);
    let agg_fn = Sum.bind(options);

    let proto = agg_fn.serialize_proto()?;
    let buf = proto.encode_to_vec();
    let decoded = pb::AggregateFn::decode(buf.as_slice())?;
    let round_tripped = AggregateFnRef::from_proto(&decoded, &session)?;

    assert_eq!(round_tripped, agg_fn);
    Ok(())
}

#[test]
fn unknown_aggregate_fn_id_allow_unknown() {
    let proto = pb::AggregateFn {
        id: "vortex.test.foreign_aggregate".to_string(),
        metadata: Some(vec![7, 8, 9]),
    };

    let agg_fn = AggregateFnRef::from_proto(&proto, &SESSION).unwrap();
    assert_eq!(agg_fn.id().as_ref(), "vortex.test.foreign_aggregate");

    let roundtrip = agg_fn.serialize_proto().unwrap();
    assert_eq!(roundtrip.id, proto.id);
    assert_eq!(roundtrip.metadata(), proto.metadata());
}
