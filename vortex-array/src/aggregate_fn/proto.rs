// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arcref::ArcRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::session::AggregateFnSessionExt;

impl AggregateFnRef {
    /// Serialize this aggregate function to its protobuf representation.
    ///
    /// Note: the serialization format is not stable and may change between versions.
    pub fn serialize_proto(&self) -> VortexResult<pb::AggregateFn> {
        let metadata = self
            .options()
            .serialize()?
            .ok_or_else(|| vortex_err!("Aggregate function '{}' is not serializable", self.id()))?;

        Ok(pb::AggregateFn {
            id: self.id().to_string(),
            metadata: Some(metadata),
        })
    }

    /// Deserialize an aggregate function from its protobuf representation.
    ///
    /// Looks up the aggregate function plugin by ID in the session's registry
    /// and delegates deserialization to it.
    ///
    /// Note: the serialization format is not stable and may change between versions.
    pub fn from_proto(proto: &pb::AggregateFn, session: &VortexSession) -> VortexResult<Self> {
        let agg_fn_id: AggregateFnId = ArcRef::new_arc(Arc::from(proto.id.as_str()));
        let plugin = session
            .aggregate_fns()
            .registry()
            .find(&agg_fn_id)
            .ok_or_else(|| vortex_err!("unknown aggregate function id: {}", proto.id))?;
        let agg_fn = plugin.deserialize(proto.metadata(), session)?;

        if agg_fn.id() != agg_fn_id {
            vortex_bail!(
                "Aggregate function ID mismatch: expected {}, got {}",
                agg_fn_id,
                agg_fn.id()
            );
        }

        Ok(agg_fn)
    }
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_error::VortexResult;
    use vortex_error::vortex_panic;
    use vortex_proto::expr as pb;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::Columnar;
    use crate::ExecutionCtx;
    use crate::aggregate_fn::AggregateFnId;
    use crate::aggregate_fn::AggregateFnRef;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::session::AggregateFnSession;
    use crate::aggregate_fn::session::AggregateFnSessionExt;
    use crate::dtype::DType;
    use crate::scalar::Scalar;

    /// A minimal serializable aggregate function used solely to exercise the serde round-trip.
    #[derive(Clone, Debug)]
    struct TestAgg;

    impl AggregateFnVTable for TestAgg {
        type Options = EmptyOptions;
        type Partial = ();

        fn id(&self) -> AggregateFnId {
            AggregateFnId::new_ref("vortex.test.proto")
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
            _options: &Self::Options,
            _input_dtype: &DType,
        ) -> VortexResult<Self::Partial> {
            Ok(())
        }

        fn combine_partials(
            &self,
            _partial: &mut Self::Partial,
            _other: Scalar,
        ) -> VortexResult<()> {
            Ok(())
        }

        fn to_scalar(&self, _partial: &Self::Partial) -> VortexResult<Scalar> {
            vortex_panic!("TestAgg is for serde tests only");
        }

        fn reset(&self, _partial: &mut Self::Partial) {}

        fn is_saturated(&self, _partial: &Self::Partial) -> bool {
            true
        }

        fn accumulate(
            &self,
            _state: &mut Self::Partial,
            _batch: &Columnar,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<()> {
            Ok(())
        }

        fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
            Ok(partials)
        }
    }

    #[test]
    fn aggregate_fn_serde() {
        let session = VortexSession::empty().with::<AggregateFnSession>();
        session.aggregate_fns().register(TestAgg);

        let agg_fn = TestAgg.bind(EmptyOptions);

        let serialized = agg_fn.serialize_proto().unwrap();
        let buf = serialized.encode_to_vec();
        let deserialized_proto = pb::AggregateFn::decode(buf.as_slice()).unwrap();
        let deserialized = AggregateFnRef::from_proto(&deserialized_proto, &session).unwrap();

        assert_eq!(deserialized, agg_fn);
    }
}
