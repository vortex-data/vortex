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
    use vortex_proto::expr as pb;
    use vortex_session::VortexSession;

    use crate::aggregate_fn::AggregateFnRef;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::session::AggregateFnSession;
    use crate::aggregate_fn::session::AggregateFnSessionExt;

    #[test]
    fn aggregate_fn_serde() {
        let session = VortexSession::empty().with::<AggregateFnSession>();
        session.aggregate_fns().register(Sum);

        let agg_fn = Sum.bind(EmptyOptions);

        let serialized = agg_fn.serialize_proto().unwrap();
        let buf = serialized.encode_to_vec();
        let deserialized_proto = pb::AggregateFn::decode(buf.as_slice()).unwrap();
        let deserialized = AggregateFnRef::from_proto(&deserialized_proto, &session).unwrap();

        assert_eq!(deserialized, agg_fn);
    }
}
