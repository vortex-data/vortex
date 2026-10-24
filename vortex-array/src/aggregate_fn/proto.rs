// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::new_foreign_aggregate_fn;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::proto::expr as pb;

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
        #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
        let agg_fn_id: AggregateFnId = AggregateFnId::new(proto.id.as_str());
        let agg_fn = if let Some(plugin) = session.aggregate_fns().find_plugin(&agg_fn_id) {
            plugin.deserialize(proto.metadata(), session)?
        } else if session.allows_unknown() {
            new_foreign_aggregate_fn(agg_fn_id, proto.metadata().to_vec())
        } else {
            return Err(vortex_err!("unknown aggregate function id: {}", proto.id));
        };

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
