// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::new_foreign_aggregate_fn;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::proto::expr as pb;

impl AggregateFnRef {
    /// Serialize this aggregate function's options, erroring if it is not serializable.
    ///
    /// The bytes are only meaningful alongside [`id`](Self::id); [`from_parts`](Self::from_parts)
    /// is the inverse.
    pub fn serialize_options(&self) -> VortexResult<Vec<u8>> {
        self.options()
            .serialize()?
            .ok_or_else(|| vortex_err!("Aggregate function '{}' is not serializable", self.id()))
    }

    /// Resolve an aggregate function from its ID and serialized options.
    ///
    /// Returns `Ok(None)` when the session has no plugin registered for `id`; the caller decides
    /// whether that is an error, a reason to skip the aggregate, or a foreign placeholder. Errors
    /// if the plugin rejects `options`.
    pub fn from_parts(
        id: AggregateFnId,
        options: &[u8],
        session: &VortexSession,
    ) -> VortexResult<Option<Self>> {
        let Some(plugin) = session.aggregate_fns().find_plugin(&id) else {
            return Ok(None);
        };

        let agg_fn = plugin.deserialize(options, session)?;
        vortex_ensure_eq!(agg_fn.id(), id, "Aggregate function ID mismatch");
        Ok(Some(agg_fn))
    }

    /// Serialize this aggregate function to its protobuf representation.
    ///
    /// The format is stable: it is persisted in zoned layout metadata. An aggregate function's
    /// options encoding is part of that contract, so changing it requires a new ID.
    pub fn serialize_proto(&self) -> VortexResult<pb::AggregateFn> {
        Ok(pb::AggregateFn {
            id: self.id().to_string(),
            metadata: Some(self.serialize_options()?),
        })
    }

    /// Deserialize an aggregate function from its protobuf representation.
    ///
    /// Absent `metadata` is read as empty options. An ID the session does not know deserializes
    /// as a foreign placeholder if the session [allows unknown plugins], and errors otherwise.
    ///
    /// [allows unknown plugins]: VortexSession::allows_unknown
    pub fn from_proto(proto: &pb::AggregateFn, session: &VortexSession) -> VortexResult<Self> {
        #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
        let agg_fn_id = AggregateFnId::new(proto.id.as_str());
        if let Some(agg_fn) = Self::from_parts(agg_fn_id, proto.metadata(), session)? {
            return Ok(agg_fn);
        }
        if session.allows_unknown() {
            return Ok(new_foreign_aggregate_fn(
                agg_fn_id,
                proto.metadata().to_vec(),
            ));
        }
        vortex_bail!("unknown aggregate function id: {}", proto.id)
    }
}
