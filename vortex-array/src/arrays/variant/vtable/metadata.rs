// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Oneof;
use vortex_proto::dtype as pb;

#[derive(Clone, prost::Message)]
pub(super) struct VariantMetadataProto {
    #[prost(oneof = "Shredded", tags = "1, 2")]
    pub(super) shredded: Option<Shredded>,
}

/// Serialized reference to a derived `shredded` child.
///
/// `slot_name` is local to `source_encoding_id`; it is not a global child name.
#[derive(Clone, prost::Message)]
pub(super) struct DerivedSlotProto {
    #[prost(string, tag = "1")]
    pub(super) source_encoding_id: String,
    #[prost(string, tag = "2")]
    pub(super) slot_name: String,
}

#[derive(Clone, Oneof)]
pub(super) enum Shredded {
    #[prost(message, tag = "1")]
    InlineDtype(pb::DType),
    #[prost(message, tag = "2")]
    DerivedSlot(DerivedSlotProto),
}
