// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Wire format for the serializable subset of authored expressions.

use prost::Message;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::proto::expr as pb;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::between::Between;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::byte_length::ByteLength;
use vortex_array::scalar_fn::fns::case_when::CaseWhen;
use vortex_array::scalar_fn::fns::case_when::CaseWhenOptions;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::ext_storage::ExtStorage;
use vortex_array::scalar_fn::fns::fill_null::FillNull;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::scalar_fn::fns::list_contains::ListContains;
use vortex_array::scalar_fn::fns::list_length::ListLength;
use vortex_array::scalar_fn::fns::list_sum::ListSum;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::mask::Mask;
use vortex_array::scalar_fn::fns::merge::DuplicateHandling;
use vortex_array::scalar_fn::fns::merge::Merge;
use vortex_array::scalar_fn::fns::not::Not;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_array::scalar_fn::fns::pack::PackOptions;
use vortex_array::scalar_fn::fns::select::FieldSelection;
use vortex_array::scalar_fn::fns::select::Select;
use vortex_array::scalar_fn::fns::variant_get::VariantGet;
use vortex_array::scalar_fn::fns::variant_get::VariantGetOptions;
use vortex_array::scalar_fn::fns::zip::Zip;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::Expression;

impl Expression {
    /// Serialize an authored expression using the existing expression wire format.
    pub fn serialize_proto(&self) -> VortexResult<pb::Expr> {
        let Self::Call {
            id,
            options,
            children,
        } = self
        else {
            return Ok(pb::Expr {
                id: "vortex.root".to_string(),
                children: vec![],
                metadata: Some(vec![]),
            });
        };
        let (wire_id, metadata) = match id.as_ref() {
            "column" => (
                GetItem.id(),
                GetItem.serialize(option::<FieldName>(id, options)?)?,
            ),
            "literal" => (
                Literal.id(),
                Literal.serialize(option::<Scalar>(id, options)?)?,
            ),
            "binary" => (
                Binary.id(),
                Binary.serialize(option::<Operator>(id, options)?)?,
            ),
            "cast" => (Cast.id(), Cast.serialize(option::<DType>(id, options)?)?),
            "not" => (Not.id(), Not.serialize(&EmptyOptions)?),
            "is_null" => (IsNull.id(), IsNull.serialize(&EmptyOptions)?),
            "is_not_null" => (IsNotNull.id(), IsNotNull.serialize(&EmptyOptions)?),
            "select" => (
                Select.id(),
                Select.serialize(option::<FieldSelection>(id, options)?)?,
            ),
            "pack" => (
                Pack.id(),
                Pack.serialize(option::<PackOptions>(id, options)?)?,
            ),
            "between" => (
                Between.id(),
                Between.serialize(option::<BetweenOptions>(id, options)?)?,
            ),
            "like" => (
                Like.id(),
                Like.serialize(option::<LikeOptions>(id, options)?)?,
            ),
            "fill_null" => (FillNull.id(), FillNull.serialize(&EmptyOptions)?),
            "byte_length" => (ByteLength.id(), ByteLength.serialize(&EmptyOptions)?),
            "merge" => (
                Merge.id(),
                Merge.serialize(option::<DuplicateHandling>(id, options)?)?,
            ),
            "list_contains" => (ListContains.id(), ListContains.serialize(&EmptyOptions)?),
            "list_length" => (ListLength.id(), ListLength.serialize(&EmptyOptions)?),
            "list_sum" => (
                ListSum.id(),
                ListSum.serialize(option::<NumericalAggregateOpts>(id, options)?)?,
            ),
            "case_when" => (
                CaseWhen.id(),
                CaseWhen.serialize(option::<CaseWhenOptions>(id, options)?)?,
            ),
            "zip" => (Zip.id(), Zip.serialize(&EmptyOptions)?),
            "mask" => (Mask.id(), Mask.serialize(&EmptyOptions)?),
            "ext_storage" => (ExtStorage.id(), ExtStorage.serialize(&EmptyOptions)?),
            "variant_get" => (
                VariantGet.id(),
                VariantGet.serialize(option::<VariantGetOptions>(id, options)?)?,
            ),
            _ => {
                return Err(vortex_err!(
                    "authored function {id} is not supported by expression serialization"
                ));
            }
        };
        let metadata =
            metadata.ok_or_else(|| vortex_err!("authored function {id} is not serializable"))?;
        let children = children
            .iter()
            .map(Self::serialize_proto)
            .collect::<VortexResult<_>>()?;
        Ok(pb::Expr {
            id: wire_id.to_string(),
            children,
            metadata: Some(metadata),
        })
    }

    /// Decode the serializable subset of authored expressions.
    pub fn from_proto(proto: &pb::Expr, session: &VortexSession) -> VortexResult<Self> {
        if proto.id == "vortex.root" {
            vortex_ensure!(
                proto.children.is_empty(),
                "root expression must have no children"
            );
            return Ok(Self::Root);
        }
        let children = proto
            .children
            .iter()
            .map(|child| Self::from_proto(child, session))
            .collect::<VortexResult<Vec<_>>>()?;
        let bytes = proto.metadata();
        let expression = match proto.id.as_str() {
            "vortex.get_item" => {
                Self::call("column", GetItem.deserialize(bytes, session)?, children)
            }
            "vortex.literal" => {
                Self::call("literal", Literal.deserialize(bytes, session)?, children)
            }
            "vortex.binary" => Self::call("binary", Binary.deserialize(bytes, session)?, children),
            "vortex.cast" => Self::call("cast", Cast.deserialize(bytes, session)?, children),
            "vortex.not" => Self::call("not", (), children),
            "vortex.is_null" => Self::call("is_null", (), children),
            "vortex.is_not_null" => Self::call("is_not_null", (), children),
            "vortex.select" => Self::call("select", Select.deserialize(bytes, session)?, children),
            "vortex.pack" => Self::call("pack", Pack.deserialize(bytes, session)?, children),
            "vortex.between" => {
                Self::call("between", Between.deserialize(bytes, session)?, children)
            }
            "vortex.like" => Self::call("like", Like.deserialize(bytes, session)?, children),
            "vortex.fill_null" => Self::call("fill_null", (), children),
            "vortex.byte_length" => Self::call("byte_length", (), children),
            "vortex.merge" => Self::call("merge", Merge.deserialize(bytes, session)?, children),
            "vortex.list.contains" => Self::call("list_contains", (), children),
            "vortex.list.length" => Self::call("list_length", (), children),
            "vortex.list.sum" => {
                Self::call("list_sum", ListSum.deserialize(bytes, session)?, children)
            }
            "vortex.zip" => Self::call("zip", (), children),
            "vortex.mask" => Self::call("mask", (), children),
            "vortex.ext.storage" => Self::call("ext_storage", (), children),
            "vortex.variant_get" => Self::call(
                "variant_get",
                VariantGet.deserialize(bytes, session)?,
                children,
            ),
            _ => {
                return Err(vortex_err!(
                    "authored function {} is not supported by expression serialization",
                    proto.id
                ));
            }
        };
        Ok(expression)
    }

    /// Return serialized bytes, suitable for language binding pickle support.
    pub fn to_bytes(&self) -> VortexResult<Vec<u8>> {
        Ok(self.serialize_proto()?.encode_to_vec())
    }
}

fn option<'a, T: 'static>(
    id: &str,
    options: &'a std::sync::Arc<dyn std::any::Any + Send + Sync>,
) -> VortexResult<&'a T> {
    options
        .downcast_ref::<T>()
        .ok_or_else(|| vortex_err!("invalid options for authored function {id}"))
}
