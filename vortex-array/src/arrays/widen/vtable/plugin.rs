// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Wire layouts of `vortex.widen` and `vortex.widen.v2` nodes.
//!
//! A node has one child and no buffers of its own. The enclosing format supplies the logical dtype
//! and length of the [`Widen`]. The child has the same length and nullability. A decimal child
//! also has the same scale.
//!
//! Readers require the first child and ignore any children after it, as well as any node buffers.
//! A later writer may append a child or buffer under this ID only if decoding without it yields the
//! same values, such as a statistic or an index. A child, buffer or metadata field that changes the
//! decoded values needs a new ID.
//!
//! A protobuf `oneof` supplies the rest of the child dtype. Tag 1 holds the integer [`PType`],
//! tag 2 holds the decimal precision as a `uint32`, and tag 3 holds the floating-point [`PType`].
//! `vortex.widen` is frozen for integer and decimal children (tags 1 and 2); floating-point
//! children (tag 3) require `vortex.widen.v2`. The child dtype must meet the requirements of
//! [`WidenArray::try_new`].

use prost::Message as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::ArrayVTable;
use crate::IntoArray;
use crate::arrays::Decimal;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::array::validate_dtypes;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::PType;
use crate::session::ArraySessionExt;

/// The wire ID for [`Widen`] arrays carrying floating-point children (`vortex.widen.v2`).
///
/// `vortex.widen` is frozen for integer and decimal children; floating-point children serialize
/// under this ID so older readers never encounter a floating-point metadata tag.
pub fn widen_v2_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.widen.v2");
    *ID
}

#[derive(Clone, prost::Message)]
struct WidenMetadata {
    #[prost(oneof = "ChildType", tags = "1, 2, 3")]
    child_type: Option<ChildType>,
}

#[derive(Clone, prost::Oneof)]
enum ChildType {
    #[prost(enumeration = "PType", tag = "1")]
    Integer(i32),
    #[prost(uint32, tag = "2")]
    DecimalPrecision(u32),
    #[prost(enumeration = "PType", tag = "3")]
    Float(i32),
}

/// Reads and writes serialized [`Widen`] arrays.
///
/// Writing reuses the child without executing its values or validity. When the child is a
/// [`Decimal`] array, this plugin writes `vortex.decimal` so older readers can read it. Integer
/// children and encoded decimals use `vortex.widen`, and floating-point children use
/// `vortex.widen.v2`.
#[derive(Clone, Debug)]
pub struct WidenPlugin;

impl ArrayPlugin for WidenPlugin {
    fn id(&self) -> ArrayId {
        ArrayVTable::id(&Widen)
    }

    fn read_ids(&self) -> Vec<ArrayId> {
        vec![self.id(), widen_v2_id()]
    }

    fn write_ids(&self) -> Vec<ArrayId> {
        // Older editions allow `vortex.decimal`, which can store the child's narrow buffer
        // under the parent's decimal dtype.
        vec![ArrayVTable::id(&Decimal), self.id(), widen_v2_id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let widen = array.as_opt::<Widen>().ok_or_else(|| {
            vortex_err!(
                "Widen serialization requires a Widen array, got {}",
                array.encoding_id()
            )
        })?;
        let values = widen.values();
        if values.is::<Decimal>() {
            // The wire node gets the buffer width and validity from the child. The enclosing
            // format supplies the parent dtype, and serialization keeps the parent's statistics.
            return session.array_serialize(values);
        }

        let (wire_id, child_type) = match values.dtype() {
            DType::Primitive(ptype, _) if ptype.is_int() => {
                (self.id(), ChildType::Integer(*ptype as i32))
            }
            DType::Primitive(ptype, _) if ptype.is_float() => {
                (widen_v2_id(), ChildType::Float(*ptype as i32))
            }
            DType::Decimal(dtype, _) => (
                self.id(),
                ChildType::DecimalPrecision(u32::from(dtype.precision())),
            ),
            dtype => {
                vortex_bail!(
                    "Widen serialization requires an integer, float, or decimal child, got {dtype}"
                )
            }
        };
        let metadata = WidenMetadata {
            child_type: Some(child_type),
        };

        Ok(Some(ArraySerialization::new(
            wire_id,
            metadata.encode_to_vec(),
            vec![],
            vec![values.clone()],
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let is_v1 = parts.serialized_id == self.id();
        let is_v2 = parts.serialized_id == widen_v2_id();
        vortex_ensure!(
            is_v1 || is_v2,
            "expected a Widen wire ID ({} or {}), got {}",
            self.id(),
            widen_v2_id(),
            parts.serialized_id
        );
        // NB: Node buffers and children after the first child are optional extensions, so they
        // are ignored. See the module documentation.
        vortex_ensure!(
            !parts.children.is_empty(),
            "Widen must carry at least one child"
        );

        let child_type = WidenMetadata::decode(parts.metadata)?
            .child_type
            .ok_or_else(|| vortex_err!("Widen metadata must name the child type"))?;
        let nullability = parts.dtype.nullability();
        let child_dtype = match (child_type, parts.dtype) {
            (ChildType::Integer(ptype), DType::Primitive(parent, _)) if parent.is_int() => {
                vortex_ensure!(
                    is_v1,
                    "integer Widen child requires wire ID {}, got {}",
                    self.id(),
                    parts.serialized_id
                );
                DType::Primitive(PType::try_from(ptype)?, nullability)
            }
            (ChildType::Float(ptype), DType::Primitive(parent, _)) if parent.is_float() => {
                vortex_ensure!(
                    is_v2,
                    "float Widen child requires wire ID {}, got {}",
                    widen_v2_id(),
                    parts.serialized_id
                );
                DType::Primitive(PType::try_from(ptype)?, nullability)
            }
            (ChildType::DecimalPrecision(precision), DType::Decimal(parent, _)) => {
                vortex_ensure!(
                    is_v1,
                    "decimal Widen child requires wire ID {}, got {}",
                    self.id(),
                    parts.serialized_id
                );
                let precision = u8::try_from(precision).map_err(|_| {
                    vortex_err!("Widen child precision {precision} is out of range")
                })?;
                DType::Decimal(
                    DecimalDType::try_new(precision, parent.scale())?,
                    nullability,
                )
            }
            (_, dtype) => {
                vortex_bail!("Widen child metadata does not match the parent dtype {dtype}")
            }
        };

        // The child decoder uses this dtype to read its buffers, so check it against the parent
        // before decoding.
        validate_dtypes(&child_dtype, parts.dtype)?;
        let values = parts.children.get(0, &child_dtype, parts.len)?;

        Ok(WidenArray::try_new(values, parts.dtype.clone())?.into_array())
    }
}
