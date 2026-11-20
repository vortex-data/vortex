// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Wire layout of a `vortex.widen` node.
//!
//! A node has one child and no buffers of its own. The enclosing format supplies the logical dtype
//! and length of the [`Widen`]. The child has the same length and nullability. A decimal child
//! also has the same scale.
//!
//! A protobuf `oneof` supplies the rest of the child dtype. Tag 1 holds the integer [`PType`].
//! Tag 2 holds the decimal precision as a `uint32`. The child dtype must meet the requirements
//! of [`WidenArray::try_new`].

use prost::Message as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

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

#[derive(Clone, prost::Message)]
struct WidenMetadata {
    #[prost(oneof = "ChildType", tags = "1, 2")]
    child_type: Option<ChildType>,
}

#[derive(Clone, prost::Oneof)]
enum ChildType {
    #[prost(enumeration = "PType", tag = "1")]
    Integer(i32),
    #[prost(uint32, tag = "2")]
    DecimalPrecision(u32),
}

/// Reads and writes serialized [`Widen`] arrays.
///
/// Writing reuses the child without executing its values or validity. When the child is a
/// [`Decimal`] array, this plugin writes `vortex.decimal` so older readers can read it. The decimal
/// reader handles those nodes.
#[derive(Clone, Debug)]
pub struct WidenPlugin;

impl ArrayPlugin for WidenPlugin {
    fn id(&self) -> ArrayId {
        ArrayVTable::id(&Widen)
    }

    fn write_ids(&self) -> Vec<ArrayId> {
        // Older editions allow `vortex.decimal`, which can store the child's narrow buffer
        // under the parent's decimal dtype.
        vec![ArrayVTable::id(&Decimal), self.id()]
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

        let child_type = match values.dtype() {
            DType::Primitive(ptype, _) if ptype.is_int() => ChildType::Integer(*ptype as i32),
            DType::Decimal(dtype, _) => ChildType::DecimalPrecision(u32::from(dtype.precision())),
            dtype => {
                vortex_bail!(
                    "Widen serialization requires an integer or decimal child, got {dtype}"
                )
            }
        };
        let metadata = WidenMetadata {
            child_type: Some(child_type),
        };

        Ok(Some(ArraySerialization::new(
            self.id(),
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
        vortex_ensure_eq!(parts.serialized_id, self.id(), "expected the Widen wire ID");
        vortex_ensure_eq!(parts.buffers.len(), 0, "Widen must carry zero node buffers");
        vortex_ensure_eq!(parts.children.len(), 1, "Widen must carry one child");

        let child_type = WidenMetadata::decode(parts.metadata)?
            .child_type
            .ok_or_else(|| vortex_err!("Widen metadata must name the child type"))?;
        let nullability = parts.dtype.nullability();
        let child_dtype = match (child_type, parts.dtype) {
            (ChildType::Integer(ptype), DType::Primitive(parent, _)) if parent.is_int() => {
                DType::Primitive(PType::try_from(ptype)?, nullability)
            }
            (ChildType::DecimalPrecision(precision), DType::Decimal(parent, _)) => {
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
