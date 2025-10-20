// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::TemporalArray;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VTable, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, Canonical, DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, vortex_bail};

use crate::{
    DateTimePartsArray, DateTimePartsEncoding, DateTimePartsMetadata, DateTimePartsVTable,
};

impl SerdeVTable<DateTimePartsVTable> for DateTimePartsVTable {
    fn build(
        _encoding: &DateTimePartsEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<<DateTimePartsVTable as VTable>::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DateTimePartsArray> {
        if children.len() != 3 {
            vortex_bail!(
                "Expected 3 children for datetime-parts encoding, found {}",
                children.len()
            )
        }

        let days = children.get(
            0,
            &DType::Primitive(metadata.days_ptype(), dtype.nullability()),
            len,
        )?;
        let seconds = children.get(
            1,
            &DType::Primitive(metadata.seconds_ptype(), Nullability::NonNullable),
            len,
        )?;
        let subseconds = children.get(
            2,
            &DType::Primitive(metadata.subseconds_ptype(), Nullability::NonNullable),
            len,
        )?;

        DateTimePartsArray::try_new(dtype.clone(), days, seconds, subseconds)
    }
}

impl EncodeVTable<DateTimePartsVTable> for DateTimePartsVTable {
    fn encode(
        _encoding: &DateTimePartsEncoding,
        canonical: &Canonical,
        _like: Option<&DateTimePartsArray>,
    ) -> VortexResult<Option<DateTimePartsArray>> {
        let ext_array = canonical.clone().into_extension();
        let temporal = TemporalArray::try_from(ext_array)?;

        Ok(Some(DateTimePartsArray::try_from(temporal)?))
    }
}

impl VisitorVTable<DateTimePartsVTable> for DateTimePartsVTable {
    fn metadata(array: &DateTimePartsArray) -> <DateTimePartsVTable as VTable>::Metadata {
        ProstMetadata(DateTimePartsMetadata {
            days_ptype: array.days().dtype().as_ptype() as i32,
            seconds_ptype: array.seconds().dtype().as_ptype() as i32,
            subseconds_ptype: array.subseconds().dtype().as_ptype() as i32,
        })
    }

    fn visit_buffers(_array: &DateTimePartsArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &DateTimePartsArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("days", array.days());
        visitor.visit_child("seconds", array.seconds());
        visitor.visit_child("subseconds", array.subseconds());
    }
}

#[cfg(test)]
mod test {
    use vortex_array::ProstMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use super::*;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_datetimeparts_metadata() {
        check_metadata(
            "datetimeparts.metadata",
            ProstMetadata(DateTimePartsMetadata {
                days_ptype: PType::I64 as i32,
                seconds_ptype: PType::I64 as i32,
                subseconds_ptype: PType::I64 as i32,
            }),
        );
    }
}
