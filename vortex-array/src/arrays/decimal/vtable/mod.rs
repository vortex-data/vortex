// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::DecimalArray;
use crate::vtable::{NotSupported, VTable, ValidityVTableFromValidityHelper};
use crate::{EncodingId, EncodingRef, vtable};

mod array;
mod canonical;
mod operations;
mod serde;
mod validity;
mod visitor;

vtable!(Decimal);

impl VTable for DecimalVTable {
    type Array = DecimalArray;
    type Encoding = DecimalEncoding;
    type Metadata = crate::ProstMetadata<serde::DecimalMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type PipelineVTable = NotSupported;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.decimal")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(DecimalEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct DecimalEncoding;
