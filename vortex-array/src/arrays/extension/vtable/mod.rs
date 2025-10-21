// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod canonical;
mod operations;
mod serde;
mod validity;
mod visitor;

use crate::arrays::extension::ExtensionArray;
use crate::vtable::{NotSupported, VTable, ValidityVTableFromChild};
use crate::{EncodingId, EncodingRef, vtable};

vtable!(Extension);

impl VTable for ExtensionVTable {
    type Array = ExtensionArray;
    type Encoding = ExtensionEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type OperatorVTable = NotSupported;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.ext")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ExtensionEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct ExtensionEncoding;
