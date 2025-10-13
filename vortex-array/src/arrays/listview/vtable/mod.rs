// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::ListViewArray;
use crate::vtable::{NotSupported, VTable, ValidityVTableFromValidityHelper};
use crate::{vtable, EncodingId, EncodingRef};

mod array;
mod canonical;
mod operations;
mod operator;
mod serde;
mod validity;
mod visitor;

vtable!(ListView);

#[derive(Clone, Debug)]
pub struct ListViewEncoding;

impl VTable for ListViewVTable {
    type Array = ListViewArray;
    type Encoding = ListViewEncoding;

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
        EncodingId::new_ref("vortex.listview")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ListViewEncoding.as_ref())
    }
}
