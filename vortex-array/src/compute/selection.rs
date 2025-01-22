use std::fmt::Display;

use serde::{Deserialize, Serialize};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

use crate::compute::{ComputeVTable, FilterMask};
use crate::stats::{StatisticsVTable, StatsSet};
use crate::validate::ValidateVTable;
use crate::validity::{ArrayValidity, LogicalValidity, ValidityVTable};
use crate::variants::{
    BinaryArrayTrait, BoolArrayTrait, ListArrayTrait, NullArrayTrait, PrimitiveArrayTrait,
    Utf8ArrayTrait, VariantsVTable,
};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{impl_encoding, ArrayDType, ArrayData, Canonical, IntoCanonical};

impl_encoding!("lol.selection", 10_000u16, Selection);

// No need for the selection metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionMetadata {
    #[serde(skip_serializing, skip_deserializing)]
    mask: Option<FilterMask>,
    #[serde(skip_serializing, skip_deserializing)]
    dtype: Option<DType>,
}

impl Default for SelectionMetadata {
    fn default() -> Self {
        todo!("wat r u doing")
    }
}

impl Display for SelectionMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SelectionMetadata{{}}")
    }
}

impl SelectionArray {
    pub fn new(data: ArrayData, mask: FilterMask) -> Self {
        assert_eq!(
            mask.len(),
            data.len(),
            "Mask length must match array length"
        );

        Self::try_from_parts(
            data.dtype().clone(),
            data.len(),
            SelectionMetadata {
                mask: Some(mask),
                dtype: Some(data.dtype().clone()),
            },
            None,
            Some([data].into()),
            StatsSet::default(),
        )
        .vortex_expect("SelectionArray try_from_parts")
    }

    pub fn backing(&self) -> VortexResult<ArrayData> {
        self.as_ref().child(
            0,
            &self.metadata().dtype.clone().unwrap(),
            self.metadata().mask.clone().unwrap().len(),
        )
    }
}

impl IntoCanonical for SelectionArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        todo!("CALL into_canoncical_with_selection on child")
    }
}

impl StatisticsVTable<SelectionArray> for SelectionEncoding {}

impl ValidityVTable<SelectionArray> for SelectionEncoding {
    fn is_valid(&self, array: &SelectionArray, index: usize) -> bool {
        array.backing().vortex_expect("backing").is_valid(index)
    }

    fn logical_validity(&self, array: &SelectionArray) -> LogicalValidity {
        array.backing().vortex_expect("backing").logical_validity()
    }
}

impl ValidateVTable<SelectionArray> for SelectionEncoding {}

impl ComputeVTable for SelectionEncoding {}

impl VisitorVTable<SelectionArray> for SelectionEncoding {
    fn accept(&self, _array: &SelectionArray, _visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        todo!("i accept you selection array")
    }
}

impl VariantsVTable<SelectionArray> for SelectionEncoding {
    fn as_null_array<'a>(&self, array: &'a SelectionArray) -> Option<&'a dyn NullArrayTrait> {
        Some(array)
    }

    fn as_bool_array<'a>(&self, array: &'a SelectionArray) -> Option<&'a dyn BoolArrayTrait> {
        Some(array)
    }

    fn as_primitive_array<'a>(
        &self,
        array: &'a SelectionArray,
    ) -> Option<&'a dyn PrimitiveArrayTrait> {
        Some(array)
    }

    fn as_utf8_array<'a>(&self, array: &'a SelectionArray) -> Option<&'a dyn Utf8ArrayTrait> {
        Some(array)
    }

    fn as_binary_array<'a>(&self, array: &'a SelectionArray) -> Option<&'a dyn BinaryArrayTrait> {
        Some(array)
    }

    // fn as_struct_array<'a>(&self, array: &'a SelectionArray) -> Option<&'a dyn StructArrayTrait> {
    //     Some(array)
    // }

    fn as_list_array<'a>(&self, array: &'a SelectionArray) -> Option<&'a dyn ListArrayTrait> {
        Some(array)
    }

    // fn as_extension_array<'a>(
    //     &self,
    //     array: &'a SelectionArray,
    // ) -> Option<&'a dyn ExtensionArrayTrait> {
    //     Some(array)
    // }
}

impl NullArrayTrait for SelectionArray {}

impl BoolArrayTrait for SelectionArray {}

impl Utf8ArrayTrait for SelectionArray {}
impl BinaryArrayTrait for SelectionArray {}
impl ListArrayTrait for SelectionArray {}
impl PrimitiveArrayTrait for SelectionArray {}
