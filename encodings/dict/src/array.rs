use std::fmt::{Debug, Display};

use arrow_buffer::BooleanBuffer;
use serde::{Deserialize, Serialize};
use vortex_array::array::BoolArray;
use vortex_array::compute::unary::scalar_at;
use vortex_array::compute::{take, TakeOptions};
use vortex_array::encoding::ids;
use vortex_array::stats::StatsSet;
use vortex_array::validity::{LogicalValidity, ValidityVTable};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::visitor::{ArrayVisitor, VisitorVTable};
use vortex_array::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoArrayData,
    IntoArrayVariant, IntoCanonical,
};
use vortex_dtype::{match_each_integer_ptype, DType, PType};
use vortex_error::{vortex_bail, vortex_panic, VortexExpect as _, VortexResult};

impl_encoding!("vortex.dict", ids::DICT, Dict);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictMetadata {
    codes_ptype: PType,
    values_len: usize,
}

impl Display for DictMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl DictArray {
    pub fn try_new(codes: ArrayData, values: ArrayData) -> VortexResult<Self> {
        if !codes.dtype().is_unsigned_int() || codes.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non-nullable unsigned int", codes.dtype());
        }
        Self::try_from_parts(
            values.dtype().clone(),
            codes.len(),
            DictMetadata {
                codes_ptype: PType::try_from(codes.dtype())
                    .vortex_expect("codes dtype must be uint"),
                values_len: values.len(),
            },
            [codes, values].into(),
            StatsSet::default(),
        )
    }

    #[inline]
    pub fn codes(&self) -> ArrayData {
        self.as_ref()
            .child(0, &DType::from(self.metadata().codes_ptype), self.len())
            .vortex_expect("DictArray is missing its codes child array")
    }

    #[inline]
    pub fn values(&self) -> ArrayData {
        self.as_ref()
            .child(1, self.dtype(), self.metadata().values_len)
            .vortex_expect("DictArray is missing its values child array")
    }
}

impl ArrayTrait for DictArray {}

impl IntoCanonical for DictArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        match self.dtype() {
            // NOTE: Utf8 and Binary will decompress into VarBinViewArray, which requires a full
            // decompression to construct the views child array.
            // For this case, it is *always* faster to decompress the values first and then create
            // copies of the view pointers.
            DType::Utf8(_) | DType::Binary(_) => {
                let canonical_values: ArrayData = self.values().into_canonical()?.into();
                take(canonical_values, self.codes(), TakeOptions::default())?.into_canonical()
            }
            // Non-string case: take and then canonicalize
            _ => take(self.values(), self.codes(), TakeOptions::default())?.into_canonical(),
        }
    }
}

impl ValidityVTable<DictArray> for DictEncoding {
    fn is_valid(&self, array: &DictArray, index: usize) -> bool {
        let values_index = scalar_at(array.codes(), index)
            .unwrap_or_else(|err| {
                vortex_panic!(err, "Failed to get index {} from DictArray codes", index)
            })
            .as_ref()
            .try_into()
            .vortex_expect("Failed to convert dictionary code to usize");
        array.values().with_dyn(|a| a.is_valid(values_index))
    }

    fn logical_validity(&self, array: &DictArray) -> LogicalValidity {
        if array.dtype().is_nullable() {
            let primitive_codes = array
                .codes()
                .into_primitive()
                .vortex_expect("Failed to convert DictArray codes to primitive array");
            match_each_integer_ptype!(primitive_codes.ptype(), |$P| {
                let is_valid = primitive_codes
                    .maybe_null_slice::<$P>();
                let is_valid_buffer = BooleanBuffer::collect_bool(is_valid.len(), |idx| {
                    is_valid[idx] != 0
                });
                LogicalValidity::Array(BoolArray::from(is_valid_buffer).into_array())
            })
        } else {
            LogicalValidity::AllValid(array.len())
        }
    }
}

impl VisitorVTable<DictArray> for DictEncoding {
    fn accept(&self, array: &DictArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("values", &array.values())?;
        visitor.visit_child("codes", &array.codes())
    }
}
