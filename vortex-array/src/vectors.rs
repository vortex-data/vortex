// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_dtype::DType;
use vortex_dtype::NativeDecimalType;
use vortex_dtype::NativePType;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::match_each_decimal_value_type;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexExpect;
use vortex_vector::Vector;
use vortex_vector::VectorOps;
use vortex_vector::binaryview::BinaryViewType;
use vortex_vector::binaryview::BinaryViewVector;
use vortex_vector::bool::BoolVector;
use vortex_vector::decimal::DVector;
use vortex_vector::decimal::DecimalVector;
use vortex_vector::fixed_size_list::FixedSizeListVector;
use vortex_vector::listview::ListViewVector;
use vortex_vector::null::NullVector;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PrimitiveVector;
use vortex_vector::struct_::StructVector;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::NullArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::validity::Validity;

/// Trait for converting vector types into arrays.
pub trait VectorIntoArray<T> {
    /// Converts the vector into an array of the specified data type.
    fn into_array(self, dtype: &DType) -> T;
}

impl VectorIntoArray<ArrayRef> for Vector {
    fn into_array(self, dtype: &DType) -> ArrayRef {
        match dtype {
            DType::Null => self.into_null().into_array(dtype).into_array(),
            DType::Bool(_) => self.into_bool().into_array(dtype).into_array(),
            DType::Primitive(..) => self.into_primitive().into_array(dtype).into_array(),
            DType::Decimal(..) => self.into_decimal().into_array(dtype).into_array(),
            DType::Utf8(_) => self.into_string().into_array(dtype).into_array(),
            DType::Binary(_) => self.into_binary().into_array(dtype).into_array(),
            DType::List(..) => self.into_list().into_array(dtype).into_array(),
            DType::FixedSizeList(..) => self.into_fixed_size_list().into_array(dtype).into_array(),
            DType::Struct(..) => self.into_struct().into_array(dtype).into_array(),
            DType::Extension(ext_dtype) => {
                let storage = self.into_array(ext_dtype.storage_dtype());
                ExtensionArray::new(ext_dtype.clone(), storage).into_array()
            }
        }
    }
}

impl VectorIntoArray<NullArray> for NullVector {
    fn into_array(self, dtype: &DType) -> NullArray {
        assert!(matches!(dtype, DType::Null));
        NullArray::new(self.len())
    }
}

impl VectorIntoArray<BoolArray> for BoolVector {
    fn into_array(self, dtype: &DType) -> BoolArray {
        assert!(matches!(dtype, DType::Bool(_)));

        let (bits, validity) = self.into_parts();
        BoolArray::from_bit_buffer(bits, Validity::from_mask(validity, dtype.nullability()))
    }
}

impl VectorIntoArray<PrimitiveArray> for PrimitiveVector {
    fn into_array(self, dtype: &DType) -> PrimitiveArray {
        match_each_native_ptype!(self.ptype(), |T| {
            <T as NativePType>::downcast(self).into_array(dtype)
        })
    }
}

impl<T: NativePType> VectorIntoArray<PrimitiveArray> for PVector<T> {
    fn into_array(self, dtype: &DType) -> PrimitiveArray {
        assert!(matches!(dtype, DType::Primitive(_, _)));
        assert_eq!(T::PTYPE, dtype.as_ptype());

        let (values, validity) = self.into_parts();
        // SAFETY: vectors maintain all invariants required for array creation
        unsafe {
            PrimitiveArray::new_unchecked::<T>(
                values,
                Validity::from_mask(validity, dtype.nullability()),
            )
        }
    }
}

impl VectorIntoArray<DecimalArray> for DecimalVector {
    fn into_array(self, dtype: &DType) -> DecimalArray {
        match_each_decimal_value_type!(self.decimal_type(), |D| {
            <D as NativeDecimalType>::downcast(self).into_array(dtype)
        })
    }
}

impl<D: NativeDecimalType> VectorIntoArray<DecimalArray> for DVector<D> {
    fn into_array(self, dtype: &DType) -> DecimalArray {
        assert!(matches!(dtype, DType::Decimal(_, _)));

        let nullability = dtype.nullability();
        let dec_dtype = dtype
            .as_decimal_opt()
            .vortex_expect("expected decimal DType");
        assert_eq!(dec_dtype.precision(), self.precision());
        assert_eq!(dec_dtype.scale(), self.scale());

        let (_ps, values, validity) = self.into_parts();
        // SAFETY: vectors maintain all invariants required for array creation
        unsafe {
            DecimalArray::new_unchecked::<D>(
                values,
                *dec_dtype,
                Validity::from_mask(validity, nullability),
            )
        }
    }
}

impl<T: BinaryViewType> VectorIntoArray<VarBinViewArray> for BinaryViewVector<T> {
    fn into_array(self, dtype: &DType) -> VarBinViewArray {
        assert!(matches!(dtype, DType::Utf8(_)));

        let (views, buffers, validity) = self.into_parts();
        let validity = Validity::from_mask(validity, dtype.nullability());

        let buffers = Arc::try_unwrap(buffers).unwrap_or_else(|b| (*b).clone());

        // SAFETY: vectors maintain all invariants required for array creation
        unsafe {
            VarBinViewArray::new_unchecked(
                views,
                buffers.into_iter().collect(),
                dtype.clone(),
                validity,
            )
        }
    }
}

impl VectorIntoArray<ListViewArray> for ListViewVector {
    fn into_array(self, dtype: &DType) -> ListViewArray {
        assert!(matches!(dtype, DType::List(_, _)));

        let (elements, offsets, sizes, validity) = self.into_parts();
        let validity = Validity::from_mask(validity, dtype.nullability());

        let elements_dtype = dtype.as_list_element_opt().vortex_expect("expected list");
        let elements = Arc::try_unwrap(elements)
            .unwrap_or_else(|e| (*e).clone())
            .into_array(elements_dtype);

        let offsets_dtype = DType::Primitive(offsets.ptype(), NonNullable);
        let offsets = offsets.into_array(&offsets_dtype);

        let sizes_dtype = DType::Primitive(sizes.ptype(), NonNullable);
        let sizes = sizes.into_array(&sizes_dtype);

        // SAFETY: vectors maintain all invariants required for array creation
        unsafe {
            ListViewArray::new_unchecked(
                elements,
                offsets.into_array(),
                sizes.into_array(),
                validity,
            )
        }
    }
}

impl VectorIntoArray<FixedSizeListArray> for FixedSizeListVector {
    fn into_array(self, dtype: &DType) -> FixedSizeListArray {
        assert!(matches!(dtype, DType::FixedSizeList(_, _, _)));

        let len = self.len();
        let (elements, size, validity) = self.into_parts();
        let validity = Validity::from_mask(validity, dtype.nullability());

        let elements_dtype = dtype
            .as_fixed_size_list_element_opt()
            .vortex_expect("expected fixed size list");
        let elements = Arc::try_unwrap(elements)
            .unwrap_or_else(|e| (*e).clone())
            .into_array(elements_dtype);

        // SAFETY: vectors maintain all invariants required for array creation
        unsafe { FixedSizeListArray::new_unchecked(elements, size, validity, len) }
    }
}

impl VectorIntoArray<StructArray> for StructVector {
    fn into_array(self, dtype: &DType) -> StructArray {
        assert!(matches!(dtype, DType::Struct(_, _)));

        let len = self.len();
        let (fields, validity) = self.into_parts();
        let validity = Validity::from_mask(validity, dtype.nullability());

        let struct_fields = dtype.as_struct_fields();
        assert_eq!(fields.len(), struct_fields.nfields());

        let field_arrays: Vec<ArrayRef> = Arc::try_unwrap(fields)
            .unwrap_or_else(|f| (*f).clone())
            .into_iter()
            .zip(struct_fields.fields())
            .map(|(field_vector, field_dtype)| field_vector.into_array(&field_dtype))
            .collect();

        // SAFETY: vectors maintain all invariants required for array creation
        unsafe { StructArray::new_unchecked(field_arrays, struct_fields.clone(), len, validity) }
    }
}
