// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encodings that enable zero-copy sharing of data with Arrow.

use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_panic};

use crate::arrays::{
    BoolArray, DecimalArray, ExtensionArray, FixedSizeListArray, ListArray, NullArray,
    PrimitiveArray, StructArray, VarBinViewArray,
};
use crate::builders::builder_with_capacity;
use crate::{Array, ArrayRef, IntoArray};

/// An enum capturing the default uncompressed encodings for each [Vortex type][DType].
///
/// Any array can be decoded into canonical form via the [`to_canonical`][Array::to_canonical]
/// trait method. This is the simplest encoding for a type, and will not be compressed but may
/// contain compressed child arrays.
///
/// Canonical form is useful for doing type-specific compute where you need to know that all
/// elements are laid out decompressed and contiguous in memory.
///
/// # Laziness
///
/// Canonical form is not recursive, so while a `StructArray` is the canonical format for any
/// `Struct` type, individual column child arrays may still be compressed. This allows
/// compute over Vortex arrays to push decoding as late as possible, and ideally many child arrays
/// never need to be decoded into canonical form at all depending on the compute.
///
/// # Arrow interoperability
///
/// All of the Vortex canonical encodings have an equivalent Arrow encoding that can be built
/// zero-copy, and the corresponding Arrow array types can also be built directly.
///
/// The full list of canonical types and their equivalent Arrow array types are:
///
/// * `NullArray`: [`arrow_array::NullArray`]
/// * `BoolArray`: [`arrow_array::BooleanArray`]
/// * `PrimitiveArray`: [`arrow_array::PrimitiveArray`]
/// * `DecimalArray`: [`arrow_array::Decimal128Array`] and [`arrow_array::Decimal256Array`]
/// * `VarBinViewArray`: [`arrow_array::GenericByteViewArray`]
/// * `ListArray`: [`arrow_array::ListArray`]
/// * `FixedSizeListArray`: [`arrow_array::FixedSizeListArray`]
/// * `StructArray`: [`arrow_array::StructArray`]
///
/// Vortex uses a logical type system, unlike Arrow which uses physical encodings for its types.
/// As an example, there are at least six valid physical encodings for a `Utf8` array. This can
/// create ambiguity.
/// Thus, if you receive an Arrow array, compress it using Vortex, and then
/// decompress it later to pass to a compute kernel, there are multiple suitable Arrow array
/// variants to hold the data.
///
/// To disambiguate, we choose a canonical physical encoding for every Vortex [`DType`], which
/// will correspond to an arrow-rs [`arrow_schema::DataType`].
///
/// # Views support
///
/// Binary and String views, also known as "German strings" are a better encoding format for
/// nearly all use-cases. Variable-length binary views are part of the Apache Arrow spec, and are
/// fully supported by the Datafusion query engine. We use them as our canonical string encoding
/// for all `Utf8` and `Binary` typed arrays in Vortex. They provide considerably faster filter
/// execution than the core `StringArray` and `BinaryArray` types, at the expense of potentially
/// needing [garbage collection][arrow_array::GenericByteViewArray::gc] to clear unreferenced items
/// from memory.
///
/// # For Developers
///
/// If you add another variant to this enum, make sure to update [`Array::is_canonical`],
/// [`ArrayRegistry::canonical_only`], and the fuzzer in `fuzz/fuzz_targets/array_ops.rs`.
///
/// [`ArrayRegistry::canonical_only`]: crate::ArrayRegistry::canonical_only
#[derive(Debug, Clone)]
pub enum Canonical {
    Null(NullArray),
    Bool(BoolArray),
    Primitive(PrimitiveArray),
    Decimal(DecimalArray),
    VarBinView(VarBinViewArray),
    // TODO(connor)[ListView]: Convert the canonical encoding of `List` to `ListViewArray`.
    List(ListArray),
    FixedSizeList(FixedSizeListArray),
    Struct(StructArray),
    Extension(ExtensionArray),
}

impl Canonical {
    /// Create an empty canonical array of the given dtype.
    pub fn empty(dtype: &DType) -> Canonical {
        builder_with_capacity(dtype, 0).finish_into_canonical()
    }
}

impl Canonical {
    /// Performs a (potentially expensive) compaction operation on the array before it is complete.
    ///
    /// This is mostly relevant for the variable-length types such as Utf8, Binary or List where
    /// they can accumulate wasted space after slicing and taking operations.
    ///
    /// This operation is very expensive and can result in things like allocations, full-scans
    /// and copy operations.
    pub fn compact(&self) -> VortexResult<Canonical> {
        match self {
            Canonical::VarBinView(array) => Ok(Canonical::VarBinView(array.compact_buffers()?)),
            Canonical::List(array) => Ok(Canonical::List(array.reset_offsets()?)),
            _ => Ok(self.clone()),
        }
    }
}

// Unwrap canonical type back down to specialized type.
impl Canonical {
    pub fn as_null(&self) -> &NullArray {
        if let Canonical::Null(a) = self {
            a
        } else {
            vortex_panic!("Cannot get NullArray from {:?}", &self)
        }
    }

    pub fn into_null(self) -> NullArray {
        if let Canonical::Null(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap NullArray from {:?}", &self)
        }
    }

    pub fn as_bool(&self) -> &BoolArray {
        if let Canonical::Bool(a) = self {
            a
        } else {
            vortex_panic!("Cannot get BoolArray from {:?}", &self)
        }
    }

    pub fn into_bool(self) -> BoolArray {
        if let Canonical::Bool(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap BoolArray from {:?}", &self)
        }
    }

    pub fn as_primitive(&self) -> &PrimitiveArray {
        if let Canonical::Primitive(a) = self {
            a
        } else {
            vortex_panic!("Cannot get PrimitiveArray from {:?}", &self)
        }
    }

    pub fn into_primitive(self) -> PrimitiveArray {
        if let Canonical::Primitive(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap PrimitiveArray from {:?}", &self)
        }
    }

    pub fn as_decimal(&self) -> &DecimalArray {
        if let Canonical::Decimal(a) = self {
            a
        } else {
            vortex_panic!("Cannot get DecimalArray from {:?}", &self)
        }
    }

    pub fn into_decimal(self) -> DecimalArray {
        if let Canonical::Decimal(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap DecimalArray from {:?}", &self)
        }
    }

    pub fn as_varbinview(&self) -> &VarBinViewArray {
        if let Canonical::VarBinView(a) = self {
            a
        } else {
            vortex_panic!("Cannot get VarBinViewArray from {:?}", &self)
        }
    }

    pub fn into_varbinview(self) -> VarBinViewArray {
        if let Canonical::VarBinView(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap VarBinViewArray from {:?}", &self)
        }
    }

    pub fn as_list(&self) -> &ListArray {
        if let Canonical::List(a) = self {
            a
        } else {
            vortex_panic!("Cannot get ListArray from {:?}", &self)
        }
    }

    pub fn into_list(self) -> ListArray {
        if let Canonical::List(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap ListArray from {:?}", &self)
        }
    }

    pub fn as_fixed_size_list(&self) -> &FixedSizeListArray {
        if let Canonical::FixedSizeList(a) = self {
            a
        } else {
            vortex_panic!("Cannot get FixedSizeListArray from {:?}", &self)
        }
    }

    pub fn into_fixed_size_list(self) -> FixedSizeListArray {
        if let Canonical::FixedSizeList(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap FixedSizeListArray from {:?}", &self)
        }
    }

    pub fn as_struct(&self) -> &StructArray {
        if let Canonical::Struct(a) = self {
            a
        } else {
            vortex_panic!("Cannot get StructArray from {:?}", &self)
        }
    }

    pub fn into_struct(self) -> StructArray {
        if let Canonical::Struct(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap StructArray from {:?}", &self)
        }
    }

    pub fn as_extension(&self) -> &ExtensionArray {
        if let Canonical::Extension(a) = self {
            a
        } else {
            vortex_panic!("Cannot get ExtensionArray from {:?}", &self)
        }
    }

    pub fn into_extension(self) -> ExtensionArray {
        if let Canonical::Extension(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap ExtensionArray from {:?}", &self)
        }
    }
}

impl AsRef<dyn Array> for Canonical {
    fn as_ref(&self) -> &(dyn Array + 'static) {
        match &self {
            Canonical::Null(a) => a.as_ref(),
            Canonical::Bool(a) => a.as_ref(),
            Canonical::Primitive(a) => a.as_ref(),
            Canonical::Decimal(a) => a.as_ref(),
            Canonical::Struct(a) => a.as_ref(),
            Canonical::List(a) => a.as_ref(),
            Canonical::FixedSizeList(a) => a.as_ref(),
            Canonical::VarBinView(a) => a.as_ref(),
            Canonical::Extension(a) => a.as_ref(),
        }
    }
}

impl IntoArray for Canonical {
    fn into_array(self) -> ArrayRef {
        match self {
            Canonical::Null(a) => a.into_array(),
            Canonical::Bool(a) => a.into_array(),
            Canonical::Primitive(a) => a.into_array(),
            Canonical::Decimal(a) => a.into_array(),
            Canonical::Struct(a) => a.into_array(),
            Canonical::List(a) => a.into_array(),
            Canonical::FixedSizeList(a) => a.into_array(),
            Canonical::VarBinView(a) => a.into_array(),
            Canonical::Extension(a) => a.into_array(),
        }
    }
}

/// Trait for types that can be converted from an owned type into an owned array variant.
///
/// # Canonicalization
///
/// This trait has a blanket implementation for all types implementing [ToCanonical].
pub trait ToCanonical {
    /// Canonicalize into a [`NullArray`] if the target is [`Null`][DType::Null] typed.
    fn to_null(&self) -> NullArray;

    /// Canonicalize into a [`BoolArray`] if the target is [`Bool`][DType::Bool] typed.
    fn to_bool(&self) -> BoolArray;

    /// Canonicalize into a [`PrimitiveArray`] if the target is [`Primitive`][DType::Primitive]
    /// typed.
    fn to_primitive(&self) -> PrimitiveArray;

    /// Canonicalize into a [`DecimalArray`] if the target is [`Decimal`][DType::Decimal]
    /// typed.
    fn to_decimal(&self) -> DecimalArray;

    /// Canonicalize into a [`StructArray`] if the target is [`Struct`][DType::Struct] typed.
    fn to_struct(&self) -> StructArray;

    /// Canonicalize into a [`ListArray`] if the target is [`List`][DType::List] typed.
    fn to_list(&self) -> ListArray;

    /// Canonicalize into a [`ListArray`] if the target is [`List`][DType::List] typed.
    fn to_fixed_size_list(&self) -> FixedSizeListArray;

    /// Canonicalize into a [`VarBinViewArray`] if the target is [`Utf8`][DType::Utf8]
    /// or [`Binary`][DType::Binary] typed.
    fn to_varbinview(&self) -> VarBinViewArray;

    /// Canonicalize into an [`ExtensionArray`] if the array is [`Extension`][DType::Extension]
    /// typed.
    fn to_extension(&self) -> ExtensionArray;
}

// Blanket impl for all Array encodings.
impl<A: Array + ?Sized> ToCanonical for A {
    fn to_null(&self) -> NullArray {
        self.to_canonical().into_null()
    }

    fn to_bool(&self) -> BoolArray {
        self.to_canonical().into_bool()
    }

    fn to_primitive(&self) -> PrimitiveArray {
        self.to_canonical().into_primitive()
    }

    fn to_decimal(&self) -> DecimalArray {
        self.to_canonical().into_decimal()
    }

    fn to_struct(&self) -> StructArray {
        self.to_canonical().into_struct()
    }

    fn to_list(&self) -> ListArray {
        self.to_canonical().into_list()
    }

    fn to_fixed_size_list(&self) -> FixedSizeListArray {
        self.to_canonical().into_fixed_size_list()
    }

    fn to_varbinview(&self) -> VarBinViewArray {
        self.to_canonical().into_varbinview()
    }

    fn to_extension(&self) -> ExtensionArray {
        self.to_canonical().into_extension()
    }
}

impl From<Canonical> for ArrayRef {
    fn from(value: Canonical) -> Self {
        match value {
            Canonical::Null(a) => a.into_array(),
            Canonical::Bool(a) => a.into_array(),
            Canonical::Primitive(a) => a.into_array(),
            Canonical::Decimal(a) => a.into_array(),
            Canonical::Struct(a) => a.into_array(),
            Canonical::List(a) => a.into_array(),
            Canonical::FixedSizeList(a) => a.into_array(),
            Canonical::VarBinView(a) => a.into_array(),
            Canonical::Extension(a) => a.into_array(),
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, Int64Type, UInt64Type};
    use arrow_array::{
        Array as ArrowArray, ArrayRef as ArrowArrayRef, ListArray as ArrowListArray,
        PrimitiveArray as ArrowPrimitiveArray, StringArray, StringViewArray,
        StructArray as ArrowStructArray,
    };
    use arrow_buffer::{NullBufferBuilder, OffsetBuffer};
    use arrow_schema::{DataType, Field};
    use vortex_buffer::buffer;

    use crate::arrays::{ConstantArray, StructArray};
    use crate::arrow::{FromArrowArray, IntoArrowArray};
    use crate::{ArrayRef, IntoArray};

    #[test]
    fn test_canonicalize_nested_struct() {
        // Create a struct array with multiple internal components.
        let nested_struct_array = StructArray::from_fields(&[
            ("a", buffer![1u64].into_array()),
            (
                "b",
                StructArray::from_fields(&[(
                    "inner_a",
                    // The nested struct contains a ConstantArray representing the primitive array
                    //   [100i64]
                    // ConstantArray is not a canonical type, so converting `into_arrow()` should
                    // map this to the nearest canonical type (PrimitiveArray).
                    ConstantArray::new(100i64, 1).into_array(),
                )])
                .unwrap()
                .into_array(),
            ),
        ])
        .unwrap();

        let arrow_struct = nested_struct_array
            .into_array()
            .into_arrow_preferred()
            .unwrap()
            .as_any()
            .downcast_ref::<ArrowStructArray>()
            .cloned()
            .unwrap();

        assert!(
            arrow_struct
                .column(0)
                .as_any()
                .downcast_ref::<ArrowPrimitiveArray<UInt64Type>>()
                .is_some()
        );

        let inner_struct = arrow_struct
            .column(1)
            .clone()
            .as_any()
            .downcast_ref::<ArrowStructArray>()
            .cloned()
            .unwrap();

        let inner_a = inner_struct
            .column(0)
            .as_any()
            .downcast_ref::<ArrowPrimitiveArray<Int64Type>>();
        assert!(inner_a.is_some());

        assert_eq!(
            inner_a.cloned().unwrap(),
            ArrowPrimitiveArray::from_iter([100i64])
        );
    }

    #[test]
    fn roundtrip_struct() {
        let mut nulls = NullBufferBuilder::new(6);
        nulls.append_n_non_nulls(4);
        nulls.append_null();
        nulls.append_non_null();
        let names = Arc::new(StringViewArray::from_iter(vec![
            Some("Joseph"),
            None,
            Some("Angela"),
            Some("Mikhail"),
            None,
            None,
        ]));
        let ages = Arc::new(ArrowPrimitiveArray::<Int32Type>::from(vec![
            Some(25),
            Some(31),
            None,
            Some(57),
            None,
            None,
        ]));

        let arrow_struct = ArrowStructArray::new(
            vec![
                Arc::new(Field::new("name", DataType::Utf8View, true)),
                Arc::new(Field::new("age", DataType::Int32, true)),
            ]
            .into(),
            vec![names, ages],
            nulls.finish(),
        );

        let vortex_struct = ArrayRef::from_arrow(&arrow_struct, true);

        assert_eq!(
            &arrow_struct,
            vortex_struct.into_arrow_preferred().unwrap().as_struct()
        );
    }

    #[test]
    fn roundtrip_list() {
        let names = Arc::new(StringArray::from_iter(vec![
            Some("Joseph"),
            Some("Angela"),
            Some("Mikhail"),
        ]));

        let arrow_list = ArrowListArray::new(
            Arc::new(Field::new_list_field(DataType::Utf8, true)),
            OffsetBuffer::from_lengths(vec![0, 2, 1]),
            names,
            None,
        );
        let list_data_type = arrow_list.data_type();

        let vortex_list = ArrayRef::from_arrow(&arrow_list, true);

        let rt_arrow_list = vortex_list.into_arrow(list_data_type).unwrap();

        assert_eq!(
            (Arc::new(arrow_list.clone()) as ArrowArrayRef).as_ref(),
            rt_arrow_list.as_ref()
        );
    }
}
