//! Encodings that enable zero-copy sharing of data with Arrow.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::arrays::{
    BoolArray, ExtensionArray, ListArray, NullArray, PrimitiveArray, StructArray, VarBinViewArray,
};
use crate::arrow::IntoArrowArray;
use crate::builders::builder_with_capacity;
use crate::compute::{preferred_arrow_data_type, to_arrow};
use crate::{Array, ArrayRef, IntoArray};

/// The set of canonical array encodings, also the set of encodings that can be transferred to
/// Arrow with zero-copy.
///
/// Note that a canonical form is not recursive, i.e. a StructArray may contain non-canonical
/// child arrays, which may themselves need to be [canonicalized](ToCanonical).
///
/// # Logical vs. Physical encodings
///
/// Vortex separates logical and physical types, however this creates ambiguity with Arrow, there is
/// no separation. Thus, if you receive an Arrow array, compress it using Vortex, and then
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
/// for all `Utf8` and `Binary` typed arrays in Vortex.
///
#[derive(Debug, Clone)]
pub enum Canonical {
    Null(NullArray),
    Bool(BoolArray),
    Primitive(PrimitiveArray),
    Struct(StructArray),
    // TODO(joe): maybe this should be a ListView, however this will be annoying in spiral
    List(ListArray),
    VarBinView(VarBinViewArray),
    Extension(ExtensionArray),
}

impl Canonical {
    // Create an empty canonical array of the given dtype.
    pub fn empty(dtype: &DType) -> Canonical {
        builder_with_capacity(dtype, 0)
            .finish()
            .to_canonical()
            .vortex_expect("cannot fail to convert an empty array to canonical")
    }
}

// Unwrap canonical type back down to specialized type.
impl Canonical {
    pub fn into_null(self) -> VortexResult<NullArray> {
        match self {
            Canonical::Null(a) => Ok(a),
            _ => vortex_bail!("Cannot unwrap NullArray from {:?}", &self),
        }
    }

    pub fn into_bool(self) -> VortexResult<BoolArray> {
        match self {
            Canonical::Bool(a) => Ok(a),
            _ => vortex_bail!("Cannot unwrap BoolArray from {:?}", &self),
        }
    }

    pub fn into_primitive(self) -> VortexResult<PrimitiveArray> {
        match self {
            Canonical::Primitive(a) => Ok(a),
            _ => vortex_bail!("Cannot unwrap PrimitiveArray from {:?}", &self),
        }
    }

    pub fn into_struct(self) -> VortexResult<StructArray> {
        match self {
            Canonical::Struct(a) => Ok(a),
            _ => vortex_bail!("Cannot unwrap StructArray from {:?}", &self),
        }
    }

    pub fn into_list(self) -> VortexResult<ListArray> {
        match self {
            Canonical::List(a) => Ok(a),
            _ => vortex_bail!("Cannot unwrap StructArray from {:?}", &self),
        }
    }

    pub fn into_varbinview(self) -> VortexResult<VarBinViewArray> {
        match self {
            Canonical::VarBinView(a) => Ok(a),
            _ => vortex_bail!("Cannot unwrap VarBinViewArray from {:?}", &self),
        }
    }

    pub fn into_extension(self) -> VortexResult<ExtensionArray> {
        match self {
            Canonical::Extension(a) => Ok(a),
            _ => vortex_bail!("Cannot unwrap ExtensionArray from {:?}", &self),
        }
    }
}

impl AsRef<dyn Array> for Canonical {
    fn as_ref(&self) -> &(dyn Array + 'static) {
        match &self {
            Canonical::Null(a) => a,
            Canonical::Bool(a) => a,
            Canonical::Primitive(a) => a,
            Canonical::Struct(a) => a,
            Canonical::List(a) => a,
            Canonical::VarBinView(a) => a,
            Canonical::Extension(a) => a,
        }
    }
}

impl IntoArray for Canonical {
    fn into_array(self) -> ArrayRef {
        match self {
            Canonical::Null(a) => a.into_array(),
            Canonical::Bool(a) => a.into_array(),
            Canonical::Primitive(a) => a.into_array(),
            Canonical::Struct(a) => a.into_array(),
            Canonical::List(a) => a.into_array(),
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
pub trait ToCanonical: Array {
    fn to_null(&self) -> VortexResult<NullArray> {
        self.to_canonical()?.into_null()
    }

    fn to_bool(&self) -> VortexResult<BoolArray> {
        self.to_canonical()?.into_bool()
    }

    fn to_primitive(&self) -> VortexResult<PrimitiveArray> {
        self.to_canonical()?.into_primitive()
    }

    fn to_struct(&self) -> VortexResult<StructArray> {
        self.to_canonical()?.into_struct()
    }

    fn to_list(&self) -> VortexResult<ListArray> {
        self.to_canonical()?.into_list()
    }

    fn to_varbinview(&self) -> VortexResult<VarBinViewArray> {
        self.to_canonical()?.into_varbinview()
    }

    fn to_extension(&self) -> VortexResult<ExtensionArray> {
        self.to_canonical()?.into_extension()
    }
}

impl<A: Array + ?Sized> ToCanonical for A {}

impl IntoArrowArray for ArrayRef {
    /// Convert this [`ArrayRef`] into an Arrow [`ArrayRef`] by using the array's preferred
    /// Arrow [`DataType`].
    fn into_arrow_preferred(self) -> VortexResult<ArrowArrayRef> {
        let data_type = preferred_arrow_data_type(&self)?;
        self.into_arrow(&data_type)
    }

    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrowArrayRef> {
        to_arrow(&self, data_type)
    }
}

/// This conversion is always "free" and should not touch underlying data. All it does is create an
/// owned pointer to the underlying concrete array type.
///
/// This combined with the above [ToCanonical] impl for [ArrayRef] allows simple two-way conversions
/// between arbitrary Vortex encodings and canonical Arrow-compatible encodings.
impl From<Canonical> for ArrayRef {
    fn from(value: Canonical) -> Self {
        match value {
            Canonical::Null(a) => a.into_array(),
            Canonical::Bool(a) => a.into_array(),
            Canonical::Primitive(a) => a.into_array(),
            Canonical::Struct(a) => a.into_array(),
            Canonical::List(a) => a.into_array(),
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

    use crate::array::Array;
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

        assert!(arrow_struct
            .column(0)
            .as_any()
            .downcast_ref::<ArrowPrimitiveArray<UInt64Type>>()
            .is_some());

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
            ArrowPrimitiveArray::from_iter([100i64]),
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
