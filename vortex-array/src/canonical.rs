//! Encodings that enable zero-copy sharing of data with Arrow.

use std::sync::Arc;

use arrow_array::types::*;
use arrow_array::{
    new_null_array, Array, ArrayRef, ArrowPrimitiveType, BooleanArray as ArrowBoolArray,
    Date32Array, Date64Array, PrimitiveArray as ArrowPrimitiveArray,
    StructArray as ArrowStructArray, Time32MillisecondArray, Time32SecondArray,
    Time64MicrosecondArray, Time64NanosecondArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
};
use arrow_buffer::ScalarBuffer;
use arrow_cast::cast;
use arrow_schema::{DataType, Field, FieldRef, Fields};
use itertools::Itertools;
use vortex_datetime_dtype::{is_temporal_ext_type, TemporalMetadata, TimeUnit};
use vortex_dtype::{DType, NativePType, PType};
use vortex_error::{vortex_bail, VortexError, VortexResult};

use crate::array::{
    varbinview_as_arrow, BoolArray, ExtensionArray, ListArray, NullArray, PrimitiveArray,
    StructArray, TemporalArray, VarBinViewArray,
};
use crate::arrow::{infer_data_type, FromArrowArray};
use crate::compute::try_cast;
use crate::encoding::Encoding;
use crate::stats::ArrayStatistics;
use crate::validity::ArrayValidity;
use crate::variants::{PrimitiveArrayTrait, StructArrayTrait};
use crate::{ArrayDType, ArrayData, ArrayLen, IntoArrayData};

/// The set of canonical array encodings, also the set of encodings that can be transferred to
/// Arrow with zero-copy.
///
/// Note that a canonical form is not recursive, i.e. a StructArray may contain non-canonical
/// child arrays, which may themselves need to be [canonicalized](IntoCanonical).
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
    /// Convert a canonical array into its equivalent [ArrayRef](Arrow array).
    ///
    /// Scalar arrays such as Bool and Primitive canonical arrays should convert with
    /// zero copies, while more complex variants such as Struct may require allocations if its child
    /// arrays require decompression.
    pub fn into_arrow(self) -> VortexResult<ArrayRef> {
        let data_type = infer_data_type(self.dtype())?;
        self.into_arrow_with_data_type(&data_type)
    }

    pub fn into_arrow_with_data_type(self, data_type: &DataType) -> VortexResult<ArrayRef> {
        match self {
            Canonical::Null(a) => null_to_arrow(a, data_type),
            Canonical::Bool(a) => bool_to_arrow(a, data_type),
            Canonical::Primitive(a) => primitive_to_arrow(a, data_type),
            Canonical::Struct(a) => struct_to_arrow(a, data_type),
            Canonical::List(a) => list_to_arrow(a, data_type),
            Canonical::VarBinView(a) => varbinview_to_arrow(&a, data_type),
            Canonical::Extension(a) => {
                if is_temporal_ext_type(a.id()) {
                    temporal_to_arrow(TemporalArray::try_from(a.into_array())?)
                } else {
                    // Convert storage array directly into arrow, losing type information
                    // that will let us round-trip.
                    // TODO(aduffy): https://github.com/spiraldb/vortex/issues/1167
                    a.storage().into_arrow_with_data_type(data_type)
                }
            }
        }
    }
}

impl Canonical {
    // Create an empty canonical array of the given dtype.
    pub fn empty(dtype: &DType) -> VortexResult<Canonical> {
        let arrow_dtype = infer_data_type(dtype)?;
        ArrayData::from_arrow(
            arrow_array::new_empty_array(&arrow_dtype),
            dtype.is_nullable(),
        )
        .into_canonical()
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

fn null_to_arrow(null_array: NullArray, data_type: &DataType) -> VortexResult<ArrayRef> {
    Ok(new_null_array(data_type, null_array.len()))
}

fn bool_to_arrow(bool_array: BoolArray, data_type: &DataType) -> VortexResult<ArrayRef> {
    debug_assert_eq!(data_type, &DataType::Boolean);
    Ok(Arc::new(ArrowBoolArray::new(
        bool_array.boolean_buffer(),
        bool_array.logical_validity()?.to_null_buffer()?,
    )))
}

fn primitive_to_arrow(
    primitive_array: PrimitiveArray,
    data_type: &DataType,
) -> VortexResult<ArrayRef> {
    fn as_arrow_array_primitive<T: ArrowPrimitiveType>(
        array: &PrimitiveArray,
        data_type: &DataType,
    ) -> VortexResult<Arc<ArrowPrimitiveArray<T>>> {
        debug_assert_eq!(data_type, &T::DATA_TYPE);
        Ok(Arc::new(ArrowPrimitiveArray::new(
            ScalarBuffer::<T::Native>::new(
                array.byte_buffer().clone().into_arrow_buffer(),
                0,
                array.len(),
            ),
            array.logical_validity()?.to_null_buffer()?,
        )))
    }

    Ok(match primitive_array.ptype() {
        // TODO(joe): verify that data_type is ignored not in debug
        PType::U8 => as_arrow_array_primitive::<UInt8Type>(&primitive_array, data_type)?,
        PType::U16 => as_arrow_array_primitive::<UInt16Type>(&primitive_array, data_type)?,
        PType::U32 => as_arrow_array_primitive::<UInt32Type>(&primitive_array, data_type)?,
        PType::U64 => as_arrow_array_primitive::<UInt64Type>(&primitive_array, data_type)?,
        PType::I8 => as_arrow_array_primitive::<Int8Type>(&primitive_array, data_type)?,
        PType::I16 => as_arrow_array_primitive::<Int16Type>(&primitive_array, data_type)?,
        PType::I32 => as_arrow_array_primitive::<Int32Type>(&primitive_array, data_type)?,
        PType::I64 => as_arrow_array_primitive::<Int64Type>(&primitive_array, data_type)?,
        PType::F16 => as_arrow_array_primitive::<Float16Type>(&primitive_array, data_type)?,
        PType::F32 => as_arrow_array_primitive::<Float32Type>(&primitive_array, data_type)?,
        PType::F64 => as_arrow_array_primitive::<Float64Type>(&primitive_array, data_type)?,
    })
}

fn struct_to_arrow(struct_array: StructArray, data_type: &DataType) -> VortexResult<ArrayRef> {
    let target_fields = match data_type {
        DataType::Struct(fields) => fields,
        _ => vortex_bail!("Expected DataType::Struct, got {:?}", data_type),
    };

    let field_arrays = target_fields
        .iter()
        .zip_eq(struct_array.children())
        .map(|(field, arr)| {
            arr.into_canonical()
                .map_err(|err| err.with_context(format!("Failed to canonicalize field {}", field)))
                .and_then(|c| c.into_arrow())
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let nulls = struct_array.logical_validity()?.to_null_buffer()?;

    if field_arrays.is_empty() {
        Ok(Arc::new(ArrowStructArray::new_empty_fields(
            struct_array.len(),
            nulls,
        )))
    } else {
        let arrow_fields = struct_array
            .names()
            .iter()
            .zip(field_arrays.iter())
            .zip(struct_array.dtypes().iter())
            .map(|((name, arrow_field), vortex_field)| {
                Field::new(
                    &**name,
                    arrow_field.data_type().clone(),
                    vortex_field.is_nullable(),
                )
            })
            .map(Arc::new)
            .collect::<Fields>();

        Ok(Arc::new(ArrowStructArray::try_new(
            arrow_fields,
            field_arrays,
            nulls,
        )?))
    }
}

pub(crate) fn varbinview_to_arrow(
    var_bin_view: &VarBinViewArray,
    data_type: &DataType,
) -> VortexResult<ArrayRef> {
    let arrow_arr = varbinview_as_arrow(var_bin_view);
    Ok(if arrow_arr.data_type() != data_type {
        cast(arrow_arr.as_ref(), data_type)?
    } else {
        arrow_arr
    })
}

// TODO(joe): unify with varbin
fn list_to_arrow(list: ListArray, data_type: &DataType) -> VortexResult<ArrayRef> {
    let offsets = list
        .offsets()
        .into_primitive()
        .map_err(|err| err.with_context("Failed to canonicalize offsets"))?;

    let (cast_ptype, element_dtype) = match data_type {
        DataType::List(field) => (PType::I32, field.data_type()),
        DataType::LargeList(field) => (PType::I64, field.data_type()),
        dt => vortex_bail!("list_to_arrow: unsupported data type: {:?}", dt),
    };

    let arrow_offsets = try_cast(offsets, cast_ptype.into())
        .map_err(|err| err.with_context("Failed to cast offsets to PrimitiveArray"))?
        .into_primitive()?;

    let values = list.elements().into_arrow_with_data_type(element_dtype)?;

    let field_ref = FieldRef::new(Field::new_list_field(
        values.data_type().clone(),
        list.validity().nullability().into(),
    ));

    let nulls = list.logical_validity()?.to_null_buffer()?;

    Ok(match arrow_offsets.ptype() {
        PType::I32 => Arc::new(arrow_array::ListArray::try_new(
            field_ref,
            arrow_offsets.buffer::<i32>().into_arrow_offset_buffer(),
            values,
            nulls,
        )?),
        PType::I64 => Arc::new(arrow_array::LargeListArray::try_new(
            field_ref,
            arrow_offsets.buffer::<i64>().into_arrow_offset_buffer(),
            values,
            nulls,
        )?),
        _ => vortex_bail!("Invalid offsets type {}", arrow_offsets.ptype()),
    })
}

fn temporal_to_arrow(temporal_array: TemporalArray) -> VortexResult<ArrayRef> {
    macro_rules! extract_temporal_values {
        ($values:expr, $prim:ty) => {{
            let temporal_values = try_cast(
                $values,
                &DType::Primitive(<$prim as NativePType>::PTYPE, $values.dtype().nullability()),
            )?
            .into_primitive()?;
            let nulls = temporal_values.logical_validity()?.to_null_buffer()?;
            let scalars = temporal_values.into_buffer().into_arrow_scalar_buffer();

            (scalars, nulls)
        }};
    }

    Ok(match temporal_array.temporal_metadata() {
        TemporalMetadata::Date(time_unit) => match time_unit {
            TimeUnit::D => {
                let (scalars, nulls) =
                    extract_temporal_values!(&temporal_array.temporal_values(), i32);
                Arc::new(Date32Array::new(scalars, nulls))
            }
            TimeUnit::Ms => {
                let (scalars, nulls) =
                    extract_temporal_values!(&temporal_array.temporal_values(), i64);
                Arc::new(Date64Array::new(scalars, nulls))
            }
            _ => vortex_bail!(
                "Invalid TimeUnit {time_unit} for {}",
                temporal_array.ext_dtype().id()
            ),
        },
        TemporalMetadata::Time(time_unit) => match time_unit {
            TimeUnit::S => {
                let (scalars, nulls) =
                    extract_temporal_values!(&temporal_array.temporal_values(), i32);
                Arc::new(Time32SecondArray::new(scalars, nulls))
            }
            TimeUnit::Ms => {
                let (scalars, nulls) =
                    extract_temporal_values!(&temporal_array.temporal_values(), i32);
                Arc::new(Time32MillisecondArray::new(scalars, nulls))
            }
            TimeUnit::Us => {
                let (scalars, nulls) =
                    extract_temporal_values!(&temporal_array.temporal_values(), i64);
                Arc::new(Time64MicrosecondArray::new(scalars, nulls))
            }
            TimeUnit::Ns => {
                let (scalars, nulls) =
                    extract_temporal_values!(&temporal_array.temporal_values(), i64);
                Arc::new(Time64NanosecondArray::new(scalars, nulls))
            }
            _ => vortex_bail!(
                "Invalid TimeUnit {time_unit} for {}",
                temporal_array.ext_dtype().id()
            ),
        },
        TemporalMetadata::Timestamp(time_unit, _) => {
            let (scalars, nulls) = extract_temporal_values!(&temporal_array.temporal_values(), i64);
            match time_unit {
                TimeUnit::Ns => Arc::new(TimestampNanosecondArray::new(scalars, nulls)),
                TimeUnit::Us => Arc::new(TimestampMicrosecondArray::new(scalars, nulls)),
                TimeUnit::Ms => Arc::new(TimestampMillisecondArray::new(scalars, nulls)),
                TimeUnit::S => Arc::new(TimestampSecondArray::new(scalars, nulls)),
                _ => vortex_bail!(
                    "Invalid TimeUnit {time_unit} for {}",
                    temporal_array.ext_dtype().id()
                ),
            }
        }
    })
}

/// Support trait for transmuting an array into the canonical encoding for its [vortex_dtype::DType].
///
/// This conversion ensures that the array's encoding matches one of the builtin canonical
/// encodings, each of which has a corresponding [Canonical] variant.
///
/// # Invariants
///
/// The DType of the array will be unchanged by canonicalization.
pub trait IntoCanonical {
    fn into_canonical(self) -> VortexResult<Canonical>;

    fn into_arrow(self) -> VortexResult<ArrayRef>
    where
        Self: Sized,
    {
        self.into_canonical()?.into_arrow()
    }

    fn into_arrow_with_data_type(self, data_type: &DataType) -> VortexResult<ArrayRef>
    where
        Self: Sized,
    {
        self.into_canonical()?.into_arrow_with_data_type(data_type)
    }
}

/// Encoding VTable for canonicalizing an array.
#[allow(clippy::wrong_self_convention)]
pub trait IntoCanonicalVTable {
    fn into_canonical(&self, array: ArrayData) -> VortexResult<Canonical>;

    fn into_arrow(&self, array: ArrayData) -> VortexResult<ArrayRef>;

    fn into_arrow_with_data_type(
        &self,
        array: ArrayData,
        data_type: &DataType,
    ) -> VortexResult<ArrayRef> {
        let arrow_array = self.into_arrow(array)?;
        if arrow_array.data_type() != data_type {
            Ok(cast(&arrow_array, data_type)?)
        } else {
            Ok(arrow_array)
        }
    }
}

/// Implement the [IntoCanonicalVTable] for all encodings with arrays implementing [IntoCanonical].
impl<E: Encoding> IntoCanonicalVTable for E
where
    E::Array: IntoCanonical,
    E::Array: TryFrom<ArrayData, Error = VortexError>,
{
    fn into_canonical(&self, data: ArrayData) -> VortexResult<Canonical> {
        #[cfg(feature = "canonical_counter")]
        data.inc_canonical_counter();
        let canonical = E::Array::try_from(data.clone())?.into_canonical()?;
        canonical.inherit_statistics(data.statistics());
        Ok(canonical)
    }

    fn into_arrow(&self, array: ArrayData) -> VortexResult<ArrayRef> {
        E::Array::try_from(array)?.into_arrow()
    }

    fn into_arrow_with_data_type(
        &self,
        array: ArrayData,
        data_type: &DataType,
    ) -> VortexResult<ArrayRef> {
        E::Array::try_from(array)?.into_arrow_with_data_type(data_type)
    }
}

/// Trait for types that can be converted from an owned type into an owned array variant.
///
/// # Canonicalization
///
/// This trait has a blanket implementation for all types implementing [IntoCanonical].
pub trait IntoArrayVariant {
    fn into_null(self) -> VortexResult<NullArray>;

    fn into_bool(self) -> VortexResult<BoolArray>;

    fn into_primitive(self) -> VortexResult<PrimitiveArray>;

    fn into_struct(self) -> VortexResult<StructArray>;

    fn into_list(self) -> VortexResult<ListArray>;

    fn into_varbinview(self) -> VortexResult<VarBinViewArray>;

    fn into_extension(self) -> VortexResult<ExtensionArray>;
}

impl<T> IntoArrayVariant for T
where
    T: IntoCanonical,
{
    fn into_null(self) -> VortexResult<NullArray> {
        self.into_canonical()?.into_null()
    }

    fn into_bool(self) -> VortexResult<BoolArray> {
        self.into_canonical()?.into_bool()
    }

    fn into_primitive(self) -> VortexResult<PrimitiveArray> {
        self.into_canonical()?.into_primitive()
    }

    fn into_struct(self) -> VortexResult<StructArray> {
        self.into_canonical()?.into_struct()
    }

    fn into_list(self) -> VortexResult<ListArray> {
        self.into_canonical()?.into_list()
    }

    fn into_varbinview(self) -> VortexResult<VarBinViewArray> {
        self.into_canonical()?.into_varbinview()
    }

    fn into_extension(self) -> VortexResult<ExtensionArray> {
        self.into_canonical()?.into_extension()
    }
}

/// IntoCanonical implementation for Array.
///
/// Canonicalizing an array requires potentially decompressing, so this requires a roundtrip through
/// the array's internal codec.
impl IntoCanonical for ArrayData {
    fn into_canonical(self) -> VortexResult<Canonical> {
        // We only care to know when we canonicalize something non-trivial.
        if !self.is_canonical() && self.len() > 1 {
            log::trace!("Canonicalizing array with encoding {:?}", self.encoding());
        }
        self.encoding().into_canonical(self)
    }

    fn into_arrow(self) -> VortexResult<ArrayRef>
    where
        Self: Sized,
    {
        self.encoding().into_arrow(self)
    }

    fn into_arrow_with_data_type(self, data_type: &DataType) -> VortexResult<ArrayRef> {
        self.encoding().into_arrow_with_data_type(self, data_type)
    }
}

/// This conversion is always "free" and should not touch underlying data. All it does is create an
/// owned pointer to the underlying concrete array type.
///
/// This combined with the above [IntoCanonical] impl for [ArrayData] allows simple two-way conversions
/// between arbitrary Vortex encodings and canonical Arrow-compatible encodings.
impl From<Canonical> for ArrayData {
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

impl AsRef<ArrayData> for Canonical {
    fn as_ref(&self) -> &ArrayData {
        match self {
            Canonical::Null(a) => a.as_ref(),
            Canonical::Bool(a) => a.as_ref(),
            Canonical::Primitive(a) => a.as_ref(),
            Canonical::Struct(a) => a.as_ref(),
            Canonical::List(a) => a.as_ref(),
            Canonical::VarBinView(a) => a.as_ref(),
            Canonical::Extension(a) => a.as_ref(),
        }
    }
}

impl IntoArrayData for Canonical {
    fn into_array(self) -> ArrayData {
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

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, Int64Type, UInt64Type};
    use arrow_array::{
        Array, ArrayRef, ListArray as ArrowListArray, PrimitiveArray as ArrowPrimitiveArray,
        StringArray, StringViewArray, StructArray as ArrowStructArray,
    };
    use arrow_buffer::{NullBufferBuilder, OffsetBuffer};
    use arrow_cast::cast;
    use arrow_schema::{DataType, Field};
    use vortex_buffer::buffer;

    use crate::array::{ConstantArray, StructArray};
    use crate::arrow::{infer_data_type, FromArrowArray};
    use crate::{ArrayDType, ArrayData, IntoArrayData, IntoCanonical};

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

        let data_type = infer_data_type(nested_struct_array.dtype()).unwrap();
        let arrow_struct = nested_struct_array
            .into_arrow_with_data_type(&data_type)
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

        let vortex_struct = ArrayData::from_arrow(&arrow_struct, true);
        let data_type = infer_data_type(vortex_struct.dtype()).unwrap();

        assert_eq!(
            &arrow_struct,
            vortex_struct
                .into_arrow_with_data_type(&data_type)
                .unwrap()
                .as_struct()
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

        let vortex_list = ArrayData::from_arrow(&arrow_list, true);

        let rt_arrow_list = cast(&vortex_list.into_arrow().unwrap(), list_data_type).unwrap();

        assert_eq!(
            (Arc::new(arrow_list.clone()) as ArrayRef).as_ref(),
            rt_arrow_list.as_ref()
        );
    }
}
