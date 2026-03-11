// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This crate exposes a Vortex encoding that supports variant arrays, encoded as parquet's
//! [Variant encoding], in order to allow for zero-copy export to Arrow's new
//! [canonical extension type].
//!
//! The encoding follows the Arrow Parquet Variant canonical extension type structure:
//! - `metadata` (binary, required): type information for arrays/objects, field names and offsets
//! - `value` (binary, optional): un-shredded serialized variant values
//! - `typed_value` (any type, optional): shredded column data with a known type
//!
//! At least one of `value` or `typed_value` must be present. The `typed_value` child supports
//! full recursive shredding — it can be a primitive type, a list (whose elements are variant
//! nodes with value/typed_value), or a struct (whose fields are variant nodes).
//!
//! [Variant encoding]: https://parquet.apache.org/docs/file-format/types/variantencoding/
//! [canonical extension type]: https://arrow.apache.org/docs/format/CanonicalExtensions.html#parquet-variant

use std::hash::Hasher;

use arrow_array::Array as ArrowArray;
use parquet_variant::Variant as ParquetVariant;
use prost::Message;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionStep;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrow::FromArrowArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar::VariantValue;
use vortex_array::scalar_fn::fns::variant_get::VariantGet;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::validity_nchildren;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_proto::dtype as pb;
use vortex_session::VortexSession;

vtable!(ParquetVariant);

#[derive(Debug)]
pub struct ParquetVariantVTable;

impl ParquetVariantVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.parquet.variant");
}

/// Serialized metadata for a [`ParquetVariantArray`].
///
/// Tracks which optional children are present so the array can be correctly
/// reconstructed during deserialization.
#[derive(Clone, Debug)]
pub struct ParquetVariantMetadata {
    /// Whether the un-shredded `value` child is present.
    pub has_value: bool,
    /// DType of the shredded `typed_value`, if present.
    ///
    /// This is required to deserialize non-variant shredded children.
    pub typed_value_dtype: Option<DType>,
}

#[derive(Clone, prost::Message)]
struct ParquetVariantMetadataProto {
    /// Whether the un-shredded `value` child is present.
    #[prost(bool, tag = "1")]
    pub has_value: bool,
    /// DType of the shredded `typed_value`, if present.
    #[prost(message, optional, tag = "2")]
    pub typed_value_dtype: Option<pb::DType>,
}

/// An array encoding that stores variant data in the Parquet Variant binary format.
///
/// Contains up to three children following the Arrow Parquet Variant canonical extension type:
/// - `metadata` (always present): binary array with variant type information
/// - `value` (optional): binary array with un-shredded serialized variant values
/// - `typed_value` (optional): array of any type with shredded column data
///
/// At least one of `value` or `typed_value` must be present.
/// The `typed_value` supports full recursive shredding — it can be a primitive, list, or struct
/// where nested struct/list elements themselves contain value/typed_value children.
#[derive(Clone, Debug)]
pub struct ParquetVariantArray {
    validity: Validity,
    metadata: ArrayRef,
    value: Option<ArrayRef>,
    typed_value: Option<ArrayRef>,
    stats_set: ArrayStats,
}

const VARIANT_DTYPE: DType = DType::Variant;

impl ParquetVariantArray {
    /// Creates a new ParquetVariantArray.
    pub fn try_new(
        metadata: ArrayRef,
        value: Option<ArrayRef>,
        typed_value: Option<ArrayRef>,
    ) -> VortexResult<Self> {
        Self::try_new_with_validity(Validity::AllValid, metadata, value, typed_value)
    }

    /// Creates a new ParquetVariantArray with explicit parent validity.
    pub fn try_new_with_validity(
        validity: Validity,
        metadata: ArrayRef,
        value: Option<ArrayRef>,
        typed_value: Option<ArrayRef>,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            value.is_some() || typed_value.is_some(),
            "at least one of value or typed_value must be present"
        );
        let len = metadata.len();
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == len,
                "validity length must match metadata length"
            );
        }
        if let Some(ref v) = value {
            vortex_ensure!(v.len() == len, "value length must match metadata length");
        }
        if let Some(ref tv) = typed_value {
            vortex_ensure!(
                tv.len() == len,
                "typed_value length must match metadata length"
            );
        }
        Ok(Self {
            validity,
            metadata,
            value,
            typed_value,
            stats_set: ArrayStats::default(),
        })
    }

    /// Returns a reference to the metadata child array.
    pub fn metadata_array(&self) -> &ArrayRef {
        &self.metadata
    }

    /// Returns a reference to the un-shredded value child array, if present.
    pub fn value_array(&self) -> Option<&ArrayRef> {
        self.value.as_ref()
    }

    /// Returns a reference to the shredded typed_value child array, if present.
    pub fn typed_value_array(&self) -> Option<&ArrayRef> {
        self.typed_value.as_ref()
    }

    /// Returns the parent row validity for the variant storage struct.
    pub fn validity(&self) -> &Validity {
        &self.validity
    }

    /// Converts an Arrow `parquet_variant_compute::VariantArray` into a Vortex `ArrayRef`
    /// wrapping `VariantArray(ParquetVariantArray(...))`.
    pub fn from_arrow_variant(
        arrow_variant: &parquet_variant_compute::VariantArray,
    ) -> VortexResult<ArrayRef> {
        let storage = arrow_variant.inner();
        let value_nullable = storage
            .fields()
            .iter()
            .find(|field| field.name() == "value")
            .map(|field| field.is_nullable())
            .unwrap_or(false);
        let typed_value_nullable = storage
            .fields()
            .iter()
            .find(|field| field.name() == "typed_value")
            .map(|field| field.is_nullable())
            .unwrap_or(false);
        let validity = arrow_variant
            .nulls()
            .map(|nulls| {
                if nulls.null_count() == nulls.len() {
                    Validity::AllInvalid
                } else {
                    Validity::from(BitBuffer::from(nulls.inner().clone()))
                }
            })
            .unwrap_or(Validity::AllValid);
        let metadata =
            ArrayRef::from_arrow(arrow_variant.metadata_field() as &dyn ArrowArray, false)?;

        let value = arrow_variant
            .value_field()
            .map(|v| ArrayRef::from_arrow(v as &dyn ArrowArray, value_nullable))
            .transpose()?;

        let typed_value = arrow_variant
            .typed_value_field()
            .map(|tv| ArrayRef::from_arrow(tv.as_ref(), typed_value_nullable))
            .transpose()?;

        let pv =
            ParquetVariantArray::try_new_with_validity(validity, metadata, value, typed_value)?;
        Ok(VariantArray::new(pv.into_array()).into_array())
    }

    fn nchildren(&self) -> usize {
        validity_nchildren(&self.validity)
            + 1
            + self.value.is_some() as usize
            + self.typed_value.is_some() as usize
    }
}

impl VTable for ParquetVariantVTable {
    type Array = ParquetVariantArray;
    type Metadata = ParquetVariantMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn len(array: &ParquetVariantArray) -> usize {
        array.metadata.len()
    }

    fn dtype(_array: &ParquetVariantArray) -> &DType {
        &VARIANT_DTYPE
    }

    fn stats(array: &ParquetVariantArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &ParquetVariantArray, state: &mut H, precision: Precision) {
        array.validity.array_hash(state, precision);
        array.metadata.array_hash(state, precision);
        if let Some(ref value) = array.value {
            value.array_hash(state, precision);
        }
        if let Some(ref typed_value) = array.typed_value {
            typed_value.array_hash(state, precision);
        }
    }

    fn array_eq(
        array: &ParquetVariantArray,
        other: &ParquetVariantArray,
        precision: Precision,
    ) -> bool {
        if !array.validity.array_eq(&other.validity, precision)
            || !array.metadata.array_eq(&other.metadata, precision)
        {
            return false;
        }
        match (&array.value, &other.value) {
            (Some(a), Some(b)) => {
                if !a.array_eq(b, precision) {
                    return false;
                }
            }
            (None, None) => {}
            _ => return false,
        }
        match (&array.typed_value, &other.typed_value) {
            (Some(a), Some(b)) => a.array_eq(b, precision),
            (None, None) => true,
            _ => false,
        }
    }

    fn nbuffers(_array: &ParquetVariantArray) -> usize {
        0
    }

    fn buffer(_array: &ParquetVariantArray, idx: usize) -> BufferHandle {
        vortex_panic!("ParquetVariantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &ParquetVariantArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(array: &ParquetVariantArray) -> usize {
        array.nchildren()
    }

    fn child(array: &ParquetVariantArray, idx: usize) -> ArrayRef {
        let vc = validity_nchildren(&array.validity);
        if idx < vc {
            validity_to_child(&array.validity, array.metadata.len())
                .vortex_expect("ParquetVariantArray validity child out of bounds")
        } else {
            match idx - vc {
                0 => array.metadata.clone(),
                1 if array.value.is_some() => array
                    .value
                    .clone()
                    .vortex_expect("ParquetVariantArray missing value child"),
                1 => array
                    .typed_value
                    .clone()
                    .vortex_expect("ParquetVariantArray missing typed_value child"),
                2 => array
                    .typed_value
                    .clone()
                    .vortex_expect("ParquetVariantArray missing typed_value child"),
                _ => vortex_panic!("ParquetVariantArray child index {idx} out of bounds"),
            }
        }
    }

    fn child_name(array: &ParquetVariantArray, idx: usize) -> String {
        let vc = validity_nchildren(&array.validity);
        match idx {
            idx if idx < vc => "validity".to_string(),
            idx => match idx - vc {
                0 => "metadata".to_string(),
                1 if array.value.is_some() => "value".to_string(),
                1 => "typed_value".to_string(),
                2 => "typed_value".to_string(),
                _ => vortex_panic!("ParquetVariantArray child_name index {idx} out of bounds"),
            },
        }
    }

    fn metadata(array: &ParquetVariantArray) -> VortexResult<Self::Metadata> {
        Ok(ParquetVariantMetadata {
            has_value: array.value.is_some(),
            typed_value_dtype: array.typed_value.as_ref().map(|tv| tv.dtype().clone()),
        })
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        let typed_value_dtype = metadata
            .typed_value_dtype
            .as_ref()
            .map(|dtype| dtype.try_into())
            .transpose()?;
        Ok(Some(
            ParquetVariantMetadataProto {
                has_value: metadata.has_value,
                typed_value_dtype,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let proto = ParquetVariantMetadataProto::decode(bytes)?;
        let typed_value_dtype = match proto.typed_value_dtype.as_ref() {
            Some(dtype) => Some(DType::from_proto(dtype, _session)?),
            None => None,
        };
        Ok(ParquetVariantMetadata {
            has_value: proto.has_value,
            typed_value_dtype,
        })
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ParquetVariantArray> {
        vortex_ensure!(matches!(dtype, DType::Variant), "Expected Variant DType");
        let has_typed_value = metadata.typed_value_dtype.is_some();
        vortex_ensure!(
            metadata.has_value || has_typed_value,
            "At least one of value or typed_value must be present"
        );

        let expected_children = 1 + metadata.has_value as usize + has_typed_value as usize;
        vortex_ensure!(
            children.len() == expected_children || children.len() == expected_children + 1,
            "Expected {} or {} children, got {}",
            expected_children,
            expected_children + 1,
            children.len()
        );

        let (validity, mut child_idx) = if children.len() == expected_children {
            (Validity::AllValid, 0)
        } else {
            (Validity::Array(children.get(0, &Validity::DTYPE, len)?), 1)
        };
        let variant_metadata =
            children.get(child_idx, &DType::Binary(Nullability::NonNullable), len)?;
        child_idx += 1;

        let value = if metadata.has_value {
            let v = children.get(child_idx, &DType::Binary(Nullability::NonNullable), len)?;
            child_idx += 1;
            Some(v)
        } else {
            None
        };

        let typed_value = if has_typed_value {
            // typed_value can be any type — primitive, list, struct, etc.
            let dtype = metadata
                .typed_value_dtype
                .clone()
                .ok_or_else(|| vortex_err!("typed_value_dtype missing for typed_value child"))?;
            let tv = children.get(child_idx, &dtype, len)?;
            Some(tv)
        } else {
            None
        };

        ParquetVariantArray::try_new_with_validity(validity, variant_metadata, value, typed_value)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == array.nchildren(),
            "ParquetVariantArray expects {} children, got {}",
            array.nchildren(),
            children.len()
        );
        let mut iter = children.into_iter();
        if validity_nchildren(&array.validity) == 1 {
            array.validity = Validity::Array(
                iter.next()
                    .vortex_expect("ParquetVariantArray missing validity child"),
            );
        }
        array.metadata = iter
            .next()
            .vortex_expect("ParquetVariantArray missing metadata child");
        if array.value.is_some() {
            array.value = Some(
                iter.next()
                    .vortex_expect("ParquetVariantArray missing value child in with_children"),
            );
        }
        if array.typed_value.is_some() {
            array.typed_value =
                Some(iter.next().vortex_expect(
                    "ParquetVariantArray missing typed_value child in with_children",
                ));
        }
        Ok(())
    }

    fn execute(array: &Self::Array, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        Ok(ExecutionStep::done(array.clone().into_array()))
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

fn scalar_to_variant_value(scalar: Scalar) -> VortexResult<VariantValue> {
    if scalar.is_null() {
        return Ok(VariantValue::Null);
    }

    Ok(match scalar.dtype() {
        DType::Null => VariantValue::Null,
        DType::Bool(_) => VariantValue::Bool(scalar.as_bool().value().unwrap_or(false)),
        DType::Primitive(..) => VariantValue::Primitive(
            *scalar
                .value()
                .vortex_expect("non-null primitive scalar must have a value")
                .as_primitive(),
        ),
        DType::Decimal(..) => VariantValue::Decimal(
            *scalar
                .value()
                .vortex_expect("non-null decimal scalar must have a value")
                .as_decimal(),
        ),
        DType::Utf8(_) => VariantValue::Utf8(
            scalar
                .value()
                .vortex_expect("non-null utf8 scalar must have a value")
                .as_utf8()
                .clone(),
        ),
        DType::Binary(_) => VariantValue::Binary(
            scalar
                .value()
                .vortex_expect("non-null binary scalar must have a value")
                .as_binary()
                .clone(),
        ),
        DType::List(..) | DType::FixedSizeList(..) => VariantValue::List(
            scalar
                .as_list()
                .elements()
                .unwrap_or_default()
                .into_iter()
                .map(scalar_to_variant_value)
                .collect::<VortexResult<Vec<_>>>()?,
        ),
        DType::Struct(fields, _) => VariantValue::Object(
            fields
                .names()
                .iter()
                .cloned()
                .zip(
                    scalar
                        .as_struct()
                        .fields_iter()
                        .vortex_expect("non-null struct scalar must have field values"),
                )
                .map(|(name, field)| Ok((name.as_ref().into(), scalar_to_variant_value(field)?)))
                .collect::<VortexResult<Vec<_>>>()?,
        ),
        DType::Extension(_) => VariantValue::Utf8(scalar.to_string().into()),
        DType::Variant => scalar
            .value()
            .vortex_expect("non-null variant scalar must have a value")
            .as_variant()
            .clone(),
    })
}

fn parquet_variant_to_variant_value(variant: ParquetVariant<'_, '_>) -> VortexResult<VariantValue> {
    Ok(match variant {
        ParquetVariant::Null => VariantValue::Null,
        ParquetVariant::Int8(v) => VariantValue::Primitive(v.into()),
        ParquetVariant::Int16(v) => VariantValue::Primitive(v.into()),
        ParquetVariant::Int32(v) => VariantValue::Primitive(v.into()),
        ParquetVariant::Int64(v) => VariantValue::Primitive(v.into()),
        ParquetVariant::Float(v) => VariantValue::Primitive(v.into()),
        ParquetVariant::Double(v) => VariantValue::Primitive(v.into()),
        ParquetVariant::BooleanTrue => VariantValue::Bool(true),
        ParquetVariant::BooleanFalse => VariantValue::Bool(false),
        ParquetVariant::Decimal4(v) => VariantValue::Decimal(v.integer().into()),
        ParquetVariant::Decimal8(v) => VariantValue::Decimal(v.integer().into()),
        ParquetVariant::Decimal16(v) => VariantValue::Decimal(v.integer().into()),
        ParquetVariant::Binary(v) => VariantValue::Binary(v.to_vec().into()),
        ParquetVariant::String(v) => VariantValue::Utf8(v.into()),
        ParquetVariant::ShortString(v) => VariantValue::Utf8(v.as_str().into()),
        ParquetVariant::Date(v) => VariantValue::Utf8(v.to_string().into()),
        ParquetVariant::TimestampMicros(v) => VariantValue::Utf8(v.to_rfc3339().into()),
        ParquetVariant::TimestampNtzMicros(v) => VariantValue::Utf8(v.to_string().into()),
        ParquetVariant::TimestampNanos(v) => VariantValue::Utf8(v.to_rfc3339().into()),
        ParquetVariant::TimestampNtzNanos(v) => VariantValue::Utf8(v.to_string().into()),
        ParquetVariant::Time(v) => VariantValue::Utf8(v.to_string().into()),
        ParquetVariant::Uuid(v) => VariantValue::Utf8(v.to_string().into()),
        ParquetVariant::List(values) => VariantValue::List(
            values
                .iter()
                .map(parquet_variant_to_variant_value)
                .collect::<VortexResult<Vec<_>>>()?,
        ),
        ParquetVariant::Object(values) => VariantValue::Object(
            values
                .iter()
                .map(|(name, value)| Ok((name.into(), parquet_variant_to_variant_value(value)?)))
                .collect::<VortexResult<Vec<_>>>()?,
        ),
    })
}

impl OperationsVTable<ParquetVariantVTable> for ParquetVariantVTable {
    fn scalar_at(array: &ParquetVariantArray, index: usize) -> VortexResult<Scalar> {
        if array.validity.is_null(index)? {
            return Ok(Scalar::null(DType::Variant));
        }

        let value = if let Some(typed_value) = array.typed_value_array()
            && typed_value.is_valid(index)?
        {
            scalar_to_variant_value(typed_value.scalar_at(index)?)?
        } else if let Some(value) = array.value_array()
            && value.is_valid(index)?
        {
            let metadata = array
                .metadata_array()
                .scalar_at(index)?
                .as_binary()
                .value()
                .cloned()
                .vortex_expect("non-null metadata row must have binary value");
            let value = value
                .scalar_at(index)?
                .as_binary()
                .value()
                .cloned()
                .vortex_expect("non-null value row must have binary value");
            parquet_variant_to_variant_value(ParquetVariant::try_new(
                metadata.as_ref(),
                value.as_ref(),
            )?)?
        } else {
            VariantValue::Null
        };

        Scalar::try_new(DType::Variant, Some(ScalarValue::Variant(value)))
    }
}

const PARENT_RULES: ParentRuleSet<ParquetVariantVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&ParquetVariantGetRule)]);

/// Rule to handle VariantGet on a ParquetVariantArray by returning the typed_value child.
#[derive(Debug)]
struct ParquetVariantGetRule;

impl ArrayParentReduceRule<ParquetVariantVTable> for ParquetVariantGetRule {
    type Parent = ExactScalarFn<VariantGet>;

    fn reduce_parent(
        &self,
        array: &ParquetVariantArray,
        parent: ScalarFnArrayView<'_, VariantGet>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let options = parent.options;
        if options.path().is_some_and(|p| !p.is_empty()) {
            vortex_bail!("ParquetVariant VariantGet only supports empty path");
        }
        let target_dtype = options.dtype().with_nullability(Nullability::Nullable);
        match array.typed_value_array() {
            Some(typed_value)
                if typed_value.dtype().with_nullability(Nullability::Nullable) == target_dtype =>
            {
                // The shredded typed_value matches the requested type.
                // Cast to ensure nullability matches (VariantGet always returns nullable).
                Ok(Some(typed_value.cast(target_dtype)?))
            }
            _ => {
                // No shredded data or type mismatch; cannot push down.
                Ok(None)
            }
        }
    }
}

impl ValidityVTable<ParquetVariantVTable> for ParquetVariantVTable {
    fn validity(array: &ParquetVariantArray) -> VortexResult<Validity> {
        Ok(array.validity.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::Int32Array;
    use arrow_array::StructArray;
    use arrow_array::builder::BinaryViewBuilder;
    use arrow_array::cast::AsArray;
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Fields;
    use parquet_variant::Variant;
    use parquet_variant_compute::VariantArray as ArrowVariantArray;
    use parquet_variant_compute::VariantArrayBuilder;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::Precision;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::VariantVTable;
    use vortex_array::arrow::ArrowArrayExecutor;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::serde::ArrayParts;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::*;

    #[test]
    fn test_from_arrow_variant_basic() -> VortexResult<()> {
        let mut builder = VariantArrayBuilder::new(3);
        builder.append_variant(Variant::from(42i32));
        builder.append_variant(Variant::from("hello"));
        builder.append_variant(Variant::from(true));
        let arrow_variant = builder.build();

        let vortex_arr = ParquetVariantArray::from_arrow_variant(&arrow_variant)?;

        assert_eq!(vortex_arr.len(), 3);
        assert_eq!(vortex_arr.dtype(), &DType::Variant);

        Ok(())
    }

    #[test]
    fn test_from_arrow_variant_with_shredded_typed_value() -> VortexResult<()> {
        // Build the underlying StructArray with metadata + typed_value fields
        let mut metadata_builder = BinaryViewBuilder::new();
        // Minimal variant metadata: version 1, no dictionary
        let min_metadata = [1u8, 0];
        for _ in 0..3 {
            metadata_builder.append_value(min_metadata);
        }
        let metadata = metadata_builder.finish();

        let typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30]));

        let struct_fields: Fields = vec![
            Arc::new(Field::new("metadata", DataType::BinaryView, false)),
            Arc::new(Field::new("typed_value", DataType::Int32, false)),
        ]
        .into();
        let struct_array =
            StructArray::try_new(struct_fields, vec![Arc::new(metadata), typed_value], None)
                .unwrap();

        let arrow_variant = ArrowVariantArray::try_new(&struct_array).unwrap();

        let vortex_arr = ParquetVariantArray::from_arrow_variant(&arrow_variant)?;
        assert_eq!(vortex_arr.len(), 3);
        assert_eq!(vortex_arr.dtype(), &DType::Variant);

        // Verify typed_value is present by downcasting through the layers
        let variant_arr = vortex_arr.as_opt::<VariantVTable>().unwrap();
        let inner = variant_arr
            .child()
            .as_opt::<ParquetVariantVTable>()
            .unwrap();
        assert!(inner.typed_value_array().is_some());

        Ok(())
    }

    #[test]
    fn test_variant_get_pushdown_with_typed_value() -> VortexResult<()> {
        // Create a ParquetVariantArray with shredded typed_value (i32 data)
        let metadata = buffer![0u8, 1, 2].into_array();
        let typed_value = buffer![10i32, 20, 30].into_array();
        let pv_array = ParquetVariantArray::try_new(metadata, None, Some(typed_value))?;

        // Wrap it in a VariantArray
        let variant_array = VariantArray::new(pv_array.into_array());

        // Apply variant_get
        let target_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let result = variant_array.into_array().variant_get(None, target_dtype)?;

        // The result should be the typed_value data, cast to nullable i32
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );
        assert_eq!(result.len(), 3);

        Ok(())
    }

    #[test]
    fn test_variant_get_no_typed_value() -> VortexResult<()> {
        // Create a ParquetVariantArray without typed_value (only value)
        let metadata = buffer![0u8, 1, 2].into_array();
        let value = buffer![0u8, 1, 2].into_array();
        let pv_array = ParquetVariantArray::try_new(metadata, Some(value), None)?;

        // Wrap it in a VariantArray
        let variant_array = VariantArray::new(pv_array.into_array());

        // Apply variant_get - the rule returns None since there's no typed_value,
        // so the optimizer creates a lazy ScalarFnArray that will error on execute.
        let target_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let result = variant_array.into_array().variant_get(None, target_dtype)?;
        // The result is a lazy expression wrapping the variant array
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );
        Ok(())
    }

    fn roundtrip(array: ArrayRef) -> ArrayRef {
        let dtype = array.dtype().clone();
        let len = array.len();

        let ctx = ArrayContext::empty();
        let serialized = array.serialize(&ctx, &SerializeOptions::default()).unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let session = VortexSession::empty().with::<vortex_array::session::ArraySession>();
        session
            .arrays()
            .register(ParquetVariantVTable::ID, ParquetVariantVTable);
        session.arrays().register(VariantVTable::ID, VariantVTable);

        let parts = ArrayParts::try_from(concat).unwrap();
        parts
            .decode(&dtype, len, &ReadContext::new(ctx.to_ids()), &session)
            .unwrap()
    }

    fn assert_arrow_variant_storage_roundtrip(struct_array: StructArray) -> VortexResult<()> {
        let arrow_variant = ArrowVariantArray::try_new(&struct_array).unwrap();
        let vortex_arr = ParquetVariantArray::from_arrow_variant(&arrow_variant)?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let roundtripped = vortex_arr.execute_arrow(None, &mut ctx)?;
        let roundtripped = roundtripped.as_struct();

        assert_eq!(struct_array.len(), roundtripped.len());
        assert_eq!(struct_array.column_names(), roundtripped.column_names());
        assert_eq!(struct_array.nulls(), roundtripped.nulls());
        assert_eq!(struct_array.fields().len(), roundtripped.fields().len());

        for (expected, actual) in struct_array
            .fields()
            .iter()
            .zip(roundtripped.fields().iter())
        {
            assert_eq!(expected.name(), actual.name());
            assert_eq!(expected.data_type(), actual.data_type());
            assert_eq!(expected.is_nullable(), actual.is_nullable());
        }

        for (expected, actual) in struct_array
            .columns()
            .iter()
            .zip(roundtripped.columns().iter())
        {
            assert_eq!(expected.to_data(), actual.to_data());
        }

        Ok(())
    }

    fn binary_view_array<const N: usize>(values: [&[u8]; N]) -> ArrowArrayRef {
        let mut builder = BinaryViewBuilder::new();
        for value in values {
            builder.append_value(value);
        }
        Arc::new(builder.finish())
    }

    #[test]
    fn test_serde_roundtrip_typed_value_variant() {
        let outer_metadata =
            VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00", b"\x01\x00"]).into_array();

        let inner_metadata =
            VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00", b"\x01\x00"]).into_array();
        let inner_value = VarBinViewArray::from_iter_bin([b"\x02", b"\x03", b"\x04"]).into_array();
        let inner_pv =
            ParquetVariantArray::try_new(inner_metadata, Some(inner_value), None).unwrap();
        let typed_value = VariantArray::new(inner_pv.into_array()).into_array();

        let outer_pv =
            ParquetVariantArray::try_new(outer_metadata, None, Some(typed_value)).unwrap();
        let array = outer_pv.into_array();
        let decoded = roundtrip(array.clone());

        assert!(array.array_eq(&decoded, Precision::Value));
        let decoded_pv = decoded.as_opt::<ParquetVariantVTable>().unwrap();
        let typed = decoded_pv.typed_value_array().unwrap();
        assert_eq!(typed.dtype(), &DType::Variant);
    }

    #[test]
    fn test_serde_roundtrip_typed_value_int32() {
        let outer_metadata =
            VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00", b"\x01\x00"]).into_array();
        let typed_value = buffer![10i32, 20, 30].into_array();

        let outer_pv =
            ParquetVariantArray::try_new(outer_metadata, None, Some(typed_value)).unwrap();
        let array = outer_pv.into_array();
        let decoded = roundtrip(array.clone());

        assert!(array.array_eq(&decoded, Precision::Value));
        let decoded_pv = decoded.as_opt::<ParquetVariantVTable>().unwrap();
        let typed = decoded_pv.typed_value_array().unwrap();
        assert_eq!(
            typed.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_arrow_variant_storage_basic() -> VortexResult<()> {
        let metadata = VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00"]).into_array();
        let value = VarBinViewArray::from_iter_bin([b"\x10", b"\x11"]).into_array();
        let pv_array = ParquetVariantArray::try_new(metadata, Some(value), None)?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let arrow = pv_array.into_array().execute_arrow(None, &mut ctx)?;
        let struct_arr = arrow.as_struct();

        assert_eq!(struct_arr.num_columns(), 2);
        assert_eq!(struct_arr.column_names(), &["metadata", "value"]);

        Ok(())
    }

    #[test]
    fn test_arrow_variant_storage_with_typed_value() -> VortexResult<()> {
        let metadata = VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00"]).into_array();
        let value = VarBinViewArray::from_iter_bin([b"\x10", b"\x11"]).into_array();
        let typed_value = buffer![1i32, 2].into_array();
        let pv_array = ParquetVariantArray::try_new(metadata, Some(value), Some(typed_value))?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let arrow = pv_array.into_array().execute_arrow(None, &mut ctx)?;
        let struct_arr = arrow.as_struct();

        assert_eq!(struct_arr.num_columns(), 3);
        assert_eq!(
            struct_arr.column_names(),
            &["metadata", "value", "typed_value"]
        );

        Ok(())
    }

    #[test]
    fn test_arrow_variant_roundtrip_unshredded_storage() -> VortexResult<()> {
        let mut builder = VariantArrayBuilder::new(3);
        builder.append_variant(Variant::from(42i32));
        builder.append_variant(Variant::from("hello"));
        builder.append_variant(Variant::from(true));

        assert_arrow_variant_storage_roundtrip(builder.build().into_inner())
    }

    #[test]
    fn test_arrow_variant_roundtrip_typed_value_only_storage() -> VortexResult<()> {
        let metadata = binary_view_array([b"\x01\x00", b"\x01\x00", b"\x01\x00"]);
        let typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30]));

        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("typed_value", DataType::Int32, false)),
            ]
            .into(),
            vec![metadata, typed_value],
            None,
        )
        .unwrap();

        assert_arrow_variant_storage_roundtrip(struct_array)
    }

    #[test]
    fn test_arrow_variant_roundtrip_value_and_typed_value_storage() -> VortexResult<()> {
        let metadata = binary_view_array([b"\x01\x00", b"\x01\x00"]);
        let value = binary_view_array([b"\x10", b"\x11"]);
        let typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![1, 2]));

        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("value", DataType::BinaryView, true)),
                Arc::new(Field::new("typed_value", DataType::Int32, false)),
            ]
            .into(),
            vec![metadata, value, typed_value],
            None,
        )
        .unwrap();

        assert_arrow_variant_storage_roundtrip(struct_array)
    }

    #[test]
    fn test_arrow_variant_roundtrip_with_outer_nulls() -> VortexResult<()> {
        let metadata = binary_view_array([b"\x01\x00", b"\x01\x00", b"\x01\x00"]);
        let value = binary_view_array([b"\x10", b"\x00", b"\x11"]);
        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("value", DataType::BinaryView, true)),
            ]
            .into(),
            vec![metadata, value],
            Some(NullBuffer::from(vec![true, false, true])),
        )
        .unwrap();

        assert_arrow_variant_storage_roundtrip(struct_array)
    }
}
