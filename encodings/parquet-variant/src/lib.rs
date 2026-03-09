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

use prost::Message;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::DeserializeMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::buffer::BufferHandle;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::variant_get::VariantGet;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
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
#[derive(Clone, prost::Message)]
pub struct ParquetVariantMetadata {
    /// Whether the un-shredded `value` child is present.
    #[prost(bool, tag = "1")]
    pub has_value: bool,
    /// Whether the shredded `typed_value` child is present.
    #[prost(bool, tag = "2")]
    pub has_typed_value: bool,
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
    metadata: ArrayRef,
    value: Option<ArrayRef>,
    typed_value: Option<ArrayRef>,
    stats_set: ArrayStats,
}

const VARIANT_DTYPE: DType = DType::Variant;

impl ParquetVariantArray {
    /// Creates a new ParquetVariantArray.
    ///
    /// # Panics
    /// Panics if neither `value` nor `typed_value` is provided, or if children have
    /// mismatched lengths.
    pub fn new(metadata: ArrayRef, value: Option<ArrayRef>, typed_value: Option<ArrayRef>) -> Self {
        assert!(
            value.is_some() || typed_value.is_some(),
            "at least one of value or typed_value must be present"
        );
        let len = metadata.len();
        if let Some(ref v) = value {
            assert_eq!(v.len(), len, "value length must match metadata length");
        }
        if let Some(ref tv) = typed_value {
            assert_eq!(
                tv.len(),
                len,
                "typed_value length must match metadata length"
            );
        }
        Self {
            metadata,
            value,
            typed_value,
            stats_set: ArrayStats::default(),
        }
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

    fn nchildren(&self) -> usize {
        1 + self.value.is_some() as usize + self.typed_value.is_some() as usize
    }
}

impl VTable for ParquetVariantVTable {
    type Array = ParquetVariantArray;
    type Metadata = ProstMetadata<ParquetVariantMetadata>;
    type OperationsVTable = NotSupported;
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
        if !array.metadata.array_eq(&other.metadata, precision) {
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
        match idx {
            0 => array.metadata.clone(),
            1 if array.value.is_some() => array.value.clone().unwrap(),
            1 => array.typed_value.clone().unwrap(),
            2 => array.typed_value.clone().unwrap(),
            _ => vortex_panic!("ParquetVariantArray child index {idx} out of bounds"),
        }
    }

    fn child_name(array: &ParquetVariantArray, idx: usize) -> String {
        match idx {
            0 => "metadata".to_string(),
            1 if array.value.is_some() => "value".to_string(),
            1 => "typed_value".to_string(),
            2 => "typed_value".to_string(),
            _ => vortex_panic!("ParquetVariantArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(array: &ParquetVariantArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(ParquetVariantMetadata {
            has_value: array.value.is_some(),
            has_typed_value: array.typed_value.is_some(),
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.encode_to_vec()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let inner =
            <ProstMetadata<ParquetVariantMetadata> as DeserializeMetadata>::deserialize(bytes)?;
        Ok(ProstMetadata(inner))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ParquetVariantArray> {
        vortex_ensure!(matches!(dtype, DType::Variant), "Expected Variant DType");
        vortex_ensure!(
            metadata.has_value || metadata.has_typed_value,
            "At least one of value or typed_value must be present"
        );

        let expected_children = 1 + metadata.has_value as usize + metadata.has_typed_value as usize;
        vortex_ensure!(
            children.len() == expected_children,
            "Expected {} children, got {}",
            expected_children,
            children.len()
        );

        let mut child_idx = 0;
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

        let typed_value = if metadata.has_typed_value {
            // typed_value can be any type — primitive, list, struct, etc.
            // We retrieve it without constraining its DType.
            let tv = children.get(child_idx, &DType::Variant, len)?;
            Some(tv)
        } else {
            None
        };

        Ok(ParquetVariantArray::new(
            variant_metadata,
            value,
            typed_value,
        ))
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == array.nchildren(),
            "ParquetVariantArray expects {} children, got {}",
            array.nchildren(),
            children.len()
        );
        let mut iter = children.into_iter();
        array.metadata = iter.next().unwrap();
        if array.value.is_some() {
            array.value = Some(iter.next().unwrap());
        }
        if array.typed_value.is_some() {
            array.typed_value = Some(iter.next().unwrap());
        }
        Ok(())
    }

    fn execute(array: &Self::Array, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(array.clone().into_array())
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
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
        match array.typed_value_array() {
            Some(typed_value) => {
                // The shredded typed_value is available; cast it to the requested dtype.
                Ok(Some(typed_value.cast(options.dtype.clone())?))
            }
            None => {
                // No shredded data available; cannot push down.
                Ok(None)
            }
        }
    }
}

impl ValidityVTable<ParquetVariantVTable> for ParquetVariantVTable {
    fn validity(_array: &ParquetVariantArray) -> VortexResult<Validity> {
        // Variant is always nullable. Null-ness of individual values is encoded
        // within the Parquet Variant binary format itself, not via a separate validity bitmap.
        Ok(Validity::AllValid)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::VariantArray;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::buffer;

    use super::*;

    #[test]
    fn test_variant_get_pushdown_with_typed_value() -> VortexResult<()> {
        // Create a ParquetVariantArray with shredded typed_value (i32 data)
        let metadata = buffer![0u8, 1, 2].into_array();
        let typed_value = buffer![10i32, 20, 30].into_array();
        let pv_array = ParquetVariantArray::new(metadata, None, Some(typed_value));

        // Wrap it in a VariantArray
        let variant_array = VariantArray::new(pv_array.into_array());

        // Apply variant_get
        let target_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let result = variant_array
            .into_array()
            .variant_get("col", target_dtype)?;

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
        let pv_array = ParquetVariantArray::new(metadata, Some(value), None);

        // Wrap it in a VariantArray
        let variant_array = VariantArray::new(pv_array.into_array());

        // Apply variant_get - the rule returns None since there's no typed_value,
        // so the optimizer creates a lazy ScalarFnArray that will error on execute.
        let target_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let result = variant_array
            .into_array()
            .variant_get("col", target_dtype)?;
        // The result is a lazy expression wrapping the variant array
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );
        Ok(())
    }
}
