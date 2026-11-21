// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Estimates of the heap memory a [`DType`] holds, for caches that weigh what they keep.

use flatbuffers::ForwardsUOffset;
use flatbuffers::Vector;
use vortex_utils::aliases::hash_set::HashSet;

use crate::dtype::DType;
use crate::dtype::FieldDType;
use crate::dtype::FieldNames;
use crate::flatbuffers::dtype as fbd;

/// A materialized field: its name, its view over the dtype flatbuffer and the cached dtype.
/// Measured with a counting allocator.
const FIELD_BYTES: usize = 150;

impl DType {
    /// Estimates the heap bytes this dtype holds once every lazily parsed field is parsed,
    /// without parsing any, including the flatbuffers its lazy fields keep alive.
    ///
    /// Fields and flatbuffers that several parts of the dtype share are counted once.
    pub fn approx_resident_bytes(&self) -> usize {
        Footprint::default().dtype_bytes(self)
    }
}

#[derive(Default)]
struct Footprint {
    fields: HashSet<*const FieldDType>,
    flatbuffers: HashSet<*const u8>,
}

impl Footprint {
    fn dtype_bytes(&mut self, dtype: &DType) -> usize {
        match dtype {
            DType::Struct(fields, _) => self.fields_bytes(fields.names(), fields.raw_fields()),
            DType::Union(variants, _) => {
                self.fields_bytes(variants.names(), variants.raw_variants())
            }
            DType::List(element, _) | DType::FixedSizeList(element, ..) => {
                FIELD_BYTES.saturating_add(self.dtype_bytes(element))
            }
            DType::Map(map, _) => {
                let (key, value) = map.raw_key_value();
                self.field_bytes(key, 0)
                    .saturating_add(self.field_bytes(value, 0))
            }
            DType::Extension(ext) => {
                FIELD_BYTES.saturating_add(self.dtype_bytes(ext.storage_dtype()))
            }
            DType::Null
            | DType::Bool(_)
            | DType::Primitive(..)
            | DType::Decimal(..)
            | DType::Utf8(_)
            | DType::Binary(_)
            | DType::Variant(_) => 0,
        }
    }

    fn fields_bytes(&mut self, names: &FieldNames, fields: &[FieldDType]) -> usize {
        sum(names
            .iter()
            .zip(fields)
            .map(|(name, field)| self.field_bytes(field, name.as_ref().len())))
    }

    /// Charges a field and its name the first time the field is seen, and nothing after.
    fn field_bytes(&mut self, field: &FieldDType, name_bytes: usize) -> usize {
        if !self.fields.insert(std::ptr::from_ref(field)) {
            return 0;
        }
        let Some(view) = field.view() else {
            let owned = field.parsed().map_or(0, |dtype| self.dtype_bytes(dtype));
            return sum([FIELD_BYTES, name_bytes, owned]);
        };
        let flatbuffer = if self.flatbuffers.insert(view.buffer().as_ptr()) {
            view.buffer().len()
        } else {
            0
        };
        let nested = match field.parsed() {
            Some(dtype) => self.dtype_bytes(dtype),
            None => serialized_bytes(view.flatbuffer()),
        };
        sum([FIELD_BYTES, name_bytes, flatbuffer, nested])
    }
}

/// Estimates the fields nested in a serialized dtype as if they were parsed.
fn serialized_bytes(dtype: fbd::DType<'_>) -> usize {
    match dtype.type_type() {
        fbd::Type::Struct_ => dtype.type__as_struct_().map_or(0, |fields| {
            serialized_fields_bytes(fields.names(), fields.dtypes())
        }),
        fbd::Type::Union => dtype.type__as_union().map_or(0, |variants| {
            serialized_fields_bytes(variants.names(), variants.dtypes())
        }),
        fbd::Type::List => dtype
            .type__as_list()
            .and_then(|list| list.element_type())
            .map_or(0, serialized_child_bytes),
        fbd::Type::FixedSizeList => dtype
            .type__as_fixed_size_list()
            .and_then(|list| list.element_type())
            .map_or(0, serialized_child_bytes),
        fbd::Type::Map => dtype.type__as_map().map_or(0, |map| {
            map.key_type()
                .map_or(0, serialized_child_bytes)
                .saturating_add(map.value_type().map_or(0, serialized_child_bytes))
        }),
        fbd::Type::Extension => dtype
            .type__as_extension()
            .and_then(|ext| ext.storage_dtype())
            .map_or(0, serialized_child_bytes),
        _ => 0,
    }
}

fn serialized_child_bytes(dtype: fbd::DType<'_>) -> usize {
    FIELD_BYTES.saturating_add(serialized_bytes(dtype))
}

fn serialized_fields_bytes<'a>(
    names: Option<Vector<'a, ForwardsUOffset<&'a str>>>,
    dtypes: Option<Vector<'a, ForwardsUOffset<fbd::DType<'a>>>>,
) -> usize {
    sum(names
        .into_iter()
        .flatten()
        .zip(dtypes.into_iter().flatten())
        .map(|(name, dtype)| serialized_child_bytes(dtype).saturating_add(name.len())))
}

/// Sums without overflowing on malformed input.
fn sum(values: impl IntoIterator<Item = usize>) -> usize {
    values.into_iter().fold(0, usize::saturating_add)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;

    #[test]
    fn shared_supplied_fields_are_charged_once() {
        let mut dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        for _ in 0..30 {
            dtype = DType::Struct(
                StructFields::from_iter([("l", dtype.clone()), ("r", dtype)]),
                Nullability::NonNullable,
            );
        }
        // 30 levels of two field slots each; walking every path would visit 2^31 fields.
        let bytes = dtype.approx_resident_bytes();
        assert!(
            (60 * FIELD_BYTES..=60 * (FIELD_BYTES + 1)).contains(&bytes),
            "{bytes}"
        );
    }
}
