// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Estimates retained heap storage for DType graphs.
//!
//! Lazy fields are projected from their wire representation without resolving them. Sharing is
//! deduplicated within one graph, but not across separate estimates; allocator bookkeeping, Arc
//! control blocks, and hidden extension allocations are omitted.

use std::mem::size_of;
use std::sync::Arc;
use std::sync::OnceLock;

use vortex_utils::aliases::hash_set::HashSet;

use crate::dtype::DType;
use crate::dtype::FieldDType;
use crate::dtype::FieldName;
use crate::dtype::MapDType;
use crate::dtype::StructFields;
use crate::dtype::UnionVariants;
use crate::dtype::serde::flatbuffers::ViewedDType;
use crate::flatbuffers::dtype as fbd;

impl DType {
    /// Estimates the heap bytes owned by this dtype, projecting lazy fields from their wire
    /// representation without resolving them.
    ///
    /// Inline storage for the root dtype is owned by its containing object and is not included.
    /// This estimate omits allocator bookkeeping, Arc control blocks, opaque extension heap
    /// allocations, and sharing with other footers.
    pub fn approx_resident_bytes(&self) -> usize {
        DTypeFootprint::default().dtype(self)
    }
}

#[derive(Default)]
struct DTypeFootprint {
    owners: HashSet<*const ()>,
    fields: HashSet<*const ()>,
    list_dtypes: HashSet<*const DType>,
    extensions: HashSet<*const ()>,
    buffers: HashSet<*const ()>,
    names: HashSet<*const ()>,
}

impl DTypeFootprint {
    fn dtype(&mut self, dtype: &DType) -> usize {
        match dtype {
            DType::Struct(fields, _) => self.struct_fields(fields),
            DType::Union(variants, _) => self.union_variants(variants),
            DType::List(element, _) | DType::FixedSizeList(element, ..) => {
                if self.list_dtypes.insert(Arc::as_ptr(element)) {
                    size_of::<DType>().saturating_add(self.dtype(element))
                } else {
                    0
                }
            }
            DType::Map(map, _) => self.map_fields(map),
            DType::Extension(extension) => {
                if self.extensions.insert(extension.approx_footprint_id()) {
                    extension
                        .approx_inline_bytes()
                        .saturating_add(self.dtype(extension.storage_dtype()))
                } else {
                    0
                }
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

    fn struct_fields(&mut self, fields: &StructFields) -> usize {
        if !self.owners.insert(fields.footprint_id()) {
            return 0;
        }

        let mut bytes = fields.footprint_bytes();
        for name in fields.names().iter() {
            bytes = bytes.saturating_add(self.name(name));
        }
        for field in fields.raw_fields() {
            bytes = bytes.saturating_add(self.field(field));
        }
        bytes
    }

    fn union_variants(&mut self, variants: &UnionVariants) -> usize {
        if !self.owners.insert(variants.footprint_id()) {
            return 0;
        }

        let mut bytes = variants.footprint_bytes();
        for name in variants.names().iter() {
            bytes = bytes.saturating_add(self.name(name));
        }
        for variant in variants.raw_variants() {
            bytes = bytes.saturating_add(self.field(variant));
        }
        bytes
    }

    fn map_fields(&mut self, map: &MapDType) -> usize {
        if !self.owners.insert(map.footprint_id()) {
            return 0;
        }

        let (key, value) = map.raw_key_value();
        map.footprint_bytes()
            .saturating_add(self.field(key))
            .saturating_add(self.field(value))
    }

    fn field(&mut self, field: &FieldDType) -> usize {
        if !self.fields.insert(field.footprint_id()) {
            return 0;
        }

        if let Some(view) = field.view() {
            let mut bytes = size_of::<OnceLock<DType>>();
            let buffer = view.buffer();
            if self.buffers.insert(buffer.as_ref().as_ptr().cast()) {
                bytes = bytes.saturating_add(buffer.len());
            }
            let nested = field
                .parsed()
                .map(|dtype| self.dtype(dtype))
                .unwrap_or_else(|| self.serialized_dtype(view.flatbuffer()));
            bytes.saturating_add(nested)
        } else {
            field.parsed().map_or(0, |dtype| self.dtype(dtype))
        }
    }

    fn serialized_dtype(&mut self, dtype: fbd::DType<'_>) -> usize {
        match dtype.type_type() {
            fbd::Type::Struct_ => {
                let Some(fields) = dtype.type__as_struct_() else {
                    return 0;
                };
                let names = fields.names();
                let dtypes = fields.dtypes();
                let count = names
                    .as_ref()
                    .map_or(0, |names| names.len())
                    .min(dtypes.as_ref().map_or(0, |dtypes| dtypes.len()));
                let mut bytes = StructFields::projected_footprint_bytes(count);
                if let Some(names) = names {
                    for name in names.iter().take(count) {
                        bytes = bytes.saturating_add(self.serialized_name(name));
                    }
                }
                if let Some(dtypes) = dtypes {
                    for child in dtypes.iter().take(count) {
                        bytes = bytes
                            .saturating_add(size_of::<OnceLock<DType>>())
                            .saturating_add(self.serialized_dtype(child));
                    }
                }
                bytes
            }
            fbd::Type::Union => {
                let Some(variants) = dtype.type__as_union() else {
                    return 0;
                };
                let names = variants.names();
                let dtypes = variants.dtypes();
                let count = names
                    .as_ref()
                    .map_or(0, |names| names.len())
                    .min(dtypes.as_ref().map_or(0, |dtypes| dtypes.len()));
                let mut bytes = UnionVariants::projected_footprint_bytes(count);
                if let Some(names) = names {
                    for name in names.iter().take(count) {
                        bytes = bytes.saturating_add(self.serialized_name(name));
                    }
                }
                if let Some(dtypes) = dtypes {
                    for child in dtypes.iter().take(count) {
                        bytes = bytes
                            .saturating_add(size_of::<OnceLock<DType>>())
                            .saturating_add(self.serialized_dtype(child));
                    }
                }
                bytes
            }
            fbd::Type::List => dtype
                .type__as_list()
                .and_then(|list| list.element_type())
                .map_or(0, |element| {
                    size_of::<DType>().saturating_add(self.serialized_dtype(element))
                }),
            fbd::Type::FixedSizeList => dtype
                .type__as_fixed_size_list()
                .and_then(|list| list.element_type())
                .map_or(0, |element| {
                    size_of::<DType>().saturating_add(self.serialized_dtype(element))
                }),
            fbd::Type::Map => dtype.type__as_map().map_or(0, |map| {
                MapDType::projected_footprint_bytes()
                    .saturating_add(size_of::<OnceLock<DType>>().saturating_mul(2))
                    .saturating_add(
                        map.key_type()
                            .map_or(0, |key| self.serialized_dtype(key)),
                    )
                    .saturating_add(
                        map.value_type()
                            .map_or(0, |value| self.serialized_dtype(value)),
                    )
            }),
            fbd::Type::Extension => dtype.type__as_extension().map_or(0, |extension| {
                let metadata_bytes = extension.metadata().map_or(0, |metadata| metadata.len());
                size_of::<DType>()
                    .saturating_add(metadata_bytes)
                    .saturating_add(
                        extension
                            .storage_dtype()
                            .map_or(0, |storage| self.serialized_dtype(storage)),
                    )
            }),
            _ => 0,
        }
    }

    fn name(&mut self, name: &FieldName) -> usize {
        let identity = Arc::as_ptr(name.inner()) as *const ();
        self.names
            .insert(identity)
            .then_some(name.as_ref().len())
            .unwrap_or_default()
    }

    fn serialized_name(&mut self, name: &str) -> usize {
        name.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::dtype::extension::ExtId;
    use crate::dtype::extension::ForeignExtDType;
    use crate::flatbuffers::WriteFlatBufferExt;
    use vortex_session::VortexSession;

    #[test]
    fn test_shared_dtype_dag_scales_with_owners() {
        let leaf = DType::Primitive(PType::I32, Nullability::NonNullable);
        let linear = (0..24).fold(leaf.clone(), |child, _| {
            DType::Struct(
                StructFields::from_iter([("child", child)]),
                Nullability::NonNullable,
            )
        });
        let shared = (0..24).fold(leaf, |child, _| {
            DType::Struct(
                StructFields::from_iter([("left", child.clone()), ("right", child)]),
                Nullability::NonNullable,
            )
        });

        assert!(
            shared.approx_resident_bytes()
                <= linear.approx_resident_bytes().saturating_mul(3)
        );
    }

    #[test]
    fn test_sizing_does_not_resolve_unknown_lazy_extension() -> vortex_error::VortexResult<()> {
        #[expect(clippy::disallowed_methods, reason = "constructing a test extension ID")]
        let id = ExtId::new("test.unknown-footprint");
        let extension = ForeignExtDType::from_parts(id, vec![1; 4096], DType::Null)?;
        let dtype = DType::Struct(
            StructFields::from_iter([("unread", DType::Extension(extension))]),
            Nullability::NonNullable,
        );
        let buffer = dtype.write_flatbuffer_bytes()?;
        let parsed = DType::from_flatbuffer(buffer, &VortexSession::empty())?;
        let fields = match &parsed {
            DType::Struct(fields, _) => fields,
            _ => {
                return Err(vortex_error::vortex_err!(
                    "serialized struct dtype should parse as a struct"
                ));
            }
        };
        assert!(fields.raw_fields()[0].parsed().is_none());

        let estimate = parsed.approx_resident_bytes();

        assert!(estimate > 4096);
        assert!(fields.raw_fields()[0].parsed().is_none());
        Ok(())
    }

    #[test]
    fn test_serialized_names_are_counted_per_occurrence() {
        let mut footprint = DTypeFootprint::default();

        assert_eq!(footprint.serialized_name("shared"), "shared".len());
        assert_eq!(footprint.serialized_name("shared"), "shared".len());
    }
}
