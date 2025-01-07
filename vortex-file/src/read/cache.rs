use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use bytes::Bytes;
use flatbuffers::root_unchecked;
use once_cell::sync::OnceCell;
use vortex_array::aliases::hash_map::HashMap;
use vortex_buffer::ByteBuffer;
use vortex_dtype::flatbuffers::{extract_field, project_and_deserialize, resolve_field};
use vortex_dtype::{DType, Field, FieldNames};
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_flatbuffers::dtype as fbd;

use crate::read::projection::Projection;
use crate::read::{LayoutPartId, MessageId};

/// A read-only cache of messages.
pub trait MessageCache {
    fn get(&self, path: &[LayoutPartId]) -> Option<Bytes>;
}

#[derive(Default, Debug, Clone)]
pub struct LayoutMessageCache {
    cache: Arc<RwLock<HashMap<MessageId, Bytes>>>,
}

impl LayoutMessageCache {
    pub fn remove(&self, path: &[LayoutPartId]) -> Option<Bytes> {
        self.cache
            .write()
            .map_err(|_| vortex_err!("Poisoned cache"))
            .vortex_expect("poisoned")
            .remove(path)
    }

    pub fn set(&self, path: MessageId, value: Bytes) {
        self.cache
            .write()
            .map_err(|_| vortex_err!("Poisoned cache"))
            .vortex_expect("poisoned")
            .insert(path, value);
    }

    pub fn set_many<I: IntoIterator<Item = (MessageId, Bytes)>>(&self, iter: I) {
        let mut guard = self
            .cache
            .write()
            .map_err(|_| vortex_err!("Poisoned cache"))
            .vortex_expect("poisoned");
        for (id, bytes) in iter.into_iter() {
            guard.insert(id, bytes);
        }
    }
}

impl MessageCache for LayoutMessageCache {
    fn get(&self, path: &[LayoutPartId]) -> Option<Bytes> {
        self.cache
            .read()
            .map_err(|_| vortex_err!("Poisoned cache"))
            .vortex_expect("poisoned")
            .get(path)
            .cloned()
    }
}

#[derive(Debug)]
enum SerializedDTypeField {
    Projection(Projection),
    Field(Field),
}

impl SerializedDTypeField {
    pub fn project(&self, fields: &[Field]) -> VortexResult<Self> {
        match self {
            SerializedDTypeField::Projection(p) => {
                Ok(SerializedDTypeField::Projection(p.project(fields)?))
            }
            SerializedDTypeField::Field(f) => {
                if fields.len() > 1 && &fields[0] != f {
                    vortex_bail!("Can't project field {f} into {fields:?}")
                }
                Ok(SerializedDTypeField::Field(f.clone()))
            }
        }
    }

    pub fn field(&self, field: &Field) -> VortexResult<Self> {
        match self {
            SerializedDTypeField::Projection(p) => {
                match p {
                    Projection::All => {}
                    Projection::Flat(fields) => {
                        if !fields.iter().any(|pf| pf == field) {
                            vortex_bail!("Can't project {fields:?} into {field}")
                        }
                    }
                }
                Ok(SerializedDTypeField::Field(field.clone()))
            }
            SerializedDTypeField::Field(f) => {
                if f != field {
                    vortex_bail!("Can't extract field from field")
                }
                Ok(SerializedDTypeField::Field(field.clone()))
            }
        }
    }
}

#[derive(Debug)]
enum LazyDTypeState {
    DType(DType),
    Serialized(ByteBuffer, OnceCell<DType>, SerializedDTypeField),
    Unknown,
}

#[derive(Debug)]
pub struct LazyDType {
    inner: LazyDTypeState,
}

impl LazyDType {
    /// Create a LazilyDeserializedDType from a flatbuffer schema bytes
    /// i.e., these bytes need to be deserializable as message::Schema
    ///
    /// # Safety
    /// This function is unsafe because it trusts the caller to pass in a valid flatbuffer
    /// representing a message::Schema.
    /// FIXME(ngates): this should take a ConstByteBuffer<8> aliased as FlatBuffer
    pub unsafe fn from_schema_bytes(dtype_bytes: ByteBuffer) -> Self {
        Self {
            inner: LazyDTypeState::Serialized(
                dtype_bytes,
                OnceCell::new(),
                SerializedDTypeField::Projection(Projection::All),
            ),
        }
    }

    pub fn from_dtype(dtype: DType) -> Self {
        Self {
            inner: LazyDTypeState::DType(dtype),
        }
    }

    pub fn unknown() -> Self {
        Self {
            inner: LazyDTypeState::Unknown,
        }
    }

    /// Restrict the underlying dtype to selected fields
    pub fn project(&self, fields: &[Field]) -> VortexResult<Arc<Self>> {
        match &self.inner {
            LazyDTypeState::DType(dtype) => {
                let DType::Struct(sdt, n) = dtype else {
                    vortex_bail!("Not a struct dtype")
                };
                Ok(Arc::new(LazyDType::from_dtype(DType::Struct(
                    sdt.project(fields)?,
                    *n,
                ))))
            }
            LazyDTypeState::Serialized(b, _, current_projection) => Ok(Arc::new(Self {
                inner: LazyDTypeState::Serialized(
                    b.clone(),
                    OnceCell::new(),
                    current_projection.project(fields)?,
                ),
            })),
            LazyDTypeState::Unknown => vortex_bail!("Unknown dtype"),
        }
    }

    /// Extract single field out of this dtype
    pub fn field(&self, field: &Field) -> VortexResult<Arc<Self>> {
        match &self.inner {
            LazyDTypeState::DType(dtype) => {
                let DType::Struct(sdt, _) = dtype else {
                    vortex_bail!("Not a struct dtype")
                };
                Ok(Arc::new(LazyDType::from_dtype(
                    sdt.field_info(field)?.dtype.clone(),
                )))
            }
            LazyDTypeState::Serialized(b, _, current_projection) => Ok(Arc::new(Self {
                inner: LazyDTypeState::Serialized(
                    b.clone(),
                    OnceCell::new(),
                    current_projection.field(field)?,
                ),
            })),
            LazyDTypeState::Unknown => vortex_bail!("Unknown dtype"),
        }
    }

    /// Extract field names from the underlying dtype if there are any
    pub fn names(&self) -> VortexResult<FieldNames> {
        match &self.inner {
            LazyDTypeState::DType(dtype) => {
                let DType::Struct(sdt, _) = dtype else {
                    vortex_bail!("Not a struct dtype")
                };
                Ok(sdt.names().clone())
            }
            LazyDTypeState::Serialized(b, _, proj) => field_names(b, proj),
            LazyDTypeState::Unknown => vortex_bail!("Unknown dtype"),
        }
    }

    /// Get vortex dtype out of serialized bytes
    pub fn value(&self) -> VortexResult<&DType> {
        match &self.inner {
            LazyDTypeState::DType(dtype) => Ok(dtype),
            LazyDTypeState::Serialized(bytes, cache, proj) => {
                cache.get_or_try_init(|| project_dtype_bytes(bytes, proj))
            }
            LazyDTypeState::Unknown => vortex_bail!("Unknown dtype"),
        }
    }

    /// Convert all name based references to index based to create globally addressable filter
    pub(crate) fn resolve_field(&self, field: &Field) -> VortexResult<usize> {
        match &self.inner {
            LazyDTypeState::DType(dtype) => {
                let DType::Struct(sdt, _) = dtype else {
                    vortex_bail!("Trying to resolve fields in non struct dtype")
                };
                match field {
                    Field::Name(n) => sdt
                        .names()
                        .iter()
                        .position(|name| name == n)
                        .ok_or_else(|| vortex_err!("Can't find {n} in the type")),
                    Field::Index(i) => Ok(*i),
                }
            }
            LazyDTypeState::Serialized(b, ..) => resolve_field(fb_struct(b.as_ref())?, field),
            LazyDTypeState::Unknown => vortex_bail!("Unknown dtype"),
        }
    }
}

fn field_names(bytes: &[u8], dtype_field: &SerializedDTypeField) -> VortexResult<FieldNames> {
    let struct_field = fb_struct(bytes)?;
    let names = struct_field
        .names()
        .ok_or_else(|| vortex_err!("Not a struct dtype"))?;
    match dtype_field {
        SerializedDTypeField::Projection(projection) => match projection {
            Projection::All => Ok(names.iter().map(Arc::from).collect()),
            Projection::Flat(fields) => fields
                .iter()
                .map(|f| resolve_field(struct_field, f))
                .map(|idx| idx.map(|i| Arc::from(names.get(i))))
                .collect(),
        },
        SerializedDTypeField::Field(f) => Ok(Arc::new([Arc::from(
            names.get(resolve_field(struct_field, f)?),
        )])),
    }
}

fn project_dtype_bytes(bytes: &[u8], dtype_field: &SerializedDTypeField) -> VortexResult<DType> {
    let fb_dtype = fb_dtype(bytes);
    match dtype_field {
        SerializedDTypeField::Projection(projection) => match projection {
            Projection::All => DType::try_from(fb_dtype),
            Projection::Flat(p) => project_and_deserialize(fb_dtype, p),
        },
        SerializedDTypeField::Field(f) => extract_field(fb_dtype, f),
    }
}

fn fb_struct(bytes: &[u8]) -> VortexResult<fbd::Struct_> {
    fb_dtype(bytes)
        .type__as_struct_()
        .ok_or_else(|| vortex_err!("The top-level type should be a struct"))
}

fn fb_dtype(bytes: &[u8]) -> fbd::DType {
    unsafe { root_unchecked::<fbd::DType>(bytes) }
}
