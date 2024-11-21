use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use bytes::Bytes;
use flatbuffers::root_unchecked;
use once_cell::sync::OnceCell;
use vortex_array::aliases::hash_map::HashMap;
use vortex_dtype::field::Field;
use vortex_dtype::flatbuffers::{extract_field, project_and_deserialize, resolve_field};
use vortex_dtype::{DType, FieldNames};
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexResult};
use vortex_flatbuffers::dtype::Struct_;
use vortex_flatbuffers::message;

use crate::read::projection::Projection;
use crate::read::{LayoutPartId, MessageId};

#[derive(Default, Debug)]
pub struct LayoutMessageCache {
    cache: HashMap<MessageId, Bytes>,
}

impl LayoutMessageCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn get(&self, path: &[LayoutPartId]) -> Option<Bytes> {
        self.cache.get(path).cloned()
    }

    pub fn remove(&mut self, path: &[LayoutPartId]) -> Option<Bytes> {
        self.cache.remove(path)
    }

    pub fn set(&mut self, path: MessageId, value: Bytes) {
        self.cache.insert(path, value);
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
    Serialized(Bytes, OnceCell<DType>, SerializedDTypeField),
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
    pub unsafe fn from_schema_bytes(dtype_bytes: Bytes) -> Self {
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
                        .position(|name| name.as_ref() == n.as_str())
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
    let fb_dtype = fb_schema(bytes)
        .dtype()
        .ok_or_else(|| vortex_err!(InvalidSerde: "Schema missing DType"))?;

    match dtype_field {
        SerializedDTypeField::Projection(projection) => match projection {
            Projection::All => DType::try_from(fb_dtype),
            Projection::Flat(p) => project_and_deserialize(fb_dtype, p),
        },
        SerializedDTypeField::Field(f) => extract_field(fb_dtype, f),
    }
}

fn fb_struct(bytes: &[u8]) -> VortexResult<Struct_> {
    fb_schema(bytes)
        .dtype()
        .and_then(|d| d.type__as_struct_())
        .ok_or_else(|| vortex_err!("The top-level type should be a struct"))
}

fn fb_schema(bytes: &[u8]) -> message::Schema {
    unsafe { root_unchecked::<message::Schema>(bytes) }
}

#[derive(Debug, Clone)]
pub struct RelativeLayoutCache {
    root: Arc<RwLock<LayoutMessageCache>>,
    dtype: Arc<LazyDType>,
    path: MessageId,
}

impl RelativeLayoutCache {
    pub fn new(root: Arc<RwLock<LayoutMessageCache>>, dtype: Arc<LazyDType>) -> Self {
        Self {
            root,
            dtype,
            path: Vec::new(),
        }
    }

    pub fn relative(&self, id: LayoutPartId, dtype: Arc<LazyDType>) -> Self {
        let mut new_path = self.path.clone();
        new_path.push(id);
        Self {
            root: self.root.clone(),
            path: new_path,
            dtype,
        }
    }

    pub fn unknown_dtype(&self, id: LayoutPartId) -> Self {
        self.relative(id, Arc::new(LazyDType::unknown()))
    }

    pub fn get(&self, path: &[LayoutPartId]) -> Option<Bytes> {
        self.root
            .read()
            .unwrap_or_else(|poison| {
                vortex_panic!(
                    "Failed to read from layout cache at path {:?} with error {}",
                    path,
                    poison
                );
            })
            .get(&self.absolute_id(path))
    }

    pub fn remove(&mut self, path: &[LayoutPartId]) -> Option<Bytes> {
        self.root
            .write()
            .unwrap_or_else(|poison| {
                vortex_panic!(
                    "Failed to write to layout cache at path {:?} with error {}",
                    path,
                    poison
                )
            })
            .remove(&self.absolute_id(path))
    }

    pub fn dtype(&self) -> &Arc<LazyDType> {
        &self.dtype
    }

    pub fn absolute_id(&self, path: &[LayoutPartId]) -> MessageId {
        let mut lookup_key = Vec::with_capacity(self.path.len() + path.len());
        lookup_key.clone_from(&self.path);
        lookup_key.extend_from_slice(path);
        lookup_key
    }
}
