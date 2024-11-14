use std::sync::{Arc, RwLock};

use bytes::Bytes;
use flatbuffers::root_unchecked;
use once_cell::sync::OnceCell;
use vortex_array::aliases::hash_map::HashMap;
use vortex_dtype::field::Field;
use vortex_dtype::flatbuffers::{deserialize_and_project, resolve_field};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexExpect, VortexResult};
use vortex_flatbuffers::message;
use vortex_schema::projection::Projection;

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
enum LazyDTypeState {
    Value(DType),
    Serialized(Bytes, OnceCell<DType>, Projection),
}

#[derive(Debug)]
pub struct LazilyDeserializedDType {
    inner: LazyDTypeState,
}

impl LazilyDeserializedDType {
    /// Create a LazilyDeserializedDType from a flatbuffer schema bytes
    /// i.e., these bytes need to be deserializable as message::Schema
    ///
    /// # Safety
    /// This function is unsafe because it trusts the caller to pass in a valid flatbuffer
    /// representing a message::Schema.
    pub unsafe fn from_schema_bytes(schema_bytes: Bytes, projection: Projection) -> Self {
        Self {
            inner: LazyDTypeState::Serialized(schema_bytes, OnceCell::new(), projection),
        }
    }

    pub fn from_dtype(dtype: DType) -> Self {
        Self {
            inner: LazyDTypeState::Value(dtype),
        }
    }

    /// Restrict the underlying dtype to selected fields
    pub fn project(&self, projection: &[Field]) -> VortexResult<Arc<Self>> {
        match &self.inner {
            LazyDTypeState::Value(dtype) => {
                let DType::Struct(sdt, n) = dtype else {
                    vortex_bail!("Not a struct dtype")
                };
                Ok(Arc::new(LazilyDeserializedDType::from_dtype(
                    DType::Struct(sdt.project(projection)?, *n),
                )))
            }
            LazyDTypeState::Serialized(b, _, proj) => {
                let projection = match proj {
                    Projection::All => Projection::Flat(projection.to_vec()),
                    // TODO(robert): Respect existing projection list, only really an issue for nested structs
                    Projection::Flat(_) => vortex_bail!("Can't project already projected dtype"),
                };
                unsafe {
                    Ok(Arc::new(LazilyDeserializedDType::from_schema_bytes(
                        b.clone(),
                        projection,
                    )))
                }
            }
        }
    }

    /// Get vortex dtype out of serialized bytes
    pub fn value(&self) -> VortexResult<&DType> {
        match &self.inner {
            LazyDTypeState::Value(dtype) => Ok(dtype),
            LazyDTypeState::Serialized(bytes, cache, proj) => cache.get_or_try_init(|| {
                let fb_dtype = Self::fb_schema(bytes)?
                    .dtype()
                    .ok_or_else(|| vortex_err!(InvalidSerde: "Schema missing DType"))?;
                match &proj {
                    Projection::All => DType::try_from(fb_dtype),
                    Projection::Flat(p) => deserialize_and_project(fb_dtype, p),
                }
            }),
        }
    }

    /// Convert all name based references to index based to create globally addressable filter
    pub(crate) fn resolve_field(&self, field: &Field) -> VortexResult<usize> {
        match &self.inner {
            LazyDTypeState::Value(dtype) => {
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
            LazyDTypeState::Serialized(b, ..) => {
                let fb_struct = Self::fb_schema(b.as_ref())?
                    .dtype()
                    .and_then(|d| d.type__as_struct_())
                    .ok_or_else(|| vortex_err!("The top-level type should be a struct"))?;
                resolve_field(fb_struct, field)
            }
        }
    }

    fn fb_schema(bytes: &[u8]) -> VortexResult<message::Schema> {
        Ok(unsafe { root_unchecked::<message::Schema>(bytes) })
    }
}

#[derive(Debug)]
pub struct RelativeLayoutCache {
    root: Arc<RwLock<LayoutMessageCache>>,
    dtype: Option<Arc<LazilyDeserializedDType>>,
    path: MessageId,
}

impl RelativeLayoutCache {
    pub fn new(root: Arc<RwLock<LayoutMessageCache>>, dtype: Arc<LazilyDeserializedDType>) -> Self {
        Self {
            root,
            dtype: Some(dtype),
            path: Vec::new(),
        }
    }

    pub fn relative(&self, id: LayoutPartId, dtype: Arc<LazilyDeserializedDType>) -> Self {
        let mut new_path = self.path.clone();
        new_path.push(id);
        Self {
            root: self.root.clone(),
            path: new_path,
            dtype: Some(dtype),
        }
    }

    pub fn unknown_dtype(&self, id: LayoutPartId) -> Self {
        let mut new_path = self.path.clone();
        new_path.push(id);
        Self {
            root: self.root.clone(),
            path: new_path,
            dtype: None,
        }
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

    pub fn dtype(&self) -> &Arc<LazilyDeserializedDType> {
        self.dtype.as_ref().vortex_expect("Must have dtype")
    }

    pub fn absolute_id(&self, path: &[LayoutPartId]) -> MessageId {
        let mut lookup_key = Vec::with_capacity(self.path.len() + path.len());
        lookup_key.clone_from(&self.path);
        lookup_key.extend_from_slice(path);
        lookup_key
    }
}
