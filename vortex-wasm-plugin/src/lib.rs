// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared ABI types and host-side array support for bundled Vortex WASM encodings.

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayContext;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::serde::ArrayChildren;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArrayRegistry;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::WriteFlatBufferExt;
use vortex_session::VortexSession;
use vortex_session::registry::Id;
use vortex_session::registry::ReadContext;

/// The current host/guest ABI version for bundled WASM encodings.
pub const ABI_VERSION: u16 = 1;

/// Guest-declared metadata describing the bundled module and the encodings it handles.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct GuestManifest {
    /// The ABI version expected by the guest module.
    #[prost(uint32, tag = "1")]
    pub abi_version: u32,
    /// The set of array encodings implemented by this module.
    #[prost(message, repeated, tag = "2")]
    pub encodings: Vec<EncodingManifest>,
}

/// Guest-declared metadata for a single array encoding.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct EncodingManifest {
    /// The array encoding ID handled by this module.
    #[prost(string, tag = "1")]
    pub id: String,
    /// The child index that supplies validity for the array, when any.
    #[prost(uint32, optional, tag = "2")]
    pub validity_from_child: Option<u32>,
    /// Constraints that the host must enforce for each child slot.
    #[prost(message, repeated, tag = "3")]
    pub child_constraints: Vec<ChildConstraint>,
}

/// A manifest constraint for a child slot in a guest-defined encoding.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct ChildConstraint {
    /// The logical slot name used for debugging and display.
    #[prost(string, tag = "1")]
    pub slot_name: String,
    /// The required child encoding ID.
    #[prost(string, tag = "2")]
    pub encoding_id: String,
}

/// Request payload sent to a guest module to canonicalize an array instance.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CanonicalizeRequest {
    /// The flatbuffer-encoded array dtype.
    #[prost(bytes = "vec", tag = "1")]
    pub dtype_bytes: Vec<u8>,
    /// The logical array length.
    #[prost(uint64, tag = "2")]
    pub len: u64,
    /// The read-context encoding IDs for the serialized array payload.
    #[prost(string, repeated, tag = "3")]
    pub ctx_ids: Vec<String>,
    /// The serialized Vortex array blob for the guest-defined subtree.
    #[prost(bytes = "vec", tag = "4")]
    pub array_bytes: Vec<u8>,
}

/// Response payload returned by a guest module after canonicalizing an array.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct CanonicalizeResponse {
    /// The read-context encoding IDs for the returned canonical array.
    #[prost(string, repeated, tag = "1")]
    pub ctx_ids: Vec<String>,
    /// The serialized canonical Vortex array blob.
    #[prost(bytes = "vec", tag = "2")]
    pub array_bytes: Vec<u8>,
}

/// A runtime child constraint derived from a guest manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmChildConstraint {
    slot_name: Arc<str>,
    encoding_id: ArrayId,
}

impl WasmChildConstraint {
    /// Create a new child constraint.
    pub fn new(slot_name: impl Into<Arc<str>>, encoding_id: ArrayId) -> Self {
        Self {
            slot_name: slot_name.into(),
            encoding_id,
        }
    }

    /// Return the slot name declared by the guest manifest.
    pub fn slot_name(&self) -> &Arc<str> {
        &self.slot_name
    }

    /// Return the child encoding ID declared by the guest manifest.
    pub fn encoding_id(&self) -> ArrayId {
        self.encoding_id
    }
}

/// The validated host-side description of a guest-defined encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmEncodingSpec {
    id: ArrayId,
    validity_from_child: Option<usize>,
    child_constraints: Arc<[WasmChildConstraint]>,
}

impl WasmEncodingSpec {
    /// Create a validated encoding specification.
    pub fn new(
        id: ArrayId,
        validity_from_child: Option<usize>,
        child_constraints: Vec<WasmChildConstraint>,
    ) -> Self {
        Self {
            id,
            validity_from_child,
            child_constraints: child_constraints.into(),
        }
    }

    /// Return the encoding ID.
    pub fn id(&self) -> ArrayId {
        self.id
    }

    /// Return the child index that supplies validity, if any.
    pub fn validity_from_child(&self) -> Option<usize> {
        self.validity_from_child
    }

    /// Return the validated child constraints.
    pub fn child_constraints(&self) -> &[WasmChildConstraint] {
        &self.child_constraints
    }
}

impl TryFrom<&EncodingManifest> for WasmEncodingSpec {
    type Error = vortex_error::VortexError;

    fn try_from(value: &EncodingManifest) -> Result<Self, Self::Error> {
        let validity_from_child = value.validity_from_child.map(usize::try_from).transpose()?;
        let child_constraints = value
            .child_constraints
            .iter()
            .map(|child| {
                Ok(WasmChildConstraint::new(
                    Arc::<str>::from(child.slot_name.as_str()),
                    ArrayId::new(&child.encoding_id),
                ))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Self::new(
            ArrayId::new(&value.id),
            validity_from_child,
            child_constraints,
        ))
    }
}

/// Host runtime used by [`WasmArrayEncoding`] to canonicalize guest-defined arrays.
pub trait WasmRuntime: 'static + Send + Sync + fmt::Debug {
    /// Canonicalize a guest-defined array and return a canonical Vortex array.
    fn canonicalize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
        array_registry: &ArrayRegistry,
    ) -> VortexResult<ArrayRef>;
}

/// Array plugin used to deserialize bundled guest-defined arrays as `WasmArray`.
#[derive(Clone, Debug)]
pub struct WasmArrayEncoding {
    spec: WasmEncodingSpec,
    runtime: Arc<dyn WasmRuntime>,
    array_registry: ArrayRegistry,
}

impl WasmArrayEncoding {
    /// Create a new array plugin for a bundled guest-defined encoding.
    pub fn new(
        spec: WasmEncodingSpec,
        runtime: Arc<dyn WasmRuntime>,
        array_registry: ArrayRegistry,
    ) -> Self {
        Self {
            spec,
            runtime,
            array_registry,
        }
    }

    /// Return the validated encoding specification.
    pub fn spec(&self) -> &WasmEncodingSpec {
        &self.spec
    }
}

/// The stored metadata for a deserialized `WasmArray`.
#[derive(Clone, Debug)]
pub struct WasmArrayData {
    metadata: Vec<u8>,
    buffers: Vec<BufferHandle>,
    spec: WasmEncodingSpec,
    runtime: Arc<dyn WasmRuntime>,
    array_registry: ArrayRegistry,
}

impl WasmArrayData {
    /// Create the data payload for a deserialized `WasmArray`.
    pub fn new(
        metadata: Vec<u8>,
        buffers: Vec<BufferHandle>,
        spec: WasmEncodingSpec,
        runtime: Arc<dyn WasmRuntime>,
        array_registry: ArrayRegistry,
    ) -> Self {
        Self {
            metadata,
            buffers,
            spec,
            runtime,
            array_registry,
        }
    }
}

impl Display for WasmArrayData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "WasmArrayData({}B)", self.metadata.len())
    }
}

impl ArrayHash for WasmArrayData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        self.metadata.hash(state);
        self.buffers.len().hash(state);
        for buffer in &self.buffers {
            buffer.array_hash(state, precision);
        }
    }
}

impl ArrayEq for WasmArrayData {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        self.metadata == other.metadata
            && self.buffers.len() == other.buffers.len()
            && self
                .buffers
                .iter()
                .zip(other.buffers.iter())
                .all(|(lhs, rhs)| lhs.array_eq(rhs, precision))
    }
}

/// Operations vtable for `WasmArray`.
pub struct WasmOperationsVTable;

impl OperationsVTable<WasmArrayEncoding> for WasmOperationsVTable {
    fn scalar_at(
        array: ArrayView<'_, WasmArrayEncoding>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<vortex_array::scalar::Scalar> {
        let canonical = array.into_owned().into_array().execute::<Canonical>(ctx)?;
        canonical.into_array().execute_scalar(index, ctx)
    }
}

/// Validity vtable for `WasmArray`.
pub struct WasmValidityVTable;

impl ValidityVTable<WasmArrayEncoding> for WasmValidityVTable {
    fn validity(array: ArrayView<'_, WasmArrayEncoding>) -> VortexResult<Validity> {
        if let Some(validity_child) = array.spec.validity_from_child() {
            let child = array
                .slots()
                .get(validity_child)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    vortex_error::vortex_err!(
                        "Missing validity child {} for {}",
                        validity_child,
                        array.encoding_id()
                    )
                })?;
            child.validity()
        } else {
            Ok(Validity::from(array.dtype().nullability()))
        }
    }
}

impl VTable for WasmArrayEncoding {
    type ArrayData = WasmArrayData;
    type OperationsVTable = WasmOperationsVTable;
    type ValidityVTable = WasmValidityVTable;

    fn id(&self) -> ArrayId {
        self.spec.id()
    }

    fn validate(
        &self,
        _data: &Self::ArrayData,
        _dtype: &DType,
        _len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == self.spec.child_constraints().len(),
            "{} expects {} child slots, got {}",
            self.id(),
            self.spec.child_constraints().len(),
            slots.len()
        );

        for (index, constraint) in self.spec.child_constraints().iter().enumerate() {
            let child = slots
                .get(index)
                .and_then(Option::as_ref)
                .ok_or_else(|| vortex_error::vortex_err!("Missing child slot {}", index))?;
            vortex_ensure!(
                child.encoding_id() == constraint.encoding_id(),
                "{} child {} must decode as {}, got {}",
                self.id(),
                index,
                constraint.encoding_id(),
                child.encoding_id()
            );
        }

        Ok(())
    }

    fn nbuffers(array: ArrayView<'_, Self>) -> usize {
        array.buffers.len()
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        array.buffers[idx].clone()
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        Some(format!("buffer[{idx}]"))
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(array.metadata.clone()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            children.len() == self.spec.child_constraints().len(),
            "{} expects {} children, got {}",
            self.id(),
            self.spec.child_constraints().len(),
            children.len()
        );

        let child_arrays = (0..children.len())
            .map(|idx| children.get(idx, dtype, len))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            WasmArrayData::new(
                metadata.to_vec(),
                buffers.to_vec(),
                self.spec.clone(),
                Arc::clone(&self.runtime),
                self.array_registry.clone(),
            ),
        )
        .with_slots(child_arrays.into_iter().map(Some).collect()))
    }

    fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String {
        array
            .spec
            .child_constraints()
            .get(idx)
            .map(|child| child.slot_name().to_string())
            .unwrap_or_else(|| format!("child[{idx}]"))
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let runtime = Arc::clone(&array.data().runtime);
        let array_registry = array.data().array_registry.clone();
        let array = array.into_array();
        Ok(ExecutionResult::done(runtime.canonicalize(
            &array,
            ctx.session(),
            &array_registry,
        )?))
    }
}

/// Pack a guest pointer/length pair into the u64 ABI used by exported WASM functions.
pub fn pack_ptr_len(ptr: u32, len: u32) -> u64 {
    (u64::from(len) << 32) | u64::from(ptr)
}

/// Unpack a guest pointer/length pair from the u64 ABI used by exported WASM functions.
pub fn unpack_ptr_len(value: u64) -> (u32, u32) {
    let [p0, p1, p2, p3, l0, l1, l2, l3] = value.to_le_bytes();
    (
        u32::from_le_bytes([p0, p1, p2, p3]),
        u32::from_le_bytes([l0, l1, l2, l3]),
    )
}

/// Build a canonicalization request for a guest-defined array subtree.
pub fn build_canonicalize_request(
    array: &ArrayRef,
    session: &VortexSession,
    array_registry: &ArrayRegistry,
) -> VortexResult<CanonicalizeRequest> {
    let ctx = ArrayContext::empty();
    let buffers = array.serialize_with_array_registry(
        &ctx,
        session,
        Some(array_registry),
        &SerializeOptions::default(),
    )?;
    Ok(CanonicalizeRequest {
        dtype_bytes: array
            .dtype()
            .write_flatbuffer_bytes()?
            .into_inner()
            .to_vec(),
        len: u64::try_from(array.len())?,
        ctx_ids: ctx.to_ids().into_iter().map(|id| id.to_string()).collect(),
        array_bytes: concat_buffers(buffers),
    })
}

/// Decode a canonicalization request inside a guest module.
pub fn decode_canonicalize_request(
    request: &CanonicalizeRequest,
    session: &VortexSession,
) -> VortexResult<(DType, usize, ReadContext, ArrayRef)> {
    let dtype = DType::from_flatbuffer(FlatBuffer::copy_from(&request.dtype_bytes), session)?;
    let len = usize::try_from(request.len)?;
    let ctx = decode_ctx_ids(&request.ctx_ids);
    let array = SerializedArray::try_from(ByteBuffer::from(request.array_bytes.clone()))?
        .decode(&dtype, len, &ctx, session)?;
    Ok((dtype, len, ctx, array))
}

/// Build a canonicalization response containing a canonical Vortex array.
pub fn build_canonicalize_response(
    array: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<CanonicalizeResponse> {
    let ctx = ArrayContext::empty();
    let buffers = array.serialize(&ctx, session, &SerializeOptions::default())?;
    Ok(CanonicalizeResponse {
        ctx_ids: ctx.to_ids().into_iter().map(|id| id.to_string()).collect(),
        array_bytes: concat_buffers(buffers),
    })
}

/// Decode a canonicalization response returned by a guest module.
pub fn decode_canonicalize_response(
    dtype: &DType,
    len: usize,
    response: &CanonicalizeResponse,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let ctx = decode_ctx_ids(&response.ctx_ids);
    SerializedArray::try_from(ByteBuffer::from(response.array_bytes.clone()))?
        .decode(dtype, len, &ctx, session)
}

fn decode_ctx_ids(ids: &[String]) -> ReadContext {
    ReadContext::new(ids.iter().map(|id| Id::new(id)).collect::<Vec<_>>())
}

fn concat_buffers(buffers: Vec<ByteBuffer>) -> Vec<u8> {
    let total_len = buffers.iter().map(ByteBuffer::len).sum();
    let mut bytes = Vec::with_capacity(total_len);
    for buffer in buffers {
        bytes.extend_from_slice(buffer.as_slice());
    }
    bytes
}
