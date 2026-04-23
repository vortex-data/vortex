// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::session::ArrayRegistry;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_layout::segments::SegmentSource;
use vortex_session::VortexSession;

use crate::footer::Footer;

pub(crate) type FileArrayPluginOverlay = ArrayRegistry;

#[cfg(feature = "wasm_plugins")]
use std::future::Future;

#[cfg(feature = "wasm_plugins")]
use parking_lot::Mutex;
#[cfg(feature = "wasm_plugins")]
use prost::Message;
#[cfg(feature = "wasm_plugins")]
use vortex_array::ArrayId;
#[cfg(feature = "wasm_plugins")]
use vortex_array::ArrayPluginRef;
#[cfg(feature = "wasm_plugins")]
use vortex_array::ArrayRef;
#[cfg(feature = "wasm_plugins")]
use vortex_array::buffer::BufferHandle;
#[cfg(feature = "wasm_plugins")]
use vortex_array::serde::SerializedArray;
#[cfg(feature = "wasm_plugins")]
use vortex_error::vortex_err;
#[cfg(feature = "wasm_plugins")]
use vortex_layout::layouts::flat::Flat;
#[cfg(feature = "wasm_plugins")]
use vortex_layout::segments::SegmentId;
#[cfg(feature = "wasm_plugins")]
use vortex_session::registry::ReadContext;
#[cfg(feature = "wasm_plugins")]
use vortex_utils::aliases::hash_map::HashMap;
#[cfg(feature = "wasm_plugins")]
use vortex_utils::aliases::hash_set::HashSet;
#[cfg(feature = "wasm_plugins")]
use vortex_wasm_plugin::ABI_VERSION;
#[cfg(feature = "wasm_plugins")]
use vortex_wasm_plugin::CanonicalizeResponse;
#[cfg(feature = "wasm_plugins")]
use vortex_wasm_plugin::GuestManifest;
#[cfg(feature = "wasm_plugins")]
use vortex_wasm_plugin::WasmArrayEncoding;
#[cfg(feature = "wasm_plugins")]
use vortex_wasm_plugin::WasmEncodingSpec;
#[cfg(feature = "wasm_plugins")]
use vortex_wasm_plugin::WasmRuntime;
#[cfg(feature = "wasm_plugins")]
use vortex_wasm_plugin::build_canonicalize_request;
#[cfg(feature = "wasm_plugins")]
use vortex_wasm_plugin::decode_canonicalize_response;
#[cfg(feature = "wasm_plugins")]
use vortex_wasm_plugin::unpack_ptr_len;
#[cfg(feature = "wasm_plugins")]
use wasmi::Engine;
#[cfg(feature = "wasm_plugins")]
use wasmi::Linker;
#[cfg(feature = "wasm_plugins")]
use wasmi::Memory;
#[cfg(feature = "wasm_plugins")]
use wasmi::Module;
#[cfg(feature = "wasm_plugins")]
use wasmi::Store;
#[cfg(feature = "wasm_plugins")]
use wasmi::TypedFunc;

#[cfg(feature = "wasm_plugins")]
use crate::footer::BundledWasmSpec;

#[cfg(feature = "wasm_plugins")]
pub(crate) async fn load_file_array_plugin_overlay(
    footer: &Footer,
    segment_source: Arc<dyn SegmentSource>,
    _session: &VortexSession,
) -> VortexResult<Option<FileArrayPluginOverlay>> {
    if footer.bundled_wasm_specs().is_empty() {
        return Ok(None);
    }

    let overlay = ArrayRegistry::empty();
    let mut seen_ids = HashSet::new();
    let mut specs = HashMap::new();

    for bundled_wasm_spec in footer.bundled_wasm_specs() {
        let array_id = footer
            .resolve_array_spec(bundled_wasm_spec.array_spec_idx)
            .ok_or_else(|| {
                vortex_err!(
                    "Bundled WASM spec references unknown array spec index {}",
                    bundled_wasm_spec.array_spec_idx
                )
            })?;
        if !seen_ids.insert(array_id) {
            vortex_bail!("Duplicate bundled WASM module for {}", array_id);
        }

        let module_bytes = read_module_bytes(
            segment_source.request(SegmentId::from(bundled_wasm_spec.segment_idx)),
        )
        .await?;
        let (runtime, manifest) = WasmModuleRuntime::try_new(&module_bytes)?;
        let spec = validate_manifest(&manifest, bundled_wasm_spec, array_id)?;
        overlay.register(
            array_id,
            Arc::new(WasmArrayEncoding::new(
                spec.clone(),
                Arc::new(runtime),
                overlay.clone(),
            )) as ArrayPluginRef,
        );
        specs.insert(array_id, spec);
    }

    validate_bundled_arrays_in_file(footer, segment_source, &specs).await?;
    Ok(Some(overlay))
}

#[cfg(not(feature = "wasm_plugins"))]
pub(crate) async fn load_file_array_plugin_overlay(
    footer: &Footer,
    _segment_source: Arc<dyn SegmentSource>,
    _session: &VortexSession,
) -> VortexResult<Option<FileArrayPluginOverlay>> {
    if footer.bundled_wasm_specs().is_empty() {
        Ok(None)
    } else {
        vortex_bail!(
            "This file contains bundled WASM modules, but vortex-file was built without the `wasm_plugins` feature"
        )
    }
}

#[cfg(feature = "wasm_plugins")]
#[derive(Debug)]
struct WasmModuleRuntime {
    inner: Mutex<WasmModuleRuntimeInner>,
}

#[cfg(feature = "wasm_plugins")]
#[derive(Debug)]
struct WasmModuleRuntimeInner {
    store: Store<()>,
    memory: Memory,
    alloc: TypedFunc<u32, u32>,
    free: TypedFunc<(u32, u32), ()>,
    canonicalize: TypedFunc<(u32, u32), u64>,
}

#[cfg(feature = "wasm_plugins")]
impl WasmModuleRuntime {
    fn try_new(module_bytes: &[u8]) -> VortexResult<(Self, GuestManifest)> {
        let engine = Engine::default();
        let module = Module::new(&engine, module_bytes)
            .map_err(|err| vortex_err!("Failed to compile bundled WASM module: {err}"))?;
        let linker = Linker::<()>::new(&engine);
        let mut store = Store::new(&engine, ());
        let instance = linker
            .instantiate_and_start(&mut store, &module)
            .map_err(|err| vortex_err!("Failed to instantiate bundled WASM module: {err}"))?;
        let memory = instance
            .get_memory(&store, "memory")
            .ok_or_else(|| vortex_err!("Bundled WASM module does not export memory"))?;
        let manifest = instance
            .get_typed_func::<(), u64>(&store, "vx_manifest")
            .map_err(|err| vortex_err!("Bundled WASM module is missing vx_manifest: {err}"))?;
        let alloc = instance
            .get_typed_func::<u32, u32>(&store, "vx_alloc")
            .map_err(|err| vortex_err!("Bundled WASM module is missing vx_alloc: {err}"))?;
        let free = instance
            .get_typed_func::<(u32, u32), ()>(&store, "vx_free")
            .map_err(|err| vortex_err!("Bundled WASM module is missing vx_free: {err}"))?;
        let canonicalize = instance
            .get_typed_func::<(u32, u32), u64>(&store, "vx_canonicalize_array")
            .map_err(|err| {
                vortex_err!("Bundled WASM module is missing vx_canonicalize_array: {err}")
            })?;

        let manifest = {
            let manifest_loc = manifest
                .call(&mut store, ())
                .map_err(|err| vortex_err!("Bundled WASM module manifest call failed: {err}"))?;
            let manifest_bytes = read_memory(&memory, &store, manifest_loc)?;
            GuestManifest::decode(manifest_bytes.as_slice())
                .map_err(|err| vortex_err!("Failed to decode bundled WASM manifest: {err}"))?
        };

        Ok((
            Self {
                inner: Mutex::new(WasmModuleRuntimeInner {
                    store,
                    memory,
                    alloc,
                    free,
                    canonicalize,
                }),
            },
            manifest,
        ))
    }
}

#[cfg(feature = "wasm_plugins")]
impl WasmRuntime for WasmModuleRuntime {
    fn canonicalize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
        array_registry: &ArrayRegistry,
    ) -> VortexResult<ArrayRef> {
        let request = build_canonicalize_request(array, session, array_registry)?;
        let request_bytes = request.encode_to_vec();
        let request_len = u32::try_from(request_bytes.len())
            .map_err(|_| vortex_err!("Canonicalize request exceeds u32 length"))?;

        let mut inner = self.inner.lock();
        let alloc = inner.alloc;
        let free = inner.free;
        let canonicalize = inner.canonicalize;
        let memory = inner.memory;
        let store = &mut inner.store;

        let request_ptr = alloc
            .call(&mut *store, request_len)
            .map_err(|err| vortex_err!("Bundled WASM allocation failed: {err}"))?;
        memory
            .write(&mut *store, request_ptr as usize, &request_bytes)
            .map_err(|err| vortex_err!("Failed to write request into guest memory: {err}"))?;

        let response_loc = canonicalize
            .call(&mut *store, (request_ptr, request_len))
            .map_err(|err| vortex_err!("Bundled WASM canonicalize call failed: {err}"))?;
        free.call(&mut *store, (request_ptr, request_len))
            .map_err(|err| vortex_err!("Failed to free guest request buffer: {err}"))?;

        let response_bytes = read_memory(&memory, &*store, response_loc)?;
        let (response_ptr, response_len) = unpack_ptr_len(response_loc);
        free.call(&mut *store, (response_ptr, response_len))
            .map_err(|err| vortex_err!("Failed to free guest response buffer: {err}"))?;

        let response = CanonicalizeResponse::decode(response_bytes.as_slice())
            .map_err(|err| vortex_err!("Failed to decode WASM canonicalize response: {err}"))?;
        decode_canonicalize_response(array.dtype(), array.len(), &response, session)
    }
}

#[cfg(feature = "wasm_plugins")]
fn validate_manifest(
    manifest: &GuestManifest,
    bundled_wasm_spec: &BundledWasmSpec,
    array_id: ArrayId,
) -> VortexResult<WasmEncodingSpec> {
    if bundled_wasm_spec.abi_version != ABI_VERSION {
        vortex_bail!(
            "Bundled WASM module for {} declares ABI {}, but the host only supports ABI {}",
            array_id,
            bundled_wasm_spec.abi_version,
            ABI_VERSION
        );
    }
    if manifest.abi_version != u32::from(ABI_VERSION) {
        vortex_bail!(
            "Bundled WASM module for {} reports ABI {}, expected {}",
            array_id,
            manifest.abi_version,
            ABI_VERSION
        );
    }
    if manifest.encodings.len() != 1 {
        vortex_bail!(
            "Bundled WASM modules must currently expose exactly one encoding, found {}",
            manifest.encodings.len()
        );
    }

    let spec = WasmEncodingSpec::try_from(&manifest.encodings[0])?;
    if spec.id() != array_id {
        vortex_bail!(
            "Bundled WASM manifest id mismatch: footer requested {}, manifest exported {}",
            array_id,
            spec.id()
        );
    }

    validate_demo_for_constraints(&spec)?;
    Ok(spec)
}

#[cfg(feature = "wasm_plugins")]
async fn validate_bundled_arrays_in_file(
    footer: &Footer,
    segment_source: Arc<dyn SegmentSource>,
    specs: &HashMap<ArrayId, WasmEncodingSpec>,
) -> VortexResult<()> {
    for layout in footer.layout().depth_first_traversal() {
        let layout = layout?;
        let Some(flat_layout) = layout.as_opt::<Flat>() else {
            continue;
        };

        let serialized = if let Some(array_tree) = flat_layout.array_tree().cloned() {
            SerializedArray::from_array_tree(array_tree)?
        } else {
            let segment = segment_source
                .request(flat_layout.segment_id())
                .await?
                .try_into_host()?
                .await?;
            SerializedArray::try_from(segment)?
        };
        validate_serialized_array(&serialized, flat_layout.array_ctx(), specs)?;
    }

    Ok(())
}

#[cfg(feature = "wasm_plugins")]
fn validate_serialized_array(
    serialized: &SerializedArray,
    ctx: &ReadContext,
    specs: &HashMap<ArrayId, WasmEncodingSpec>,
) -> VortexResult<()> {
    let encoding_id = ctx
        .resolve(serialized.encoding_id())
        .ok_or_else(|| vortex_err!("Unknown encoding index {}", serialized.encoding_id()))?;

    if let Some(spec) = specs.get(&encoding_id) {
        if serialized.nchildren() != spec.child_constraints().len() {
            vortex_bail!(
                "Bundled {} expects {} children, but the file contains {}",
                encoding_id,
                spec.child_constraints().len(),
                serialized.nchildren()
            );
        }

        for (index, child_constraint) in spec.child_constraints().iter().enumerate() {
            let child = serialized.child(index);
            let child_encoding_id = ctx.resolve(child.encoding_id()).ok_or_else(|| {
                vortex_err!("Unknown child encoding index {}", child.encoding_id())
            })?;
            if child_encoding_id != child_constraint.encoding_id() {
                vortex_bail!(
                    "Bundled {} requires child {} to use {}, found {}",
                    encoding_id,
                    index,
                    child_constraint.encoding_id(),
                    child_encoding_id
                );
            }
        }
    }

    for index in 0..serialized.nchildren() {
        validate_serialized_array(&serialized.child(index), ctx, specs)?;
    }
    Ok(())
}

#[cfg(feature = "wasm_plugins")]
fn validate_demo_for_constraints(spec: &WasmEncodingSpec) -> VortexResult<()> {
    let primitive_id = ArrayId::new("vortex.primitive");
    if spec.id() != ArrayId::new("fastlanes.for") {
        vortex_bail!(
            "Only bundled fastlanes.for is supported in this experimental implementation, got {}",
            spec.id()
        );
    }
    if spec.validity_from_child() != Some(0) {
        vortex_bail!(
            "Bundled {} must declare validity_from_child = 0 for the demo ABI",
            spec.id()
        );
    }
    if spec.child_constraints().len() != 1 {
        vortex_bail!(
            "Bundled {} must declare exactly one child constraint for the demo ABI",
            spec.id()
        );
    }
    let child = &spec.child_constraints()[0];
    if child.slot_name().as_ref() != "encoded" || child.encoding_id() != primitive_id {
        vortex_bail!(
            "Bundled {} must require slot `encoded` with child encoding {}",
            spec.id(),
            primitive_id
        );
    }
    Ok(())
}

#[cfg(feature = "wasm_plugins")]
async fn read_module_bytes(
    segment: impl Future<Output = VortexResult<BufferHandle>>,
) -> VortexResult<Vec<u8>> {
    Ok(segment.await?.try_into_host()?.await?.as_slice().to_vec())
}

#[cfg(feature = "wasm_plugins")]
fn read_memory(memory: &Memory, store: &Store<()>, ptr_len: u64) -> VortexResult<Vec<u8>> {
    let (ptr, len) = unpack_ptr_len(ptr_len);
    let mut bytes = vec![0; len as usize];
    memory
        .read(store, ptr as usize, &mut bytes)
        .map_err(|err| vortex_err!("Failed to read guest memory: {err}"))?;
    Ok(bytes)
}
