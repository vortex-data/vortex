// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The embedded WebAssembly decode runtime.
//!
//! [`WasmKernel`] wraps a compiled `wasmi` module and drives the host/guest ABI: it copies the
//! serialized array into guest memory, calls the guest's `vx_decode` export, services
//! `vx_decode_child` callbacks from a [`HostDecoder`], and reconstructs a Vortex array from the
//! returned [`CanonicalMessage`](crate::message).

use vortex_array::ArrayRef;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use wasmi::Caller;
use wasmi::Engine;
use wasmi::Extern;
use wasmi::Linker;
use wasmi::Module;
use wasmi::Store;

use crate::abi::ALLOC_EXPORT;
use crate::abi::DECODE_CHILD_IMPORT;
use crate::abi::DECODE_EXPORT;
use crate::abi::HOST_LOG_IMPORT;
use crate::abi::HOST_MODULE;
use crate::abi::MEMORY_EXPORT;
use crate::message::decode_message;

/// Host-side callback used by a kernel to decode child arrays.
///
/// When a guest encounters a child node it cannot (or does not want to) decode itself, it calls
/// the `vx_decode_child` host import. The kernel forwards that to this trait, which is expected to
/// decode the child through the [`VortexSession`](vortex_session::VortexSession) and return the
/// result as [`CanonicalMessage`](crate::message) bytes.
pub trait HostDecoder {
    /// Decode the child array at `node_index` (document order within the serialized array header)
    /// and return its `CanonicalMessage` bytes.
    fn decode_child(&self, node_index: usize) -> VortexResult<Vec<u8>>;
}

/// Mutable host state threaded through a single `decode` call.
struct HostState<'a> {
    decoder: &'a dyn HostDecoder,
    /// Captures a host-side error raised inside an import so it can surface as the decode error
    /// rather than an opaque wasm trap.
    error: Option<VortexError>,
}

/// A compiled, reusable WebAssembly decoder kernel.
///
/// Compilation (the expensive step) happens once in [`WasmKernel::new`]. Each [`decode`] call
/// instantiates a fresh store and memory so that decodes are independent.
///
/// [`decode`]: WasmKernel::decode
pub struct WasmKernel {
    engine: Engine,
    module: Module,
}

impl WasmKernel {
    /// Compile a kernel from raw `.wasm` bytes.
    pub fn new(wasm_bytes: impl AsRef<[u8]>) -> VortexResult<Self> {
        let engine = Engine::default();
        let module = Module::new(&engine, wasm_bytes.as_ref())
            .map_err(|e| vortex_err!("failed to compile wasm kernel: {e}"))?;
        Ok(Self { engine, module })
    }

    /// Decode the serialized array `input`, servicing child decodes through `decoder`.
    ///
    /// `input` is the serialized Vortex array (flatbuffer header followed by its data buffers) for
    /// the wasm-encoded node. It may be empty if the guest sources all of its data through
    /// `vx_decode_child`.
    pub fn decode(&self, input: &[u8], decoder: &dyn HostDecoder) -> VortexResult<ArrayRef> {
        let mut store = Store::new(
            &self.engine,
            HostState {
                decoder,
                error: None,
            },
        );
        let mut linker = Linker::<HostState>::new(&self.engine);

        linker
            .func_wrap(
                HOST_MODULE,
                DECODE_CHILD_IMPORT,
                |mut caller: Caller<'_, HostState>, node_index: i32, out_ptr: i32| -> i32 {
                    match host_decode_child(&mut caller, node_index, out_ptr) {
                        Ok(()) => 0,
                        Err(e) => {
                            caller.data_mut().error = Some(e);
                            -1
                        }
                    }
                },
            )
            .map_err(|e| vortex_err!("failed to link {DECODE_CHILD_IMPORT}: {e}"))?;

        linker
            .func_wrap(
                HOST_MODULE,
                HOST_LOG_IMPORT,
                |caller: Caller<'_, HostState>, ptr: i32, len: i32| {
                    if let Some(mem) = caller
                        .get_export(MEMORY_EXPORT)
                        .and_then(Extern::into_memory)
                    {
                        let mut buf = vec![0u8; len.max(0) as usize];
                        if mem.read(&caller, ptr.max(0) as usize, &mut buf).is_ok()
                            && let Ok(s) = std::str::from_utf8(&buf)
                        {
                            tracing_log(s);
                        }
                    }
                },
            )
            .map_err(|e| vortex_err!("failed to link {HOST_LOG_IMPORT}: {e}"))?;

        let instance = linker
            .instantiate_and_start(&mut store, &self.module)
            .map_err(|e| vortex_err!("failed to instantiate wasm kernel: {e}"))?;

        let memory = instance
            .get_memory(&store, MEMORY_EXPORT)
            .ok_or_else(|| vortex_err!("wasm kernel does not export memory '{MEMORY_EXPORT}'"))?;

        let alloc = instance
            .get_typed_func::<i32, i32>(&store, ALLOC_EXPORT)
            .map_err(|e| vortex_err!("wasm kernel missing {ALLOC_EXPORT}: {e}"))?;

        let input_len = i32::try_from(input.len())?;
        let input_ptr = if input.is_empty() {
            0
        } else {
            let ptr = alloc
                .call(&mut store, input_len)
                .map_err(|e| map_trap(&mut store, e))?;
            memory
                .write(&mut store, ptr.max(0) as usize, input)
                .map_err(|e| vortex_err!("failed to write input to guest memory: {e}"))?;
            ptr
        };

        let decode = instance
            .get_typed_func::<(i32, i32), i32>(&store, DECODE_EXPORT)
            .map_err(|e| vortex_err!("wasm kernel missing {DECODE_EXPORT}: {e}"))?;

        let result_ptr = decode
            .call(&mut store, (input_ptr, input_len))
            .map_err(|e| map_trap(&mut store, e))?;
        if result_ptr < 0 {
            if let Some(err) = store.data_mut().error.take() {
                return Err(err);
            }
            vortex_bail!("wasm kernel {DECODE_EXPORT} returned error code {result_ptr}");
        }

        let mut len_bytes = [0u8; 4];
        memory
            .read(&store, result_ptr as usize, &mut len_bytes)
            .map_err(|e| vortex_err!("failed to read result length: {e}"))?;
        let msg_len = u32::from_le_bytes(len_bytes) as usize;
        let mut msg = vec![0u8; msg_len];
        memory
            .read(&store, result_ptr as usize + 4, &mut msg)
            .map_err(|e| vortex_err!("failed to read result message: {e}"))?;

        decode_message(&msg)
    }
}

fn tracing_log(message: &str) {
    // Kept deliberately simple; kernels log only when debugging.
    eprintln!("[wasm kernel] {message}");
}

/// Service a `vx_decode_child` import call.
fn host_decode_child(
    caller: &mut Caller<'_, HostState>,
    node_index: i32,
    out_ptr: i32,
) -> VortexResult<()> {
    let node_index = usize::try_from(node_index)?;
    let msg = caller.data().decoder.decode_child(node_index)?;

    let alloc = caller
        .get_export(ALLOC_EXPORT)
        .and_then(Extern::into_func)
        .ok_or_else(|| vortex_err!("guest missing {ALLOC_EXPORT} export"))?
        .typed::<i32, i32>(&*caller)
        .map_err(|e| vortex_err!("guest {ALLOC_EXPORT} has wrong signature: {e}"))?;
    let dst = alloc
        .call(&mut *caller, i32::try_from(msg.len())?)
        .map_err(|e| vortex_err!("guest {ALLOC_EXPORT} trapped: {e}"))?;

    let memory = caller
        .get_export(MEMORY_EXPORT)
        .and_then(Extern::into_memory)
        .ok_or_else(|| vortex_err!("guest missing memory export"))?;
    memory
        .write(&mut *caller, dst.max(0) as usize, &msg)
        .map_err(|e| vortex_err!("failed to write child message into guest memory: {e}"))?;

    let mut out = [0u8; 8];
    out[0..4].copy_from_slice(&(dst as u32).to_le_bytes());
    out[4..8].copy_from_slice(&(msg.len() as u32).to_le_bytes());
    memory
        .write(&mut *caller, out_ptr.max(0) as usize, &out)
        .map_err(|e| vortex_err!("failed to write decode_child out-params: {e}"))?;
    Ok(())
}

/// Prefer a host-side error stashed during an import over an opaque wasm trap.
fn map_trap(store: &mut Store<HostState>, err: wasmi::Error) -> VortexError {
    store
        .data_mut()
        .error
        .take()
        .unwrap_or_else(|| vortex_err!("wasm kernel trapped: {err}"))
}
