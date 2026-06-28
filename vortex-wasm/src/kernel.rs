// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The embedded WebAssembly decode runtime.
//!
//! [`WasmKernel`] wraps a compiled `wasmtime` module and drives the host/guest ABI: it copies the
//! kernel input into guest memory, calls the guest's `vx_decode` export, services `vx_decode_child`
//! callbacks from a [`HostDecoder`], and reconstructs a Vortex array from the Arrow C Data
//! Interface structs the guest returns (see [`crate::arrow_ffi`]).
//!
//! Kernels are untrusted file data. The runtime is `wasmtime` with its default Cranelift backend
//! (not Winch/Pulley, which are less battle-tested); each decode runs in a fresh [`Store`] whose
//! linear memory growth is capped via [`StoreLimits`]. CPU-time bounding (fuel / epoch
//! interruption) is a planned follow-up — see `docs/design/wasm-encodings.md`.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use wasmtime::Caller;
use wasmtime::Engine;
use wasmtime::Extern;
use wasmtime::Linker;
use wasmtime::Memory;
use wasmtime::Module;
use wasmtime::ResourceLimiter;
use wasmtime::Store;
use wasmtime::StoreLimits;
use wasmtime::StoreLimitsBuilder;
use wasmtime::TypedFunc;

use crate::abi::ALLOC_EXPORT;
use crate::abi::DECODE_CHILD_IMPORT;
use crate::abi::DECODE_EXPORT;
use crate::abi::HOST_LOG_IMPORT;
use crate::abi::HOST_MODULE;
use crate::abi::MEMORY_EXPORT;
use crate::arrow_ffi;
use crate::arrow_ffi::GuestMem;

/// Maximum linear memory a kernel may grow to in a single decode, as a coarse DoS guard against
/// untrusted kernels. Generous enough for legitimate decodes (wasm32 memory tops out at 4 GiB); a
/// starting value, not a tuned one.
const MAX_GUEST_MEMORY_BYTES: usize = 1 << 30;

/// Host-side callback used by a kernel to decode child arrays.
///
/// When a guest needs a decoded child it calls the `vx_decode_child` host import. The kernel
/// forwards that here; the implementation decodes the child through the
/// [`VortexSession`](vortex_session::VortexSession) and returns it as a canonical array, which the
/// kernel then hands to the guest as Arrow C Data Interface structs.
///
/// `Send + Sync` so it can live in a `wasmtime::Store`, whose data must be `'static`.
pub trait HostDecoder: Send + Sync {
    /// Decode the child array at `node_index` and return it in canonical form.
    fn decode_child(&self, node_index: usize) -> VortexResult<Canonical>;
}

/// Host state threaded through a single `decode` call.
///
/// `wasmtime::Store` requires its data to be `'static`, so this owns the decoder and session rather
/// than borrowing them (unlike the previous `wasmi` runtime, which allowed borrowed store data).
struct HostState {
    decoder: Arc<dyn HostDecoder>,
    session: VortexSession,
    /// Captures a host-side error raised inside an import so it can surface as the decode error
    /// rather than an opaque wasm trap.
    error: Option<VortexError>,
    /// Caps the guest's linear-memory growth for this decode.
    limits: StoreLimits,
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

    /// Decode `input`, servicing child decodes through `decoder`.
    ///
    /// `input` is the encoding-specific bytes the kernel consumes. Child decodes and the kernel's
    /// result cross the boundary as Arrow C Data Interface structs; `session` is used to encode the
    /// host-decoded children.
    pub fn decode(
        &self,
        input: &[u8],
        decoder: Arc<dyn HostDecoder>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let mut store = Store::new(
            &self.engine,
            HostState {
                decoder,
                session: session.clone(),
                error: None,
                limits: StoreLimitsBuilder::new()
                    .memory_size(MAX_GUEST_MEMORY_BYTES)
                    .build(),
            },
        );
        store.limiter(|state| &mut state.limits as &mut dyn ResourceLimiter);

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
                |mut caller: Caller<'_, HostState>, ptr: i32, len: i32| {
                    if let Some(mem) = caller
                        .get_export(MEMORY_EXPORT)
                        .and_then(Extern::into_memory)
                    {
                        let mut buf = vec![0u8; len.max(0) as usize];
                        if mem.read(&caller, ptr.max(0) as usize, &mut buf).is_ok()
                            && let Ok(s) = std::str::from_utf8(&buf)
                        {
                            eprintln!("[wasm kernel] {s}");
                        }
                    }
                },
            )
            .map_err(|e| vortex_err!("failed to link {HOST_LOG_IMPORT}: {e}"))?;

        let instance = linker
            .instantiate(&mut store, &self.module)
            .map_err(|e| vortex_err!("failed to instantiate wasm kernel: {e}"))?;

        let memory = instance
            .get_memory(&mut store, MEMORY_EXPORT)
            .ok_or_else(|| vortex_err!("wasm kernel does not export memory '{MEMORY_EXPORT}'"))?;
        let alloc = instance
            .get_typed_func::<i32, i32>(&mut store, ALLOC_EXPORT)
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
            .get_typed_func::<(i32, i32), i32>(&mut store, DECODE_EXPORT)
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

        // The result is a pointer to an (array_ptr: u32, schema_ptr: u32) pair.
        let mut pair = [0u8; 8];
        memory
            .read(&store, result_ptr as usize, &mut pair)
            .map_err(|e| vortex_err!("failed to read result pair: {e}"))?;
        let array_ptr = u32::from_le_bytes(pair[0..4].try_into().expect("4 bytes"));
        let schema_ptr = u32::from_le_bytes(pair[4..8].try_into().expect("4 bytes"));

        arrow_ffi::import(memory.data(&store), array_ptr, schema_ptr)
    }
}

/// A [`GuestMem`] that allocates via the guest's exported `vx_alloc` and writes through `wasmtime`.
struct CallerGuestMem<'c, 'b> {
    caller: &'c mut Caller<'b, HostState>,
    memory: Memory,
    alloc: TypedFunc<i32, i32>,
}

impl GuestMem for CallerGuestMem<'_, '_> {
    fn alloc(&mut self, len: u32) -> VortexResult<u32> {
        let ptr = self
            .alloc
            .call(&mut *self.caller, i32::try_from(len)?)
            .map_err(|e| vortex_err!("guest {ALLOC_EXPORT} trapped: {e}"))?;
        Ok(ptr.max(0) as u32)
    }

    fn write(&mut self, off: u32, bytes: &[u8]) -> VortexResult<()> {
        self.memory
            .write(&mut *self.caller, off as usize, bytes)
            .map_err(|e| vortex_err!("failed to write guest memory: {e}"))
    }
}

/// Service a `vx_decode_child` import call: decode the child, export it as Arrow C structs into
/// guest memory, and write the resulting `(array_ptr, schema_ptr)` pair at `out_ptr`.
fn host_decode_child(
    caller: &mut Caller<'_, HostState>,
    node_index: i32,
    out_ptr: i32,
) -> VortexResult<()> {
    let node_index = usize::try_from(node_index)?;
    let canonical = caller.data().decoder.decode_child(node_index)?;
    let mut ctx: ExecutionCtx = caller.data().session.create_execution_ctx();

    let memory = caller
        .get_export(MEMORY_EXPORT)
        .and_then(Extern::into_memory)
        .ok_or_else(|| vortex_err!("guest missing memory export"))?;
    let alloc = caller
        .get_export(ALLOC_EXPORT)
        .and_then(Extern::into_func)
        .ok_or_else(|| vortex_err!("guest missing {ALLOC_EXPORT} export"))?
        .typed::<i32, i32>(&*caller)
        .map_err(|e| vortex_err!("guest {ALLOC_EXPORT} has wrong signature: {e}"))?;

    let (array_ptr, schema_ptr) = {
        let mut guest_mem = CallerGuestMem {
            caller,
            memory,
            alloc,
        };
        arrow_ffi::export(&canonical, &mut ctx, &mut guest_mem)?
    };

    let mut out = [0u8; 8];
    out[0..4].copy_from_slice(&array_ptr.to_le_bytes());
    out[4..8].copy_from_slice(&schema_ptr.to_le_bytes());
    memory
        .write(&mut *caller, out_ptr.max(0) as usize, &out)
        .map_err(|e| vortex_err!("failed to write decode_child out-params: {e}"))?;
    Ok(())
}

/// Prefer a host-side error stashed during an import over an opaque wasm trap.
fn map_trap(store: &mut Store<HostState>, err: impl std::fmt::Display) -> VortexError {
    store
        .data_mut()
        .error
        .take()
        .unwrap_or_else(|| vortex_err!("wasm kernel trapped: {err}"))
}
