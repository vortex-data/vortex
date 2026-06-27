// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end exercise of the host/guest ABI using a hand-written WAT kernel.
//!
//! The kernel imports `vortex_host.vx_decode_child`, asks the host to decode child 0, and echoes
//! the resulting `CanonicalMessage` back as its own decode output. This proves both directions of
//! the boundary (host -> guest child delivery and guest -> host result) plus the
//! `CanonicalMessage` wire format, without needing a full Rust-to-wasm guest build.

use vortex_array::Canonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_wasm::HostDecoder;
use vortex_wasm::WasmKernel;
use vortex_wasm::message::encode_canonical;

/// A kernel that delegates entirely to the host: it requests child 0 and returns it verbatim.
const ECHO_KERNEL_WAT: &str = r#"
(module
  (import "vortex_host" "vx_decode_child" (func $decode_child (param i32 i32) (result i32)))
  (memory (export "memory") 4)
  (global $next (mut i32) (i32.const 1024))

  (func $alloc (export "vx_alloc") (param $len i32) (result i32)
    (local $p i32)
    (local.set $p (global.get $next))
    (global.set $next (i32.add (global.get $next) (local.get $len)))
    (local.get $p))

  (func (export "vx_decode") (param $in_ptr i32) (param $in_len i32) (result i32)
    (local $out i32)
    (local $child_off i32)
    (local $child_len i32)
    (local $res i32)
    ;; scratch region for the (offset, len) out-params
    (local.set $out (call $alloc (i32.const 8)))
    ;; ask the host to decode child 0 into guest memory
    (drop (call $decode_child (i32.const 0) (local.get $out)))
    (local.set $child_off (i32.load (local.get $out)))
    (local.set $child_len (i32.load offset=4 (local.get $out)))
    ;; result buffer = [u32 len][child message bytes]
    (local.set $res (call $alloc (i32.add (i32.const 4) (local.get $child_len))))
    (i32.store (local.get $res) (local.get $child_len))
    (memory.copy
      (i32.add (local.get $res) (i32.const 4))
      (local.get $child_off)
      (local.get $child_len))
    (local.get $res)))
"#;

struct EchoDecoder {
    message: Vec<u8>,
}

impl HostDecoder for EchoDecoder {
    fn decode_child(&self, node_index: usize) -> VortexResult<Vec<u8>> {
        assert_eq!(node_index, 0, "echo kernel only requests child 0");
        Ok(self.message.clone())
    }
}

#[test]
fn echo_kernel_round_trips_a_primitive_child() -> VortexResult<()> {
    let wasm = wat::parse_str(ECHO_KERNEL_WAT).expect("valid wat");
    let kernel = WasmKernel::new(&wasm)?;

    let mut ctx = vortex_array::array_session().create_execution_ctx();
    let child = PrimitiveArray::new(buffer![10i64, 20, 30, 40], Validity::NonNullable);
    let decoder = EchoDecoder {
        message: encode_canonical(&Canonical::Primitive(child), &mut ctx)?,
    };

    let decoded = kernel.decode(&[], &decoder)?;
    assert_eq!(decoded.len(), 4);
    let expected: Vec<u8> = [10i64, 20, 30, 40]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(decoded.buffers()[0].as_ref(), expected.as_slice());
    Ok(())
}
