// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::sync::LazyLock;

use prost::Message;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::session::DTypeSession;
use vortex_array::memory::MemorySession;
use vortex_array::session::ArraySession;
use vortex_session::VortexSession;
use vortex_wasm_plugin::ABI_VERSION;
use vortex_wasm_plugin::CanonicalizeRequest;
use vortex_wasm_plugin::ChildConstraint;
use vortex_wasm_plugin::EncodingManifest;
use vortex_wasm_plugin::GuestManifest;
use vortex_wasm_plugin::build_canonicalize_response;
use vortex_wasm_plugin::decode_canonicalize_request;
use vortex_wasm_plugin::pack_ptr_len;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<DTypeSession>()
        .with::<ArraySession>()
        .with::<MemorySession>();
    vortex_fastlanes::initialize(&session);
    session
});

static MANIFEST_BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
    GuestManifest {
        abi_version: u32::from(ABI_VERSION),
        encodings: vec![EncodingManifest {
            id: "fastlanes.for".to_string(),
            validity_from_child: Some(0),
            child_constraints: vec![ChildConstraint {
                slot_name: "encoded".to_string(),
                encoding_id: "vortex.primitive".to_string(),
            }],
        }],
    }
    .encode_to_vec()
});

#[unsafe(no_mangle)]
pub extern "C" fn vx_manifest() -> u64 {
    pack_ptr_len(
        MANIFEST_BYTES.as_ptr() as u32,
        u32::try_from(MANIFEST_BYTES.len()).expect("manifest fits in u32"),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn vx_alloc(size: u32) -> u32 {
    let mut bytes = Vec::<u8>::with_capacity(size as usize);
    let ptr = bytes.as_mut_ptr();
    mem::forget(bytes);
    ptr as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn vx_free(ptr: u32, size: u32) {
    if ptr == 0 {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(ptr as *mut u8, 0, size as usize));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn vx_canonicalize_array(req_ptr: u32, req_len: u32) -> u64 {
    let req_bytes = unsafe { std::slice::from_raw_parts(req_ptr as *const u8, req_len as usize) };
    let request = CanonicalizeRequest::decode(req_bytes).expect("request must decode");
    let (_dtype, _len, _ctx, array) =
        decode_canonicalize_request(&request, &SESSION).expect("request array must decode");
    assert_eq!(array.encoding_id().as_str(), "fastlanes.for");

    let mut ctx = SESSION.create_execution_ctx();
    let canonical = array
        .execute::<PrimitiveArray>(&mut ctx)
        .expect("FoR guest canonicalization must succeed");
    let response = build_canonicalize_response(&canonical.into_array(), &SESSION)
        .expect("response serialization must succeed");

    let mut bytes = response.encode_to_vec();
    bytes.shrink_to_fit();
    assert_eq!(bytes.len(), bytes.capacity());
    let ptr = bytes.as_mut_ptr();
    let len = bytes.len();
    mem::forget(bytes);
    pack_ptr_len(ptr as u32, u32::try_from(len).expect("response fits in u32"))
}
