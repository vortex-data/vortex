// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::emit::{ArgKey, EmitCtx, SigBuilder};
use crate::form::PType;
use crate::stage::{JitStage, Placement};

/// Post-loop stage: scatter sparse patch values into the assembled output.
///
/// Runs after the block loop. Calls into a Rust helper because random scatter
/// is awkward to express as block-aligned IR and rarely on the hot path
/// (patches are sparse by construction).
///
/// The expected Rust helper has the signature
/// `unsafe extern "C" fn(out_ptr: *mut T, idx: *const u64, val: *const T, n: u64)`,
/// monomorphized per `ptype` and registered with `Compiler::new`.
#[derive(Debug, Clone, Copy)]
pub struct ApplyPatchesPostLoop {
    pub ptype: PType,
    /// Name of the registered extern symbol that scatters patches for this ptype.
    pub helper_name: &'static str,
}

impl JitStage for ApplyPatchesPostLoop {
    fn tag(&self) -> &'static str {
        "ApplyPatchesPostLoop"
    }

    fn fingerprint(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(1 + self.helper_name.len());
        v.push(self.ptype as u8);
        v.extend_from_slice(self.helper_name.as_bytes());
        v
    }

    fn placement(&self) -> Placement {
        Placement::PostLoop
    }

    fn declare(&self, sig: &mut SigBuilder) {
        sig.request_arg(ArgKey::OutPtr);
        sig.request_arg(ArgKey::Named("patch_idx"));
        sig.request_arg(ArgKey::Named("patch_val"));
        sig.request_arg(ArgKey::Named("patch_n"));
        sig.request_extern(self.helper_name);
    }

    fn emit(&self, cx: &mut EmitCtx<'_, '_>) -> VortexResult<()> {
        let out_ptr = cx.runtime_arg(&ArgKey::OutPtr)?;
        let idx_ptr = cx.runtime_arg(&ArgKey::Named("patch_idx"))?;
        let val_ptr = cx.runtime_arg(&ArgKey::Named("patch_val"))?;
        let n_ptr = cx.runtime_arg(&ArgKey::Named("patch_n"))?;
        // `patch_n` is passed as a pointer to a u64 so the kernel signature
        // stays a pure pointer-of-bytes contract.
        let n = cx.load_lane(n_ptr, 0, PType::I64);
        drop(cx.extern_call_by_name(self.helper_name, &[out_ptr, idx_ptr, val_ptr, n]));
        Ok(())
    }
}
