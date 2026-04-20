// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tiny shared slot used by `vortex-array` and `vortex-arrow` to register
//! Arrow-backed compute fallbacks.
//!
//! This crate exists for one reason: under Cargo's test build, `vortex-array`
//! may be linked into the test binary TWICE - once as the `--test` target, and
//! once as a dependency of `vortex-arrow`. A `static` declared inside
//! `vortex-array` has two distinct addresses in that case, which breaks a
//! straightforward runtime registration. Putting the slot in a separate
//! crate that is compiled exactly once (because it has no `[features]` or
//! test-specific codegen) gives us a single shared slot that both copies of
//! `vortex-array` see.
//!
//! The value stored here is opaque (`*const ()`). `vortex-array` provides
//! strongly-typed wrappers around it.

use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering;

static SLOT: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

/// Install a raw pointer in the global slot. Subsequent calls are ignored;
/// the first installed pointer wins. Returns `true` if this call performed
/// the install.
pub fn set(ptr: *const ()) -> bool {
    SLOT.compare_exchange(
        std::ptr::null_mut(),
        ptr as *mut (),
        Ordering::AcqRel,
        Ordering::Acquire,
    )
    .is_ok()
}

/// Return the raw pointer installed in the global slot, or null if nothing is
/// installed.
pub fn get() -> *const () {
    SLOT.load(Ordering::Acquire) as *const ()
}
