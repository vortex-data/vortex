#![allow(special_module_name)]

mod lib;

// To build the Wasm target, a `staticlib` crate-type is required
//
// This is different than the default needed in native, and there is
// currently no way to select crate-type depending on target.
//
// This file sole purpose is remapping the content of lib as an
// example, do not change the content of the file.
//
// To build the Wasm target explicitly, use:
//   cargo build --example $PACKAGE_NAME
