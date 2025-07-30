// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod read;
mod write;

#[cxx::bridge(namespace = "vortex::ffi")]
mod ffi {}
