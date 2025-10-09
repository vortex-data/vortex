// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::ConstantArray;

mod compute;

mod vtable;
pub use vtable::{
    ConstantEncoding,
    ConstantVTable,
};
