// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::FilterArray;
pub use array::FilterArrayParts;

mod execute;

mod kernel;
pub use kernel::FilterExecuteAdaptor;
pub use kernel::FilterKernel;
pub use kernel::FilterReduce;
pub use kernel::FilterReduceAdaptor;

mod rules;

mod vtable;
pub use vtable::FilterVTable;
