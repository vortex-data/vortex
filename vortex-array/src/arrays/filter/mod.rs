// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::FilterArrayParts;
pub use array::FilterData;
pub use vtable::FilterArray;

mod execute;

mod kernel;
pub use kernel::FilterExecuteAdaptor;
pub use kernel::FilterKernel;
pub use kernel::FilterReduce;
pub use kernel::FilterReduceAdaptor;

mod rules;

mod vtable;
pub use vtable::Filter;
