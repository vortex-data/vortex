// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::TakeArray;
pub use array::TakeArrayParts;

mod execute;

mod kernel;
pub use kernel::TakeExecute;
pub use kernel::TakeExecuteAdaptor;
pub use kernel::TakeReduce;
pub use kernel::TakeReduceAdaptor;

mod rules;

mod vtable;
pub use vtable::TakeVTable;
