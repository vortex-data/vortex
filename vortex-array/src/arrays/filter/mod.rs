// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::FilterArrayExt;
pub use array::FilterData;
pub use array::FilterDataParts;
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

pub(crate) fn initialize(session: &mut vortex_session::VortexSessionBuilder) {
    kernel::initialize(session);
}
