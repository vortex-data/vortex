// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::FilterArraySlotsExt;
pub use array::FilterData;
pub use array::FilterDataParts;
pub use array::FilterSlots;
pub use array::FilterSlotsView;
pub use vtable::FilterArray;

mod execute;
pub(crate) use execute::buffer::filter_buffer;
pub(crate) use execute::byte_compress::filter_buffer as filter_buffer_byte_compress;
pub(crate) use execute::filter_validity;

mod kernel;
pub use kernel::FilterExecuteAdaptor;
pub use kernel::FilterKernel;
pub use kernel::FilterReduce;
pub use kernel::FilterReduceAdaptor;

mod rules;

mod vtable;
pub use vtable::Filter;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    kernel::initialize(session);
}
