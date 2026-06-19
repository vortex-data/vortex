// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use array::*;
pub use compress::*;

mod array;
mod canonical;
mod compress;
mod compute;
mod ops;
mod timestamp;

use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::session::ArraySession;
use vortex_session::VortexSessionBuilder;

/// Initialize datetime-parts encoding in the given session.
pub fn initialize(session: &mut VortexSessionBuilder) {
    session.get_mut::<ArraySession>().register(DateTimeParts);
    compute::kernel::initialize(session);

    session
        .get_mut::<AggregateFnSession>()
        .register_aggregate_kernel(
            DateTimeParts.id(),
            Some(IsConstant.id()),
            &compute::is_constant::DateTimePartsIsConstantKernel,
        );
}

#[cfg(test)]
mod test {
    use prost::Message;
    use vortex_array::dtype::PType;
    use vortex_array::test_harness::check_metadata;

    use crate::DateTimePartsMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_datetimeparts_metadata() {
        check_metadata(
            "datetimeparts.metadata",
            &DateTimePartsMetadata {
                days_ptype: PType::I64 as i32,
                seconds_ptype: PType::I64 as i32,
                subseconds_ptype: PType::I64 as i32,
            }
            .encode_to_vec(),
        );
    }
}
