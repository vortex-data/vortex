// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use array::*;
pub use compress::*;

mod array;
mod canonical;
mod compress;
mod timestamp;

use vortex_array::session::ArraySessionExt;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_session::VortexSession;

/// Initialize datetime-parts encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.scalar_fns().register(DateTimeParts);
    session.arrays().register(DateTimePartsArrayPlugin);
    session.arrays().register(LegacyDateTimePartsArrayPlugin);
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
