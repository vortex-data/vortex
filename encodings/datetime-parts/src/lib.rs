// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use array::*;
pub use compress::*;
use vortex_array::session::ArraySessionExt;
use vortex_array::vtable::ArrayVTableExt;
use vortex_session::VortexSession;

mod array;
mod canonical;
mod compress;
mod compute;
mod ops;
mod rules;
mod timestamp;

/// Initialize the DateTimeParts encoding in the given session.
pub fn initialize(session: &mut VortexSession) {
    session.arrays().register(DateTimePartsVTable.as_vtable());
    // session
    //     .arrays_mut()
    //     .optimizer_mut()
    //     .register_reduce_rule(DateTimePartsExpandRule);
}

#[cfg(test)]
mod test {
    use vortex_array::test_harness::check_metadata;
    use vortex_array::ProstMetadata;
    use vortex_dtype::PType;

    use crate::DateTimePartsMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_datetimeparts_metadata() {
        check_metadata(
            "datetimeparts.metadata",
            ProstMetadata(DateTimePartsMetadata {
                days_ptype: PType::I64 as i32,
                seconds_ptype: PType::I64 as i32,
                subseconds_ptype: PType::I64 as i32,
            }),
        );
    }
}
