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

#[cfg(test)]
mod test {
    use vortex_array::ProstMetadata;
    use vortex_array::dtype::PType;
    use vortex_array::test_harness::check_metadata;

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
