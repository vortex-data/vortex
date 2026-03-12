#![allow(dead_code)]

use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use super::Fixture;

macro_rules! encoding_stub {
    ($name:ident, $file:expr) => {
        pub struct $name;

        impl Fixture for $name {
            fn name(&self) -> &str {
                $file
            }

            fn build(&self) -> VortexResult<Vec<ArrayRef>> {
                todo!(concat!("blocked on stable-encodings RFC — ", $file))
            }
        }
    };
}

encoding_stub!(DictEncodingFixture, "enc_dict.vortex");
encoding_stub!(RunEndEncodingFixture, "enc_runend.vortex");
encoding_stub!(ConstantEncodingFixture, "enc_constant.vortex");
encoding_stub!(SparseEncodingFixture, "enc_sparse.vortex");
encoding_stub!(AlpEncodingFixture, "enc_alp.vortex");
encoding_stub!(BitPackedEncodingFixture, "enc_bitpacked.vortex");
encoding_stub!(FsstEncodingFixture, "enc_fsst.vortex");
