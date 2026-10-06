// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The candidates every chunk is compressed with, by name.
//!
//! - `production`: today's compressor, `BtrBlocksCompressorBuilder::from_session` (default
//!   encodings only).
//! - `sizemodel`: production with RunEnd and Sparse estimated from a size model.
//! - `forced/<scheme>`: `<scheme>` forced at the root, production below it.

use std::collections::BTreeMap;

use anyhow::bail;
use vortex::session::VortexSession;
use vortex_array::Canonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::CompressionSessionExt;
use vortex_btrblocks::SchemeExt;
use vortex_compressor::scheme::Scheme;

use crate::wrap::Mode;
use crate::wrap::Wrapped;

/// The integer schemes registered in the session, in registration order.
pub fn int_schemes(session: &VortexSession) -> Vec<&'static dyn Scheme> {
    let probe = Canonical::Primitive(PrimitiveArray::new(
        vortex::buffer::buffer![1i64, 2, 3],
        Validity::NonNullable,
    ));
    session
        .compression()
        .schemes()
        .iter()
        .copied()
        .filter(|s| s.matches(&probe))
        .collect()
}

/// The short name used in candidate names, e.g. `for` for `vortex.int.for`.
pub fn short_name(scheme: &dyn Scheme) -> &'static str {
    let name = scheme.scheme_name();
    name.rsplit('.').next().unwrap_or(name)
}

/// Production with every integer scheme replaced, keeping registration order, wrapping the ones
/// `wrap` picks.
pub fn replaced(
    session: &VortexSession,
    wrap: &dyn Fn(&'static dyn Scheme) -> Option<Mode>,
) -> BtrBlocksCompressor {
    let schemes = int_schemes(session);
    let ids: Vec<_> = schemes.iter().map(|s| s.id()).collect();
    let mut builder = BtrBlocksCompressorBuilder::from_session(session).exclude_schemes(ids);
    for scheme in &schemes {
        builder = builder.with_new_scheme(match wrap(*scheme) {
            Some(mode) => Wrapped::leak(*scheme, mode),
            None => *scheme,
        });
    }
    builder.build()
}

/// Builds the named candidates.
pub fn build(
    session: &VortexSession,
    names: &[String],
) -> anyhow::Result<BTreeMap<String, BtrBlocksCompressor>> {
    let schemes = int_schemes(session);
    names
        .iter()
        .map(|name| {
            let compressor = match name.as_str() {
                "production" => BtrBlocksCompressorBuilder::from_session(session).build(),
                "sizemodel" => replaced(session, &|_| Some(Mode::SizeModel)),
                other => {
                    let Some(short) = other.strip_prefix("forced/") else {
                        bail!("unknown candidate `{other}`");
                    };
                    let Some(scheme) = schemes.iter().find(|s| short_name(**s) == short) else {
                        bail!("no integer scheme named `{short}`");
                    };
                    BtrBlocksCompressorBuilder::from_session(session)
                        .with_new_scheme(Wrapped::leak(*scheme, Mode::ForceRoot))
                        .build()
                }
            };
            Ok((name.clone(), compressor))
        })
        .collect()
}
