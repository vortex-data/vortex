// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::StructFields;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::execute::PackSource;
use crate::execution::operator::*;
use crate::execution::pipeline::Arena;
use crate::execution::pipeline::port::SourceEnd;
use crate::execution::pipeline::synthetic::row_dtype;
use crate::test::SESSION;

#[rstest]
fn pack_readiness_tracks_changes_and_consumption(
    #[values(2, 3)] second_len: u64,
) -> VortexResult<()> {
    let mut arena = Arena::default();
    let inlets = [
        arena.create(8, None, SourceEnd::Unclaimed),
        arena.create(8, None, SourceEnd::Unclaimed),
    ];
    let mut bytes = VecDeque::new();
    let mut reads = Default::default();
    let mut exec = SESSION.create_execution_ctx();
    let mut touched = InletSet::new(2);
    let mut changed = InletSet::all(2);
    let mut cx = Cx {
        arena: &mut arena,
        inlets: &inlets,
        bytes: &mut bytes,
        reads: &mut reads,
        session: &SESSION,
        exec: &mut exec,
        touched: &mut touched,
        changed: &mut changed,
    };
    let fields = StructFields::from_iter([("a", row_dtype()), ("b", row_dtype())]);
    let mut pack = PackSource::new(fields, false, 2);
    assert!(matches!(
        pack.ready(&mut cx)?,
        SourceReady::Blocked(SourceBlocked::Inlet(_))
    ));

    cx.arena
        .push(inlets[0], Buffer::from_iter(0_u64..2).into_array());
    cx.arena.close(inlets[0]);
    cx.changed.insert(0);
    assert_eq!(
        pack.ready(&mut cx)?,
        SourceReady::Blocked(SourceBlocked::Inlet(1))
    );
    cx.touched.clear();
    assert_eq!(
        pack.ready(&mut cx)?,
        SourceReady::Blocked(SourceBlocked::Inlet(1))
    );
    assert!(
        cx.touched.indices().is_empty(),
        "unchanged inlets were scanned"
    );

    cx.arena
        .push(inlets[1], Buffer::from_iter(0..second_len).into_array());
    cx.arena.close(inlets[1]);
    cx.changed.insert(1);
    assert_eq!(pack.ready(&mut cx)?, SourceReady::Ready);
    assert_eq!(cx.touched.indices(), &[1]);
    let Step::Final(array) = pack.compute(Input::None, &mut cx)? else {
        return Err(vortex_err!(
            "expected Pack's last currently available batch"
        ));
    };
    let expected = StructArray::from_fields(&[
        ("a", Buffer::from_iter(0_u64..2).into_array()),
        ("b", Buffer::from_iter(0_u64..2).into_array()),
    ])?;
    assert_arrays_eq!(array, expected.into_array(), cx.exec());

    // Computing consumed every available row from the first inlet, without writer changes.
    cx.touched.clear();
    let ready = pack.ready(&mut cx);
    assert!(
        cx.touched.indices().is_empty(),
        "consumed inlets were rescanned"
    );
    if second_len == 2 {
        assert_eq!(ready?, SourceReady::Ready);
        assert!(matches!(pack.compute(Input::None, &mut cx)?, Step::End));
    } else {
        let error = ready.expect_err("a remaining row must fail Pack readiness");
        assert!(
            error
                .to_string()
                .contains("Pack fields ended at different rows")
        );
    }
    Ok(())
}
