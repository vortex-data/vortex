// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::builders::LazyBitBufferBuilder;
use crate::dtype::Nullability;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::Nullability::Nullable;
use crate::validity::Validity;

/// Accumulates the validity of a nested [`ArrayBuilder`](crate::builders::ArrayBuilder) without
/// materializing a null buffer for it.
///
/// A nested builder learns about validity from two sources: one row at a time, as scalars are
/// appended, and a whole array's worth at a time, as arrays are. Only the former needs a null
/// buffer. An appended array already carries its validity in whatever form it was stored in —
/// [`Validity::AllValid`] and [`Validity::AllInvalid`] cost nothing at all, and an array-backed
/// validity is a bool array that is already built — so this builder keeps those as runs and
/// concatenates them at the end, exactly as [`Validity::concat`] does.
///
/// Materializing them instead would mean executing every appended array's validity into a
/// [`Mask`](vortex_mask::Mask) and copying its bits, which for a builder assembling many chunks is
/// the dominant cost of tracking validity at all.
pub(crate) struct ValidityBuilder {
    /// Completed runs, in logical order, with the number of values each covers. Never contains an
    /// empty run.
    runs: Vec<(Validity, usize)>,

    /// The summed length of `runs`.
    runs_len: usize,

    /// Null buffer holding the bits appended since the last run.
    pending: LazyBitBufferBuilder,
}

impl ValidityBuilder {
    /// Creates a new `ValidityBuilder` whose null buffer is pre-allocated for `capacity` bits.
    pub fn new(capacity: usize) -> Self {
        Self {
            runs: Vec::new(),
            runs_len: 0,
            pending: LazyBitBufferBuilder::new(capacity),
        }
    }

    /// The number of values whose validity has been recorded so far.
    pub fn len(&self) -> usize {
        self.runs_len + self.pending.len()
    }

    /// Records one valid value.
    pub fn append_non_null(&mut self) {
        self.pending.append_non_null()
    }

    /// Records `n` valid values.
    pub fn append_n_non_nulls(&mut self, n: usize) {
        self.pending.append_n_non_nulls(n)
    }

    /// Records `n` null values.
    pub fn append_n_nulls(&mut self, n: usize) {
        self.pending.append_n_nulls(n)
    }

    /// Records the validity of a whole appended array, covering `len` values, as a run of its own.
    ///
    /// However few values the run covers, it is kept as it arrived rather than executed into a
    /// mask, so a builder's validity is split on exactly the boundaries its children are.
    pub fn append_validity(&mut self, validity: Validity, len: usize) {
        if len == 0 {
            return;
        }

        self.flush_pending();
        self.runs_len += len;
        self.runs.push((validity, len));
    }

    /// Allocates space for `additional` more bits in the null buffer.
    pub fn reserve_exact(&mut self, additional: usize) {
        self.pending.reserve_exact(additional)
    }

    /// Finishes the validity, concatenating the accumulated runs.
    ///
    /// # Panics
    ///
    /// Panics if a non-nullable builder recorded a null, matching
    /// [`LazyBitBufferBuilder::finish_with_nullability`].
    pub fn finish_with_nullability(&mut self, nullability: Nullability) -> Validity {
        if self.runs.is_empty() {
            return self.pending.finish_with_nullability(nullability);
        }

        self.flush_pending();
        self.runs_len = 0;
        let runs = std::mem::take(&mut self.runs);

        // `Validity::concat` treats `NonNullable` and `AllValid` as different kinds and falls back
        // to a bool array when both appear, which they do as soon as a non-nullable array is
        // appended next to a scalar. Both mean "no nulls", so answer from the nullability instead.
        if runs
            .iter()
            .all(|(validity, _)| validity.definitely_no_nulls())
        {
            return nullability.into();
        }

        let validity = Validity::concat(runs).vortex_expect("runs is not empty");
        if nullability == NonNullable {
            vortex_panic!("cannot finish a non-nullable builder holding {validity:?} validity");
        }
        validity
    }

    /// Moves whatever the null buffer holds into `runs`, keeping the runs in logical order.
    fn flush_pending(&mut self) {
        let len = self.pending.len();
        if len == 0 {
            return;
        }
        // A run is only ever read back through `Validity::concat`, which takes the nullability
        // from the runs as a whole, so an all-valid null buffer can stay lazy here.
        let validity = self.pending.finish_with_nullability(Nullable);
        self.runs_len += len;
        self.runs.push((validity, len));
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::ValidityBuilder;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Chunked;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::validity::Validity;

    /// An arbitrary run length. `ValidityBuilder` treats no length specially, so the tests only
    /// need a length long enough to tell runs apart.
    const RUN_LEN: usize = 64;

    /// A `Validity` backed by a bool array, which is the case a run avoids executing.
    fn array_backed(len: usize) -> Validity {
        Validity::from_mask(Mask::from_iter((0..len).map(|i| i % 2 == 0)), Nullable)
    }

    #[test]
    fn test_whole_array_validity_is_kept_as_a_run() {
        let mut builder = ValidityBuilder::new(0);

        builder.append_validity(array_backed(RUN_LEN), RUN_LEN);
        builder.append_validity(array_backed(RUN_LEN), RUN_LEN);
        assert_eq!(builder.len(), 2 * RUN_LEN);

        let Validity::Array(array) = builder.finish_with_nullability(Nullable) else {
            panic!("expected array-backed validity");
        };
        assert!(
            array.is::<Chunked>(),
            "the runs should have been concatenated, not copied into one buffer",
        );
    }

    /// Uniform runs collapse instead of becoming a bool array.
    #[test]
    fn test_all_valid_runs_stay_lazy() {
        let mut builder = ValidityBuilder::new(0);

        builder.append_validity(Validity::AllValid, RUN_LEN);
        builder.append_validity(Validity::AllValid, RUN_LEN);

        assert!(matches!(
            builder.finish_with_nullability(Nullable),
            Validity::AllValid
        ));
    }

    /// A one-row validity earns a run too. Uniform runs still collapse, so a builder fed an array
    /// at a time does not pay a bool array for validity it never had.
    #[test]
    fn test_short_validity_is_kept_as_a_run_too() {
        let mut builder = ValidityBuilder::new(0);

        for _ in 0..RUN_LEN {
            builder.append_validity(Validity::AllInvalid, 1);
        }
        assert_eq!(builder.len(), RUN_LEN);

        assert!(matches!(
            builder.finish_with_nullability(Nullable),
            Validity::AllInvalid
        ));
    }

    /// Bits and runs interleave, and have to come back out in the order they went in.
    #[test]
    fn test_bits_and_runs_keep_their_order() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ValidityBuilder::new(0);

        builder.append_n_nulls(1);
        builder.append_validity(Validity::AllValid, RUN_LEN);
        builder.append_non_null();
        builder.append_validity(Validity::AllInvalid, RUN_LEN);

        let validity = builder.finish_with_nullability(Nullable);
        let mask = validity.execute_mask(2 * RUN_LEN + 2, &mut ctx)?;

        let expected = Mask::from_iter(
            [false]
                .into_iter()
                .chain(std::iter::repeat_n(true, RUN_LEN))
                .chain([true])
                .chain(std::iter::repeat_n(false, RUN_LEN)),
        );
        assert_eq!(mask, expected);

        Ok(())
    }

    #[test]
    fn test_non_nullable_finishes_non_nullable() {
        let mut builder = ValidityBuilder::new(0);

        builder.append_validity(Validity::NonNullable, RUN_LEN);
        builder.append_n_non_nulls(1);

        assert!(matches!(
            builder.finish_with_nullability(NonNullable),
            Validity::NonNullable
        ));
    }

    #[test]
    fn test_finish_resets_the_builder() {
        let mut builder = ValidityBuilder::new(0);

        builder.append_validity(Validity::AllInvalid, RUN_LEN);
        assert_eq!(builder.finish_with_nullability(Nullable).maybe_len(), None);

        assert_eq!(builder.len(), 0);
        builder.append_n_nulls(1);
        assert!(matches!(
            builder.finish_with_nullability(Nullable),
            Validity::AllInvalid
        ));
    }
}
