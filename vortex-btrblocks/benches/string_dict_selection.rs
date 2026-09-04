// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(not(codspeed))]
mod benchmarks {
    use std::sync::LazyLock;

    use divan::Bencher;
    use divan::counter::ItemsCount;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_btrblocks::BtrBlocksCompressor;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    #[derive(Clone, Copy)]
    enum Distribution {
        Uniform,
        Clustered,
        VariedPrefix,
    }

    #[derive(Clone, Copy)]
    struct Case {
        name: &'static str,
        rows: usize,
        distinct_values: usize,
        value_length: usize,
        distribution: Distribution,
        nullable: bool,
    }

    impl std::fmt::Debug for Case {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name)
        }
    }

    impl Case {
        fn value_index(self, index: usize) -> usize {
            match self.distribution {
                Distribution::Uniform | Distribution::VariedPrefix => index % self.distinct_values,
                Distribution::Clustered => index * self.distinct_values / self.rows,
            }
        }

        fn value(self, index: usize) -> String {
            let value_index = self.value_index(index);
            let mut value = match self.distribution {
                Distribution::VariedPrefix => format!(
                    "{:016x}-value-{value_index:08x}",
                    value_index.wrapping_mul(0x9e3779b9)
                ),
                _ if self.value_length <= 12 => format!("{value_index:08x}"),
                _ => format!("common-prefix-value-{value_index:08x}"),
            };
            value.extend(std::iter::repeat_n(
                'x',
                self.value_length.saturating_sub(value.len()),
            ));
            value
        }

        fn make_array(self) -> ArrayRef {
            let values = (0..self.rows)
                .map(|index| self.value(index))
                .collect::<Vec<_>>();
            let nullability = if self.nullable {
                Nullability::Nullable
            } else {
                Nullability::NonNullable
            };

            VarBinViewArray::from_iter(
                values.iter().enumerate().map(|(index, value)| {
                    (!self.nullable || index % 10 != 0).then_some(value.as_str())
                }),
                DType::Utf8(nullability),
            )
            .into_array()
        }
    }

    const fn case(
        name: &'static str,
        rows: usize,
        distinct_values: usize,
        value_length: usize,
        distribution: Distribution,
    ) -> Case {
        Case {
            name,
            rows,
            distinct_values,
            value_length,
            distribution,
            nullable: false,
        }
    }

    const CASES: [Case; 9] = [
        case("Inline4096", 65_536, 4096, 8, Distribution::Uniform),
        case("Outlined16", 65_536, 16, 28, Distribution::Uniform),
        case("Outlined4096", 65_536, 4096, 28, Distribution::Uniform),
        case("Outlined8192", 65_536, 8192, 28, Distribution::Uniform),
        Case {
            nullable: true,
            ..case(
                "NullableOutlined4096",
                65_536,
                4096,
                28,
                Distribution::Uniform,
            )
        },
        case("Clustered4096", 65_536, 4096, 28, Distribution::Clustered),
        case("Long256", 65_536, 4096, 256, Distribution::Uniform),
        case(
            "UniqueCommonPrefix",
            65_536,
            65_536,
            28,
            Distribution::Uniform,
        ),
        case(
            "UniqueVariedPrefix",
            65_536,
            65_536,
            31,
            Distribution::VariedPrefix,
        ),
    ];

    #[divan::bench(args = CASES)]
    fn compress(bencher: Bencher, case: Case) {
        let array = case.make_array();
        let compressor = BtrBlocksCompressor::default();
        bencher
            .with_inputs(|| (&array, SESSION.create_execution_ctx()))
            .input_counter(|(array, _)| ItemsCount::new(array.len()))
            .bench_refs(|(array, ctx)| compressor.compress(array, ctx));
    }

    #[divan::bench(args = CASES)]
    fn decompress(bencher: Bencher, case: Case) {
        let compressor = BtrBlocksCompressor::default();
        let mut ctx = SESSION.create_execution_ctx();
        let compressed = compressor
            .compress(&case.make_array(), &mut ctx)
            .vortex_expect("benchmark input must compress");
        bencher
            .with_inputs(|| (&compressed, SESSION.create_execution_ctx()))
            .input_counter(|(array, _)| ItemsCount::new(array.len()))
            .bench_refs(|(array, ctx)| array.clone().execute::<VarBinViewArray>(ctx));
    }
}

fn main() {
    divan::main();
}
