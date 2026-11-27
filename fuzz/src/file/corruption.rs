// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::Cell;
use std::fmt;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use flatbuffers::root;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_file::EOF_SIZE;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::POSTSCRIPT_LEN_BYTES;
use vortex_file::VERSION_BYTES;
use vortex_file::flatbuffers::footer as fb;

use crate::FUZZ_FILE_ARRAY_MAX_LEN;
use crate::RUNTIME;
use crate::SESSION;
use crate::file::WriteSettings;
use crate::file::has_duplicate_field_names;
use crate::file::has_nullable_struct;
use crate::file::report_on_panic;

const MAX_CORRUPTIONS: usize = 8;
const CORRUPTION_SETS: usize = 1024;
const SET_ENTROPY_BYTES: usize = 64;

#[derive(Clone, Copy)]
pub enum SegmentField {
    /// Although valid Offset is u64, generated files are usually small, so most
    /// such offsets are rejected in file size check. Same for Length
    Offset(u16),
    Length(u16),
    /// Valid values are 0-16, we generate 17 as well
    AlignmentExponent(u8),
}

#[derive(Clone, Copy)]
pub enum Region {
    DataSegment(u8),
    Dtype,
    Layout,
    Statistics,
    Footer,
}

impl fmt::Debug for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataSegment(index) => write!(f, "data_segment[{index}]"),
            Self::Dtype => f.write_str("dtype"),
            Self::Layout => f.write_str("layout"),
            Self::Statistics => f.write_str("statistics"),
            Self::Footer => f.write_str("footer"),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Corruption {
    FooterSegmentSpec {
        index: u8,
        field: SegmentField,
    },
    Bytes {
        region: Region,
        offset: u32,
        byte: u8,
    },
}

impl fmt::Debug for Corruption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FooterSegmentSpec { index, field } => {
                let (name, value) = match field {
                    SegmentField::Offset(v) => ("offset", u32::from(*v)),
                    SegmentField::Length(v) => ("length", u32::from(*v)),
                    SegmentField::AlignmentExponent(v) => ("alignment_exponent", u32::from(*v)),
                };
                write!(f, "footer.segments[{index}].{name} = {value}")
            }
            Self::Bytes { region, byte, .. } => write!(f, "{region:?} ^= {byte:#04x}"),
        }
    }
}

impl<'a> Arbitrary<'a> for Corruption {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(match u.int_in_range(0u8..=7)? {
            kind @ 0..=2 => Self::FooterSegmentSpec {
                index: u.arbitrary()?,
                field: match kind {
                    0 => SegmentField::Offset(u.arbitrary()?),
                    1 => SegmentField::Length(u.arbitrary()?),
                    _ => SegmentField::AlignmentExponent(u.int_in_range(0..=17)?),
                },
            },
            kind => Self::Bytes {
                region: match kind {
                    3 => Region::DataSegment(u.arbitrary()?),
                    4 => Region::Dtype,
                    5 => Region::Layout,
                    6 => Region::Statistics,
                    _ => Region::Footer,
                },
                offset: u.arbitrary()?,
                byte: u.arbitrary()?,
            },
        })
    }
}

pub struct FuzzFileCorruption {
    pub array: ArrayRef,
    pub seed: u32,
    pub write_settings: WriteSettings,
}

impl fmt::Debug for FuzzFileCorruption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FuzzFileCorruption")
            .field("array", &self.array)
            .field("seed", &self.seed)
            .field("write_settings", &self.write_settings)
            .finish()
    }
}

impl<'a> Arbitrary<'a> for FuzzFileCorruption {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let config = ArbitraryArrayConfig {
            dtype: None,
            len: 0..=FUZZ_FILE_ARRAY_MAX_LEN,
        };
        let array = ArbitraryArray::arbitrary_with_config(u, &config)?.0;
        Ok(Self {
            array,
            seed: u.arbitrary()?,
            write_settings: WriteSettings::arbitrary(u)?,
        })
    }
}

thread_local! {
    static CURRENT_SET: Cell<[Option<(usize, Corruption)>; MAX_CORRUPTIONS]> =
        const { Cell::new([None; MAX_CORRUPTIONS]) };
}

pub fn run_file_corruption(fuzz: FuzzFileCorruption) -> VortexResult<bool> {
    let FuzzFileCorruption {
        array,
        seed,
        write_settings,
    } = fuzz;
    if has_nullable_struct(array.dtype()) || has_duplicate_field_names(array.dtype()) {
        return Ok(false);
    }

    let mut file = Vec::new();
    write_settings
        .write_options()
        .blocking(&*RUNTIME)
        .write(&mut file, array.to_array_iterator())?;

    let targets = Targets::new(&file).vortex_expect("written file has a footer");

    report_on_panic(|| {
        for (position, corruption) in CURRENT_SET.get().into_iter().flatten() {
            eprintln!("corruption {position} {corruption:?}");
        }
    });

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(u64::from(seed));
    let mut ctx = SESSION.create_execution_ctx();
    for _ in 0..CORRUPTION_SETS {
        let mut entropy = [0u8; SET_ENTROPY_BYTES];
        rng.fill(&mut entropy);
        let mut u = Unstructured::new(&entropy);
        let mut corrupted = file.clone();
        let mut applied = [None; MAX_CORRUPTIONS];
        let len = u.int_in_range(1..=MAX_CORRUPTIONS).unwrap_or(1);
        for slot in applied.iter_mut().take(len) {
            let corruption = u.arbitrary().ok();
            *slot = corruption.and_then(|corruption| targets.apply(&mut corrupted, corruption));
        }
        CURRENT_SET.set(applied);
        read_corrupted(&mut ctx, corrupted).ok();
    }
    CURRENT_SET.set([None; MAX_CORRUPTIONS]);
    Ok(true)
}

fn read_corrupted(ctx: &mut ExecutionCtx, file: Vec<u8>) -> VortexResult<()> {
    let iter = SESSION
        .open_options()
        .open_buffer(file)?
        .scan()?
        .into_array_iter(&*RUNTIME)?;
    for batch in iter {
        let _canonical = batch?.execute::<RecursiveCanonical>(ctx);
    }
    Ok(())
}

fn postscript(file: &[u8]) -> Option<fb::Postscript<'_>> {
    let eof_start = file.len() - EOF_SIZE;
    let start = eof_start + VERSION_BYTES;
    let end = start + POSTSCRIPT_LEN_BYTES;
    let len_bytes: [u8; POSTSCRIPT_LEN_BYTES] = file.get(start..end)?.try_into().ok()?;
    let postscript_len = usize::from(u16::from_le_bytes(len_bytes));
    let start = eof_start.checked_sub(postscript_len)?;
    root::<fb::Postscript>(file.get(start..eof_start)?).ok()
}

fn span(file: &[u8], offset: u64, length: u32) -> Option<(usize, usize)> {
    let start = usize::try_from(offset).ok()?;
    let length = usize::try_from(length).ok()?;
    let end = start.checked_add(length)?;
    (end <= file.len()).then_some((start, length))
}

struct Targets {
    segment_specs: (usize, usize),
    dtype: Option<(usize, usize)>,
    layout: Option<(usize, usize)>,
    statistics: Option<(usize, usize)>,
    footer: (usize, usize),
}

impl Targets {
    fn new(file: &[u8]) -> Option<Self> {
        let postscript = postscript(file)?;
        let segment = |segment: Option<fb::PostscriptSegment<'_>>| {
            segment.and_then(|segment| span(file, segment.offset(), segment.length()))
        };
        let footer @ (start, length) = segment(postscript.footer())?;
        let specs = root::<fb::Footer>(&file[start..start + length])
            .ok()?
            .segment_specs()?;
        let position = (specs.bytes().as_ptr() as usize).checked_sub(file.as_ptr() as usize)?;
        Some(Self {
            segment_specs: (position, specs.len()),
            dtype: segment(postscript.dtype()),
            layout: segment(postscript.layout()),
            statistics: segment(postscript.statistics()),
            footer,
        })
    }

    fn segment_spec(&self, file: &[u8], index: u8) -> Option<(u8, usize, fb::SegmentSpec)> {
        let (start, len) = self.segment_specs;
        let index = usize::from(index).checked_rem(len)?;
        let position = start + index * size_of::<fb::SegmentSpec>();
        let bytes = file.get(position..position + size_of::<fb::SegmentSpec>())?;
        let spec = fb::SegmentSpec(bytes.try_into().ok()?);
        Some((u8::try_from(index).ok()?, position, spec))
    }

    fn apply(&self, file: &mut [u8], corruption: Corruption) -> Option<(usize, Corruption)> {
        match corruption {
            Corruption::FooterSegmentSpec { index, field } => {
                let (index, position, mut spec) = self.segment_spec(file, index)?;
                match field {
                    SegmentField::Offset(value) => spec.set_offset(u64::from(value)),
                    SegmentField::Length(value) => spec.set_length(u32::from(value)),
                    SegmentField::AlignmentExponent(value) => spec.set_alignment_exponent(value),
                }
                file.get_mut(position..position + spec.0.len())?
                    .copy_from_slice(&spec.0);
                Some((position, Corruption::FooterSegmentSpec { index, field }))
            }
            Corruption::Bytes { byte: 0, .. } => None,
            Corruption::Bytes {
                region,
                offset,
                byte,
            } => {
                let (region, (start, len)) = match region {
                    Region::DataSegment(index) => {
                        let (index, _, spec) = self.segment_spec(file, index)?;
                        let span = span(file, spec.offset(), spec.length())?;
                        (Region::DataSegment(index), span)
                    }
                    Region::Dtype => (region, self.dtype?),
                    Region::Layout => (region, self.layout?),
                    Region::Statistics => (region, self.statistics?),
                    Region::Footer => (region, self.footer),
                };
                let position = start + usize::try_from(offset).ok()?.checked_rem(len)?;
                *file.get_mut(position)? ^= byte;
                let corruption = Corruption::Bytes {
                    region,
                    offset,
                    byte,
                };
                Some((position, corruption))
            }
        }
    }
}
