// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;

use futures::TryStreamExt;
use serde::Serialize;
use vortex::array::ArrayRef;
use vortex::array::LEGACY_SESSION;
use vortex::array::ToCanonical;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::listview::ListViewArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::VortexFile;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::scalar::Scalar;
use vortex::session::VortexSession;

const SOURCE_URI_COL: &str = "source_uri";
const CONTAINER_COL: &str = "container";
const CODEC_COL: &str = "codec";
const TRACK_LANGUAGE_COL: &str = "track_language";
const WIDTH_COL: &str = "width";
const HEIGHT_COL: &str = "height";
const FPS_NUM_COL: &str = "fps_num";
const FPS_DEN_COL: &str = "fps_den";
const TIMESCALE_COL: &str = "timescale";
const DURATION_TS_COL: &str = "duration_ts";
const DURATION_MS_COL: &str = "duration_ms";
const FILE_SIZE_BYTES_COL: &str = "file_size_bytes";
const PRIMARY_TRACK_ID_COL: &str = "primary_track_id";
const NAL_LENGTH_SIZE_COL: &str = "nal_length_size";
const AVCC_SEQ_PARAM_SET_COL: &str = "avcc_seq_param_set";
const AVCC_PIC_PARAM_SET_COL: &str = "avcc_pic_param_set";
const GOPS_COL: &str = "gops";
const FRAMES_BY_VIDEO_COL: &str = "frames_by_video";
const SAMPLES_BY_DECODE_COL: &str = "samples_by_decode";
const TRACKS_COL: &str = "tracks";
const TRACK_ID_COL: &str = "track_id";

const GOP_POS_COL: &str = "gop_pos";
const START_PTS_COL: &str = "start_pts";
const END_PTS_COL: &str = "end_pts";
const START_DTS_COL: &str = "start_dts";
const END_DTS_COL: &str = "end_dts";
const START_BYTE_OFFSET_COL: &str = "start_byte_offset";
const BYTE_LENGTH_COL: &str = "byte_length";
const FRAME_COUNT_COL: &str = "frame_count";
const KEYFRAME_DECODE_POS_COL: &str = "keyframe_decode_pos";
const DEPENDENCY_TREE_HEIGHT_COL: &str = "dependency_tree_height";
const FRAMES_COL: &str = "frames";

const SAMPLE_ID_COL: &str = "sample_id";
const GLOBAL_DECODE_POS_COL: &str = "global_decode_pos";
const VIDEO_FRAME_POS_COL: &str = "video_frame_pos";
const GOP_FRAME_POS_COL: &str = "gop_frame_pos";
const GOP_DECODE_POS_COL: &str = "gop_decode_pos";
const PTS_COL: &str = "pts";
const DTS_COL: &str = "dts";
const DURATION_COL: &str = "duration";
const DISPLAY_POS_COL: &str = "display_pos";
const DECODE_POS_COL: &str = "decode_pos";
const FRAME_TYPE_COL: &str = "frame_type";
const IS_SYNC_COL: &str = "is_sync";
const FRAME_NUM_COL: &str = "frame_num";
const IS_REFERENCE_COL: &str = "is_reference";
const SAMPLE_BYTE_OFFSET_COL: &str = "sample_byte_offset";
const SAMPLE_BYTE_LENGTH_COL: &str = "sample_byte_length";
const REF_L0_DECODE_POSITIONS_COL: &str = "ref_l0_decode_positions";
const REF_L1_DECODE_POSITIONS_COL: &str = "ref_l1_decode_positions";
const REF_L0_GLOBAL_DECODE_POSITIONS_COL: &str = "ref_l0_global_decode_positions";
const REF_L1_GLOBAL_DECODE_POSITIONS_COL: &str = "ref_l1_global_decode_positions";
const REF_PREV_DECODE_POS_COL: &str = "ref_prev_decode_pos";
const REF_NEXT_DECODE_POS_COL: &str = "ref_next_decode_pos";
const DEPENDENCY_DEPTH_COL: &str = "dependency_depth";
const CLOSURE_LOCAL_DECODE_MASK_LE_COL: &str = "closure_local_decode_mask_le";
const CLOSURE_EXTERNAL_DECODE_POSITIONS_COL: &str = "closure_external_decode_positions";

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VideoPlanningFrameInfoJson {
    pub video_frame_pos: u32,
    pub global_decode_pos: u32,
    pub gop_pos: u32,
    pub gop_decode_pos: u16,
    pub sample_byte_offset: u64,
    pub sample_byte_length: u32,
    pub closure_local_decode_mask_le: Vec<u8>,
    pub closure_external_decode_positions: Vec<u32>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DecodeSampleInfoJson {
    pub sample_id: u32,
    pub global_decode_pos: u32,
    pub video_frame_pos: u32,
    pub gop_pos: u32,
    pub gop_frame_pos: u16,
    pub gop_decode_pos: u16,
    pub pts: i64,
    pub dts: i64,
    pub duration: u32,
    pub sample_byte_offset: u64,
    pub sample_byte_length: u32,
    pub is_sync: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VideoFrameInfoJson {
    pub sample_id: u32,
    pub global_decode_pos: u32,
    pub video_frame_pos: u32,
    pub gop_pos: u32,
    pub gop_frame_pos: u16,
    pub pts: i64,
    pub dts: i64,
    pub duration: u32,
    pub display_pos: u16,
    pub decode_pos: u16,
    pub frame_type: String,
    pub is_sync: bool,
    pub frame_num: Option<u32>,
    pub is_reference: bool,
    pub sample_byte_offset: u64,
    pub sample_byte_length: u32,
    pub ref_l0_decode_positions: Vec<u16>,
    pub ref_l1_decode_positions: Vec<u16>,
    pub ref_l0_global_decode_positions: Vec<u32>,
    pub ref_l1_global_decode_positions: Vec<u32>,
    pub ref_prev_decode_pos: Option<u32>,
    pub ref_next_decode_pos: Option<u32>,
    pub dependency_depth: Option<u8>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VideoGopInfoJson {
    pub gop_pos: u32,
    pub start_pts: i64,
    pub end_pts: i64,
    pub start_dts: i64,
    pub end_dts: i64,
    pub start_byte_offset: u64,
    pub byte_length: u64,
    pub frame_count: u16,
    pub keyframe_decode_pos: u16,
    pub dependency_tree_height: u8,
    pub start_global_decode_pos: u32,
    pub end_global_decode_pos: u32,
    pub frames: Vec<VideoFrameInfoJson>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VideoTrackInfoJson {
    pub track_id: u32,
    pub track_language: String,
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub timescale: u32,
    pub duration_ts: i64,
    pub duration_ms: u64,
    pub nal_length_size: u8,
    pub frame_count: usize,
    pub gops: Vec<VideoGopInfoJson>,
    pub frames: Vec<VideoFrameInfoJson>,
    pub planning_frames: Vec<VideoPlanningFrameInfoJson>,
    pub samples_by_decode: Vec<DecodeSampleInfoJson>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VideoIndexInfoJson {
    pub source_uri: String,
    pub container: String,
    pub codec: String,
    pub primary_track_id: u32,
    pub track_language: String,
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub timescale: u32,
    pub duration_ts: i64,
    pub duration_ms: u64,
    pub file_size_bytes: u64,
    pub nal_length_size: u8,
    pub frame_count: usize,
    pub gops: Vec<VideoGopInfoJson>,
    pub frames: Vec<VideoFrameInfoJson>,
    pub planning_frames: Vec<VideoPlanningFrameInfoJson>,
    pub samples_by_decode: Vec<DecodeSampleInfoJson>,
    pub tracks: Vec<VideoTrackInfoJson>,
}

impl VideoIndexInfoJson {
    fn try_from_struct_row(struct_array: &StructArray, row_idx: usize) -> VortexResult<Self> {
        if row_idx >= struct_array.len() {
            return Err(vortex_err!(
                "video index row {} out of bounds for len {}",
                row_idx,
                struct_array.len()
            ));
        }

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        if !struct_array.is_valid(row_idx, &mut ctx)? {
            return Err(vortex_err!("video index row {} is null", row_idx));
        }

        let source_uri: String = scalar_at(struct_array, SOURCE_URI_COL, row_idx)?.try_into()?;
        let container: String = scalar_at(struct_array, CONTAINER_COL, row_idx)?.try_into()?;
        let codec: String = scalar_at(struct_array, CODEC_COL, row_idx)?.try_into()?;
        let file_size_bytes: Option<u64> =
            (&scalar_at(struct_array, FILE_SIZE_BYTES_COL, row_idx)?).try_into()?;
        let primary_track_id =
            scalar_optional::<u32>(struct_array, PRIMARY_TRACK_ID_COL, row_idx)?.unwrap_or(1);
        let primary_track = parse_track_row(struct_array, row_idx, Some(primary_track_id))?;
        let mut tracks = parse_tracks(struct_array, row_idx)?;
        if tracks.is_empty() {
            tracks.push(primary_track.clone());
        } else if tracks
            .iter()
            .all(|track| track.track_id != primary_track_id)
        {
            tracks.push(primary_track.clone());
        }
        tracks.sort_by_key(|track| track.track_id);

        Ok(Self {
            source_uri,
            container,
            codec,
            primary_track_id,
            track_language: primary_track.track_language.clone(),
            width: primary_track.width,
            height: primary_track.height,
            fps_num: primary_track.fps_num,
            fps_den: primary_track.fps_den,
            timescale: primary_track.timescale,
            duration_ts: primary_track.duration_ts,
            duration_ms: primary_track.duration_ms,
            file_size_bytes: file_size_bytes
                .ok_or_else(|| vortex_err!("missing {}", FILE_SIZE_BYTES_COL))?,
            nal_length_size: primary_track.nal_length_size,
            frame_count: primary_track.frames.len(),
            gops: primary_track.gops.clone(),
            frames: primary_track.frames.clone(),
            planning_frames: primary_track.planning_frames.clone(),
            samples_by_decode: primary_track.samples_by_decode.clone(),
            tracks,
        })
    }
}

pub(crate) async fn try_read_video_index_info(
    vxf: &VortexFile,
    session: &VortexSession,
) -> VortexResult<Option<VideoIndexInfoJson>> {
    if vxf.row_count() != 1 {
        return Ok(None);
    }

    let Some(root_array) = try_read_single_root_row(vxf, session).await? else {
        return Ok(None);
    };
    if !root_array.dtype().is_struct() {
        return Ok(None);
    }

    let struct_array = root_array.to_struct();
    Ok(VideoIndexInfoJson::try_from_struct_row(&struct_array, 0).ok())
}

async fn try_read_single_root_row(
    vxf: &VortexFile,
    session: &VortexSession,
) -> VortexResult<Option<ArrayRef>> {
    let reader = vxf
        .footer()
        .layout()
        .new_reader("root".into(), vxf.segment_source(), session)?;
    let stream = ScanBuilder::new(session.clone(), reader)
        .with_limit(1)
        .into_array_stream()?;
    let arrays: Vec<ArrayRef> = stream.try_collect().await?;
    let mut non_empty = arrays.into_iter().filter(|array| array.len() > 0);
    let Some(first) = non_empty.next() else {
        return Ok(None);
    };
    if non_empty.next().is_some() || first.len() != 1 {
        return Ok(None);
    }
    Ok(Some(first))
}

fn parse_track_row(
    struct_array: &StructArray,
    row_idx: usize,
    fallback_track_id: Option<u32>,
) -> VortexResult<VideoTrackInfoJson> {
    let track_id = scalar_optional::<u32>(struct_array, TRACK_ID_COL, row_idx)?
        .or(fallback_track_id)
        .ok_or_else(|| vortex_err!("missing {}", TRACK_ID_COL))?;
    let track_language: String =
        scalar_at(struct_array, TRACK_LANGUAGE_COL, row_idx)?.try_into()?;
    let width: Option<u32> = (&scalar_at(struct_array, WIDTH_COL, row_idx)?).try_into()?;
    let height: Option<u32> = (&scalar_at(struct_array, HEIGHT_COL, row_idx)?).try_into()?;
    let fps_num: Option<u32> = (&scalar_at(struct_array, FPS_NUM_COL, row_idx)?).try_into()?;
    let fps_den: Option<u32> = (&scalar_at(struct_array, FPS_DEN_COL, row_idx)?).try_into()?;
    let timescale: Option<u32> = (&scalar_at(struct_array, TIMESCALE_COL, row_idx)?).try_into()?;
    let duration_ts: Option<i64> =
        (&scalar_at(struct_array, DURATION_TS_COL, row_idx)?).try_into()?;
    let duration_ms: Option<u64> =
        (&scalar_at(struct_array, DURATION_MS_COL, row_idx)?).try_into()?;
    let nal_length_size: Option<u8> =
        (&scalar_at(struct_array, NAL_LENGTH_SIZE_COL, row_idx)?).try_into()?;
    let _avcc_seq_param_set = binary_optional(struct_array, AVCC_SEQ_PARAM_SET_COL, row_idx)?;
    let _avcc_pic_param_set = binary_optional(struct_array, AVCC_PIC_PARAM_SET_COL, row_idx)?;

    let planning_frames = parse_planning_frames(struct_array, row_idx)?;
    let samples_by_decode = parse_samples_by_decode(struct_array, row_idx)?;
    let mut gops = parse_gops(struct_array, row_idx)?;
    annotate_gop_decode_bounds(&mut gops, &samples_by_decode)?;

    let mut frames = gops
        .iter()
        .flat_map(|gop| gop.frames.iter().cloned())
        .collect::<Vec<_>>();
    frames.sort_by_key(|frame| frame.video_frame_pos);

    Ok(VideoTrackInfoJson {
        track_id,
        track_language,
        width: width.ok_or_else(|| vortex_err!("missing {}", WIDTH_COL))?,
        height: height.ok_or_else(|| vortex_err!("missing {}", HEIGHT_COL))?,
        fps_num: fps_num.ok_or_else(|| vortex_err!("missing {}", FPS_NUM_COL))?,
        fps_den: fps_den.ok_or_else(|| vortex_err!("missing {}", FPS_DEN_COL))?,
        timescale: timescale.ok_or_else(|| vortex_err!("missing {}", TIMESCALE_COL))?,
        duration_ts: duration_ts.ok_or_else(|| vortex_err!("missing {}", DURATION_TS_COL))?,
        duration_ms: duration_ms.ok_or_else(|| vortex_err!("missing {}", DURATION_MS_COL))?,
        nal_length_size: nal_length_size
            .ok_or_else(|| vortex_err!("missing {}", NAL_LENGTH_SIZE_COL))?,
        frame_count: frames.len(),
        gops,
        frames,
        planning_frames,
        samples_by_decode,
    })
}

fn parse_tracks(
    struct_array: &StructArray,
    row_idx: usize,
) -> VortexResult<Vec<VideoTrackInfoJson>> {
    let Some(tracks_field) = struct_array.unmasked_field_by_name_opt(TRACKS_COL) else {
        return Ok(Vec::new());
    };
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    if !tracks_field.is_valid(row_idx, &mut ctx)? {
        return Ok(Vec::new());
    }

    let tracks = tracks_field
        .to_listview()
        .list_elements_at(row_idx)?
        .to_struct();
    let mut result = Vec::with_capacity(tracks.len());
    for idx in 0..tracks.len() {
        result.push(parse_track_row(&tracks, idx, None)?);
    }
    Ok(result)
}

fn parse_gops(struct_array: &StructArray, row_idx: usize) -> VortexResult<Vec<VideoGopInfoJson>> {
    let gops = struct_array
        .unmasked_field_by_name(GOPS_COL)?
        .to_listview()
        .list_elements_at(row_idx)?
        .to_struct();

    let mut result = Vec::with_capacity(gops.len());
    for idx in 0..gops.len() {
        let frames = parse_gop_frames(&gops, idx)?;
        result.push(VideoGopInfoJson {
            gop_pos: scalar_required(&gops, GOP_POS_COL, idx)?,
            start_pts: scalar_required(&gops, START_PTS_COL, idx)?,
            end_pts: scalar_required(&gops, END_PTS_COL, idx)?,
            start_dts: scalar_required(&gops, START_DTS_COL, idx)?,
            end_dts: scalar_required(&gops, END_DTS_COL, idx)?,
            start_byte_offset: scalar_required(&gops, START_BYTE_OFFSET_COL, idx)?,
            byte_length: scalar_required(&gops, BYTE_LENGTH_COL, idx)?,
            frame_count: scalar_required(&gops, FRAME_COUNT_COL, idx)?,
            keyframe_decode_pos: scalar_required(&gops, KEYFRAME_DECODE_POS_COL, idx)?,
            dependency_tree_height: scalar_required(&gops, DEPENDENCY_TREE_HEIGHT_COL, idx)?,
            start_global_decode_pos: 0,
            end_global_decode_pos: 0,
            frames,
        });
    }

    Ok(result)
}

fn parse_gop_frames(gops: &StructArray, gop_idx: usize) -> VortexResult<Vec<VideoFrameInfoJson>> {
    let frames = gops
        .unmasked_field_by_name(FRAMES_COL)?
        .to_listview()
        .list_elements_at(gop_idx)?
        .to_struct();

    let mut result = Vec::with_capacity(frames.len());
    for idx in 0..frames.len() {
        let frame_num: Option<u32> = scalar_optional(&frames, FRAME_NUM_COL, idx)?;
        let ref_prev_decode_pos: Option<u32> =
            scalar_optional(&frames, REF_PREV_DECODE_POS_COL, idx)?;
        let ref_next_decode_pos: Option<u32> =
            scalar_optional(&frames, REF_NEXT_DECODE_POS_COL, idx)?;
        let dependency_depth: Option<u8> = scalar_optional(&frames, DEPENDENCY_DEPTH_COL, idx)?;

        result.push(VideoFrameInfoJson {
            sample_id: scalar_required(&frames, SAMPLE_ID_COL, idx)?,
            global_decode_pos: scalar_required(&frames, GLOBAL_DECODE_POS_COL, idx)?,
            video_frame_pos: scalar_required(&frames, VIDEO_FRAME_POS_COL, idx)?,
            gop_pos: scalar_required(&frames, GOP_POS_COL, idx)?,
            gop_frame_pos: scalar_required(&frames, GOP_FRAME_POS_COL, idx)?,
            pts: scalar_required(&frames, PTS_COL, idx)?,
            dts: scalar_required(&frames, DTS_COL, idx)?,
            duration: scalar_required(&frames, DURATION_COL, idx)?,
            display_pos: scalar_required(&frames, DISPLAY_POS_COL, idx)?,
            decode_pos: scalar_required(&frames, DECODE_POS_COL, idx)?,
            frame_type: scalar_required::<String>(&frames, FRAME_TYPE_COL, idx)?,
            is_sync: scalar_required(&frames, IS_SYNC_COL, idx)?,
            frame_num,
            is_reference: scalar_required(&frames, IS_REFERENCE_COL, idx)?,
            sample_byte_offset: scalar_required(&frames, SAMPLE_BYTE_OFFSET_COL, idx)?,
            sample_byte_length: scalar_required(&frames, SAMPLE_BYTE_LENGTH_COL, idx)?,
            ref_l0_decode_positions: scalar_required(&frames, REF_L0_DECODE_POSITIONS_COL, idx)?,
            ref_l1_decode_positions: scalar_required(&frames, REF_L1_DECODE_POSITIONS_COL, idx)?,
            ref_l0_global_decode_positions: scalar_required(
                &frames,
                REF_L0_GLOBAL_DECODE_POSITIONS_COL,
                idx,
            )?,
            ref_l1_global_decode_positions: scalar_required(
                &frames,
                REF_L1_GLOBAL_DECODE_POSITIONS_COL,
                idx,
            )?,
            ref_prev_decode_pos,
            ref_next_decode_pos,
            dependency_depth,
        });
    }

    Ok(result)
}

fn parse_planning_frames(
    struct_array: &StructArray,
    row_idx: usize,
) -> VortexResult<Vec<VideoPlanningFrameInfoJson>> {
    let frames = struct_array
        .unmasked_field_by_name(FRAMES_BY_VIDEO_COL)?
        .to_listview()
        .list_elements_at(row_idx)?
        .to_struct();

    let mut result = Vec::with_capacity(frames.len());
    for idx in 0..frames.len() {
        result.push(VideoPlanningFrameInfoJson {
            video_frame_pos: scalar_required(&frames, VIDEO_FRAME_POS_COL, idx)?,
            global_decode_pos: scalar_required(&frames, GLOBAL_DECODE_POS_COL, idx)?,
            gop_pos: scalar_required(&frames, GOP_POS_COL, idx)?,
            gop_decode_pos: scalar_required(&frames, GOP_DECODE_POS_COL, idx)?,
            sample_byte_offset: scalar_required(&frames, SAMPLE_BYTE_OFFSET_COL, idx)?,
            sample_byte_length: scalar_required(&frames, SAMPLE_BYTE_LENGTH_COL, idx)?,
            closure_local_decode_mask_le: binary_required(
                &frames,
                CLOSURE_LOCAL_DECODE_MASK_LE_COL,
                idx,
            )?,
            closure_external_decode_positions: scalar_required(
                &frames,
                CLOSURE_EXTERNAL_DECODE_POSITIONS_COL,
                idx,
            )?,
        });
    }

    Ok(result)
}

fn parse_samples_by_decode(
    struct_array: &StructArray,
    row_idx: usize,
) -> VortexResult<Vec<DecodeSampleInfoJson>> {
    let samples = struct_array
        .unmasked_field_by_name(SAMPLES_BY_DECODE_COL)?
        .to_listview()
        .list_elements_at(row_idx)?
        .to_struct();

    let mut result = Vec::with_capacity(samples.len());
    for idx in 0..samples.len() {
        result.push(DecodeSampleInfoJson {
            sample_id: scalar_required(&samples, SAMPLE_ID_COL, idx)?,
            global_decode_pos: scalar_required(&samples, GLOBAL_DECODE_POS_COL, idx)?,
            video_frame_pos: scalar_required(&samples, VIDEO_FRAME_POS_COL, idx)?,
            gop_pos: scalar_required(&samples, GOP_POS_COL, idx)?,
            gop_frame_pos: scalar_required(&samples, GOP_FRAME_POS_COL, idx)?,
            gop_decode_pos: scalar_required(&samples, GOP_DECODE_POS_COL, idx)?,
            pts: scalar_required(&samples, PTS_COL, idx)?,
            dts: scalar_required(&samples, DTS_COL, idx)?,
            duration: scalar_required(&samples, DURATION_COL, idx)?,
            sample_byte_offset: scalar_required(&samples, SAMPLE_BYTE_OFFSET_COL, idx)?,
            sample_byte_length: scalar_required(&samples, SAMPLE_BYTE_LENGTH_COL, idx)?,
            is_sync: scalar_required(&samples, IS_SYNC_COL, idx)?,
        });
    }

    for (expected, sample) in result.iter().enumerate() {
        if sample.global_decode_pos as usize != expected {
            return Err(vortex_err!(
                "samples_by_decode must be dense and sorted by global_decode_pos"
            ));
        }
    }

    Ok(result)
}

fn annotate_gop_decode_bounds(
    gops: &mut [VideoGopInfoJson],
    samples: &[DecodeSampleInfoJson],
) -> VortexResult<()> {
    let mut gop_ranges = BTreeMap::new();
    for sample in samples {
        let entry = gop_ranges
            .entry(sample.gop_pos)
            .or_insert((sample.global_decode_pos, sample.global_decode_pos));
        entry.0 = entry.0.min(sample.global_decode_pos);
        entry.1 = entry.1.max(sample.global_decode_pos);
    }

    for gop in gops {
        let (start, end) = gop_ranges
            .get(&gop.gop_pos)
            .copied()
            .ok_or_else(|| vortex_err!("missing decode samples for gop {}", gop.gop_pos))?;
        gop.start_global_decode_pos = start;
        gop.end_global_decode_pos = end;
    }

    Ok(())
}

fn scalar_required<T>(struct_array: &StructArray, col_name: &str, idx: usize) -> VortexResult<T>
where
    for<'a> T: TryFrom<&'a Scalar>,
    for<'a> <T as TryFrom<&'a Scalar>>::Error: std::error::Error + Send + Sync + 'static,
{
    let scalar = struct_array
        .unmasked_field_by_name(col_name)?
        .scalar_at(idx)?;
    (&scalar)
        .try_into()
        .map_err(|error| vortex_err!("failed to decode {}[{}]: {}", col_name, idx, error))
}

fn scalar_optional<T>(
    struct_array: &StructArray,
    col_name: &str,
    idx: usize,
) -> VortexResult<Option<T>>
where
    for<'a> T: TryFrom<&'a Scalar>,
    for<'a> <T as TryFrom<&'a Scalar>>::Error: std::error::Error + Send + Sync + 'static,
{
    let Some(field) = struct_array.unmasked_field_by_name_opt(col_name) else {
        return Ok(None);
    };
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    if !field.is_valid(idx, &mut ctx)? {
        return Ok(None);
    }
    let scalar = field.scalar_at(idx)?;
    (&scalar)
        .try_into()
        .map(Some)
        .map_err(|error| vortex_err!("failed to decode {}[{}]: {}", col_name, idx, error))
}

fn binary_required(
    struct_array: &StructArray,
    col_name: &str,
    idx: usize,
) -> VortexResult<Vec<u8>> {
    let scalar = struct_array
        .unmasked_field_by_name(col_name)?
        .scalar_at(idx)?;
    let bytes: ByteBuffer = (&scalar)
        .try_into()
        .map_err(|error| vortex_err!("failed to decode {}[{}]: {}", col_name, idx, error))?;
    Ok(bytes.as_slice().to_vec())
}

fn binary_optional(
    struct_array: &StructArray,
    col_name: &str,
    idx: usize,
) -> VortexResult<Option<Vec<u8>>> {
    let Some(field) = struct_array.unmasked_field_by_name_opt(col_name) else {
        return Ok(None);
    };
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    if !field.is_valid(idx, &mut ctx)? {
        return Ok(None);
    }
    let scalar = field.scalar_at(idx)?;
    let bytes: ByteBuffer = (&scalar)
        .try_into()
        .map_err(|error| vortex_err!("failed to decode {}[{}]: {}", col_name, idx, error))?;
    Ok(Some(bytes.as_slice().to_vec()))
}

fn scalar_at(struct_array: &StructArray, col_name: &str, idx: usize) -> VortexResult<Scalar> {
    struct_array
        .unmasked_field_by_name(col_name)?
        .scalar_at(idx)
}
