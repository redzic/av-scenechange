mod y4m;

use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ptr::addr_of,
    sync::Arc,
    time::Instant,
};

use av_metrics_decoders::{Decoder, FfmpegDecoder, VapoursynthDecoder};
use ffmpeg::frame;
pub use rav1e::scenechange::SceneChangeDetector;
use rav1e::{
    config::{CpuFeatureLevel, EncoderConfig},
    prelude::{ChromaSamplePosition, Frame, Pixel, Plane, PlaneConfig, PlaneData, Sequence},
};
use vapoursynth::prelude::FrameRef;

/// Options determining how to run scene change detection.
#[derive(Debug, Clone, Copy)]
pub struct DetectionOptions {
    /// The speed of detection algorithm to use.
    /// Slower algorithms are more accurate/better for use in encoders.
    pub analysis_speed: SceneDetectionSpeed,
    /// Enabling this will utilize heuristics to avoid scenecuts
    /// that are too close to each other.
    /// This is generally useful if you want scenecut detection
    /// for use in an encoder.
    /// If you want a raw list of scene changes, you should disable this.
    pub detect_flashes: bool,
    /// The minimum distane between two scene changes.
    pub min_scenecut_distance: Option<usize>,
    /// The maximum distance between two scene changes.
    pub max_scenecut_distance: Option<usize>,
    /// The distance to look ahead in the video
    /// for scene flash detection.
    ///
    /// Not used if `detect_flashes` is `false`.
    pub lookahead_distance: usize,
}

impl Default for DetectionOptions {
    fn default() -> Self {
        DetectionOptions {
            analysis_speed: SceneDetectionSpeed::Standard,
            detect_flashes: true,
            lookahead_distance: 5,
            min_scenecut_distance: None,
            max_scenecut_distance: None,
        }
    }
}

/// Results from a scene change detection pass.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
pub struct DetectionResults {
    /// The 0-indexed frame numbers where scene changes were detected.
    pub scene_changes: Vec<usize>,
    /// The total number of frames read.
    pub frame_count: usize,
}

pub fn new_detector<T: Pixel>(
    // dec: &mut FfmpegDecoder,
    dec: &mut VapoursynthDecoder,
    opts: DetectionOptions,
) -> SceneChangeDetector<T> {
    let video_details = dec.get_video_details();
    let mut config =
        EncoderConfig::with_speed_preset(if opts.analysis_speed == SceneDetectionSpeed::Fast {
            10
        } else {
            8
        });

    config.min_key_frame_interval = opts.min_scenecut_distance.map_or(0, |val| val as u64);
    config.max_key_frame_interval = opts
        .max_scenecut_distance
        .map_or_else(|| u32::MAX.into(), |val| val as u64);
    config.width = video_details.width;
    config.height = video_details.height;
    config.bit_depth = video_details.bit_depth;
    config.time_base = unsafe { std::mem::transmute(video_details.time_base) };
    config.chroma_sampling = unsafe { std::mem::transmute(video_details.chroma_sampling) };
    // This value does not seem to be needed, so just initialize it
    // to this (since video_details no longer contains this)
    config.chroma_sample_position = ChromaSamplePosition::Colocated;
    // force disable temporal RDO to disable intra cost caching
    config.speed_settings.transform.tx_domain_distortion = true;

    let sequence = Arc::new(Sequence::new(&config));
    SceneChangeDetector::new(
        config,
        CpuFeatureLevel::default(),
        if opts.detect_flashes {
            opts.lookahead_distance
        } else {
            1
        },
        sequence,
    )
}

fn align_power_of_two_and_shift(x: usize, n: usize) -> usize {
    (x + (1 << n) - 1) >> n
}

/// Runs through a y4m video clip,
/// detecting where scene changes occur.
/// This is adjustable based on the `opts` parameters.
///
/// This is the preferred, simplified interface
/// for analyzing a whole clip for scene changes.
///
/// # Arguments
///
/// - `progress_callback`: An optional callback that will fire after each frame is analyzed.
///   Arguments passed in will be, in order,
///   the number of frames analyzed, and the number of keyframes detected.
///   This is generally useful for displaying progress, etc.
#[allow(clippy::needless_pass_by_value)]
pub fn detect_scene_changes<T: Pixel + av_metrics_decoders::Pixel>(
    // dec: &mut FfmpegDecoder,
    dec: &mut VapoursynthDecoder,
    opts: DetectionOptions,
    _frame_limit: Option<usize>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> DetectionResults {
    assert!(opts.lookahead_distance >= 1);

    let empty_plane = || Plane::<T> {
        cfg: PlaneConfig {
            alloc_height: 0,
            height: 0,
            stride: 0,
            width: 0,
            xdec: 0,
            xorigin: 0,
            xpad: 0,
            ydec: 0,
            yorigin: 0,
            ypad: 0,
        },
        data: unsafe { PlaneData::new_ref(&[]) },
    };

    let mut detector = new_detector::<T>(dec, opts);
    let video_details = dec.get_video_details();

    const SB_SIZE_LOG2: u32 = 6;
    let (alloc_height, stride) = if opts.analysis_speed == SceneDetectionSpeed::Fast {
        (video_details.height as u32, video_details.width as u32)
    } else {
        (
            (align_power_of_two_and_shift(video_details.height, SB_SIZE_LOG2 as usize)
                << SB_SIZE_LOG2) as u32,
            (align_power_of_two_and_shift(video_details.width, SB_SIZE_LOG2 as usize)
                << SB_SIZE_LOG2) as u32,
        )
    };

    // TODO: handle 422

    let plane_cfg_luma: PlaneConfig = PlaneConfig {
        alloc_height: alloc_height as usize,
        height: video_details.height,
        stride: stride as usize,
        width: video_details.width,
        xdec: 0,
        xorigin: 0,
        xpad: 0,
        ydec: 0,
        yorigin: 0,
        ypad: 0,
    };

    // Frame index, frame allocation
    let mut v = Vec::<(usize, FrameRef)>::new();
    let mut keyframes: Vec<usize> = vec![0];

    let mut frameno: usize = 0;

    let fill_vec = |frame_queue: &[(usize, FrameRef)]| {
        frame_queue
            .iter()
            .map(|(_, v)| unsafe {
                ManuallyDrop::new(Arc::new(Frame::<T> {
                    planes: [
                        {
                            Plane::<T> {
                                cfg: plane_cfg_luma.clone(),
                                data: PlaneData::new_ref(std::slice::from_raw_parts(
                                    v.data_ptr(0).cast(),
                                    stride as usize * video_details.height,
                                )),
                            }
                        },
                        empty_plane(),
                        empty_plane(),
                    ],
                }))
            })
            .collect::<Vec<_>>()
    };

    // TODO: Handle edge case where number of frames is less than lookahead
    // for (_, v) in v.iter_mut().take(opts.lookahead_distance + 1) {
    for i in 0..opts.lookahead_distance + 2 {
        v.push((
            i,
            if let Ok(frame) = dec.receive_frame() {
                frame
            } else {
                // return keyframes;
                return DetectionResults {
                    scene_changes: keyframes,
                    frame_count: frameno,
                };
            },
        ));
    }

    frameno += 1;

    // We always have a keyframe at 0, so we "skip" the first analyze_next_frame call
    if let Some(progress_fn) = progress_callback {
        progress_fn(frameno, keyframes.len());
    }

    // TODO double check that order of this is correct
    let x1 = fill_vec(&v);
    let y1 = x1.iter().map(|x| &**x).collect::<Vec<_>>();

    if detector.analyze_next_frame(
        &*y1,
        frameno as u64,
        *keyframes.iter().last().unwrap() as u64,
    ) {
        keyframes.push(frameno);
    };

    frameno += 1;

    loop {
        let first = v.remove(0);
        let new_last = v.last().unwrap().0 + 1;

        v.push(first);
        let len = v.len();

        // let frame_received = dec.receive_frame_with_alloc::<T>(&mut v[len - 1].1);
        let frame = dec.receive_frame();

        // if frame_received {
        if let Ok(frame) = frame {
            v[len - 1].1 = frame;
            v[len - 1].0 = new_last;
        } else {
            v.pop().unwrap();
            break;
        }

        let x1 = fill_vec(&v);
        let y1 = x1.iter().map(|x| &**x).collect::<Vec<_>>();
        if detector.analyze_next_frame(
            &*y1,
            frameno as u64,
            *keyframes.iter().last().unwrap() as u64,
        ) {
            keyframes.push(frameno);
        };

        frameno += 1;

        if let Some(progress_fn) = progress_callback {
            progress_fn(frameno, keyframes.len());
        }
    }

    while v.len() != 1 {
        frameno += 1;

        let x1 = fill_vec(&v);
        let y1 = x1.iter().map(|x| &**x).collect::<Vec<_>>();

        if detector.analyze_next_frame(
            &*y1,
            frameno as u64,
            *keyframes.iter().last().unwrap() as u64,
        ) {
            keyframes.push(frameno);
        };

        if let Some(progress_fn) = progress_callback {
            progress_fn(frameno, keyframes.len());
        }

        v.remove(0);
    }

    DetectionResults {
        scene_changes: keyframes,
        frame_count: frameno,
    }
}

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Eq)]
pub enum SceneDetectionSpeed {
    /// Fastest scene detection using pixel-wise comparison
    Fast,
    /// Scene detection using motion vectors
    Standard,
}
