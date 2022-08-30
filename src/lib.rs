mod y4m;

use std::{
    mem::{transmute, ManuallyDrop},
    sync::Arc,
};

use av_metrics_decoders::{Decoder2, Frame2};
pub use rav1e::scenechange::SceneChangeDetector;
use rav1e::{
    config::{CpuFeatureLevel, EncoderConfig},
    prelude::{ChromaSamplePosition, Frame, Pixel, Sequence},
};

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

pub fn new_detector<F, D: Decoder2<F>, T: Pixel>(
    // dec: &mut FfmpegDecoder,
    dec: &mut D,
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
pub fn detect_scene_changes<F, D: Decoder2<F>, T: Pixel>(
    dec: &mut D,
    opts: DetectionOptions,
    _frame_limit: Option<usize>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> DetectionResults {
    assert!(opts.lookahead_distance >= 1);

    let mut detector = new_detector::<F, D, T>(dec, opts);
    let video_details = dec.get_video_details();

    const SB_SIZE_LOG2: u32 = 6;
    let (alloc_height, stride) = if opts.analysis_speed == SceneDetectionSpeed::Fast {
        (video_details.height, video_details.width)
    } else {
        (
            (align_power_of_two_and_shift(video_details.height, SB_SIZE_LOG2 as usize)
                << SB_SIZE_LOG2),
            (align_power_of_two_and_shift(video_details.width, SB_SIZE_LOG2 as usize)
                << SB_SIZE_LOG2),
        )
    };

    // Frame index, frame allocation
    let mut v = Vec::<(usize, F)>::new();
    let mut keyframes: Vec<usize> = vec![0];

    let mut frameno: usize = 0;

    let (w, h) = {
        let vd = dec.get_video_details();
        (vd.width, vd.height)
    };

    let strict = opts.analysis_speed == SceneDetectionSpeed::Standard;

    let fill_vec = |frame_queue: &[(usize, F)]| {
        frame_queue
            .iter()
            .map(|(_, v)| {
                ManuallyDrop::new(Arc::new(unsafe {
                    match D::get_frame_ref::<T>(v, h, w, stride, alloc_height, strict) {
                        Frame2::Ref(x) => transmute(x),
                        // hmm is there a memory leak going on in this case
                        Frame2::Owned(x) => transmute(x),
                    }
                }))
            })
            .collect::<Vec<_>>()
    };
    fn map_vec<T: Pixel>(x: &[ManuallyDrop<Arc<Frame<T>>>]) -> Vec<&Arc<Frame<T>>> {
        x.iter().map(|x| &**x).collect::<Vec<_>>()
    }

    for i in 0..opts.lookahead_distance + 1 {
        v.push((
            i,
            if let Some(frame) = dec.receive_frame_init::<T>(stride, alloc_height) {
                frame
            } else {
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

    if let Some(frame) = dec.receive_frame_init::<T>(stride, alloc_height) {
        v.push((opts.lookahead_distance + 1, frame));
    } else {
        return DetectionResults {
            scene_changes: keyframes,
            frame_count: frameno,
        };
    }

    let x1 = fill_vec(&v);
    let y1 = map_vec(&x1);

    if detector.analyze_next_frame(&y1, frameno as u64, *keyframes.last().unwrap() as u64) {
        keyframes.push(frameno);
    };

    frameno += 1;

    if let Some(progress_fn) = progress_callback {
        progress_fn(frameno, keyframes.len());
    }

    loop {
        let first = v.remove(0);
        let new_last = v.last().unwrap().0 + 1;

        v.push(first);
        let len = v.len();

        let frame_received = dec.receive_frame::<T>(&mut v[len - 1].1);
        if frame_received {
            v[len - 1].0 = new_last;
        } else {
            v.pop().unwrap();
            break;
        }

        let x1 = fill_vec(&v);
        let y1 = map_vec(&x1);
        if detector.analyze_next_frame(&y1, frameno as u64, *keyframes.last().unwrap() as u64) {
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
        let y1 = map_vec(&x1);

        if detector.analyze_next_frame(&y1, frameno as u64, *keyframes.last().unwrap() as u64) {
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
