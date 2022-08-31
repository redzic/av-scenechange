pub mod decode;
pub mod ffmpeg;
pub mod vapoursynth;

mod y4m;

use std::{
    mem::{transmute, ManuallyDrop},
    sync::Arc,
};

pub use rav1e::scenechange::SceneChangeDetector;
use rav1e::{
    config::{CpuFeatureLevel, EncoderConfig},
    prelude::{ChromaSamplePosition, Frame, Pixel, Plane, PlaneConfig, PlaneData, Sequence},
};

use crate::decode::{Decoder, FrameView};

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

pub fn new_detector<F, D: Decoder<F>, T: Pixel>(
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
pub fn detect_scene_changes<F, D: Decoder<F>, T: Pixel>(
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
    let mut frame_queue = Vec::<(usize, F)>::with_capacity(opts.lookahead_distance + 2);
    let mut keyframes: Vec<usize> = vec![0];

    let mut frameno: usize = 0;

    let (w, h) = {
        let vd = dec.get_video_details();
        (vd.width, vd.height)
    };

    let strict = opts.analysis_speed == SceneDetectionSpeed::Standard;

    // If the stride doesn't match, we have to copy it over to another buffer.
    let frame_copy_needed = !dec.stride_matches::<T>(stride, alloc_height);

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
        // data: PlaneData::new_ref(&[]),
        data: PlaneData::new(0),
    };

    let plane_cfg_luma: PlaneConfig = PlaneConfig {
        alloc_height,
        height: h,
        stride,
        width: w,
        xdec: 0,
        xorigin: 0,
        xpad: 0,
        ydec: 0,
        yorigin: 0,
        ypad: 0,
    };

    let mut frame_copies: Vec<Arc<Frame<T>>> = Vec::new();

    // converts frame_queue or frame_copies
    let fill_vec = |frame_queue: &[(usize, F)]| {
        frame_queue
            .iter()
            .map(|(_, v)| {
                ManuallyDrop::new(Arc::new(unsafe {
                    match D::get_frame_ref::<T>(v, h, w, stride, alloc_height, strict, None) {
                        FrameView::Ref(x) => transmute(x),
                        FrameView::Owned(x) => transmute(x),
                    }
                }))
            })
            .collect::<Vec<_>>()
    };

    fn map_vec<T: Pixel>(x: &[ManuallyDrop<Arc<Frame<T>>>]) -> Vec<&Arc<Frame<T>>> {
        x.iter().map(|x| &**x).collect::<Vec<_>>()
    }

    fn map_vec_copy<T: Pixel>(x: &[Arc<Frame<T>>]) -> Vec<&Arc<Frame<T>>> {
        x.iter().map(|x| &*x).collect::<Vec<_>>()
    }

    for i in 0..opts.lookahead_distance + 1 {
        // receive_frame{_init} does not do frame copy,
        // so we have to do it ourselves
        frame_queue.push((
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

        if frame_copy_needed {
            frame_copies.push(Arc::new(Frame {
                planes: [
                    D::make_copy(
                        &frame_queue.last().unwrap().1,
                        h,
                        w,
                        stride,
                        alloc_height,
                        None,
                    )
                    .unwrap(),
                    empty_plane(),
                    empty_plane(),
                ],
            }));
        }
    }

    frameno += 1;

    // We always have a keyframe at 0, so we "skip" the first analyze_next_frame call
    if let Some(progress_fn) = progress_callback {
        progress_fn(frameno, keyframes.len());
    }

    if let Some(frame) = dec.receive_frame_init::<T>(stride, alloc_height) {
        frame_queue.push((opts.lookahead_distance + 1, frame));

        if frame_copy_needed {
            frame_copies.push(Arc::new(Frame {
                planes: [
                    D::make_copy(
                        &frame_queue.last().unwrap().1,
                        h,
                        w,
                        stride,
                        alloc_height,
                        None,
                    )
                    .unwrap(),
                    empty_plane(),
                    empty_plane(),
                ],
            }));
        }
    } else {
        return DetectionResults {
            scene_changes: keyframes,
            frame_count: frameno,
        };
    }

    // frame_queue holds ORIGINAL frames
    let x1;
    let y1;
    let y2;

    // need to pass in &frame_copies then
    let x = if frame_copy_needed {
        y2 = map_vec_copy(&frame_copies);
        &*y2
    } else {
        x1 = fill_vec(&frame_queue);
        y1 = map_vec(&x1);
        &*y1
    };

    // get &Arc<Frame<T>> from frame_copies

    if detector.analyze_next_frame(x, frameno as u64, *keyframes.last().unwrap() as u64) {
        keyframes.push(frameno);
    };

    // if should_drop {
    //     drop(y1);
    //     while let Some(x) = x1.pop() {
    //         let _ = ManuallyDrop::into_inner(x);
    //     }
    // }

    frameno += 1;

    if let Some(progress_fn) = progress_callback {
        progress_fn(frameno, keyframes.len());
    }

    loop {
        let first = frame_queue.remove(0);
        let new_last = frame_queue.last().unwrap().0 + 1;

        frame_queue.push(first);

        if frame_copy_needed {
            let first_copy = frame_copies.remove(0);
            frame_copies.push(first_copy);
        }

        let len = frame_queue.len();

        let frame_received = dec.receive_frame::<T>(&mut frame_queue[len - 1].1);
        if frame_received {
            if frame_copy_needed {
                D::make_copy::<T>(
                    &frame_queue[len - 1].1,
                    h,
                    w,
                    stride,
                    alloc_height,
                    // Some(&mut frame_copies[len - 1].planes[0]),
                    #[allow(mutable_transmutes)]
                    unsafe {
                        Some(transmute(&frame_copies[len - 1].planes[0]))
                    },
                );
            }
            frame_queue[len - 1].0 = new_last;
        } else {
            frame_queue.pop().unwrap();
            if frame_copy_needed {
                frame_copies.pop().unwrap();
            }
            break;
        }

        // frame_queue holds ORIGINAL frames
        let x1;
        let y1;
        let y2;

        // need to pass in &frame_copies then
        let x = if frame_copy_needed {
            y2 = map_vec_copy(&frame_copies);
            &*y2
        } else {
            x1 = fill_vec(&frame_queue);
            y1 = map_vec(&x1);
            &*y1
        };

        if detector.analyze_next_frame(x, frameno as u64, *keyframes.last().unwrap() as u64) {
            keyframes.push(frameno);
        };

        frameno += 1;

        if let Some(progress_fn) = progress_callback {
            progress_fn(frameno, keyframes.len());
        }
    }

    while frame_queue.len() != 1 {
        frameno += 1;

        // frame_queue holds ORIGINAL frames
        let x1;
        let y1;
        let y2;

        // need to pass in &frame_copies then
        let x = if frame_copy_needed {
            y2 = map_vec_copy(&frame_copies);
            &*y2
        } else {
            x1 = fill_vec(&frame_queue);
            y1 = map_vec(&x1);
            &*y1
        };

        if detector.analyze_next_frame(x, frameno as u64, *keyframes.last().unwrap() as u64) {
            keyframes.push(frameno);
        };

        if let Some(progress_fn) = progress_callback {
            progress_fn(frameno, keyframes.len());
        }

        frame_queue.remove(0);
        if frame_copy_needed {
            frame_copies.remove(0);
        }
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
