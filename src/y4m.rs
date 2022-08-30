use std::io::Read;

use rav1e::prelude::{
    ChromaSamplePosition,
    ChromaSampling,
    Frame,
    Pixel,
    Plane,
    PlaneConfig,
    Rational,
};

use crate::decode::VideoDetails;

#[allow(unused)]
pub(crate) fn get_video_details<R: Read>(dec: &y4m::Decoder<R>) -> VideoDetails {
    let width = dec.get_width();
    let height = dec.get_height();
    let color_space = dec.get_colorspace();
    let bit_depth = color_space.get_bit_depth();
    let (chroma_sampling, chroma_sample_position) = map_y4m_color_space(color_space);
    let framerate = dec.get_framerate();
    let time_base = Rational::new(framerate.den as u64, framerate.num as u64);

    VideoDetails {
        width,
        height,
        bit_depth,
        chroma_sampling,
        chroma_sample_position,
        time_base,
        luma_padding: 0,
    }
}

#[allow(unused)]
const fn map_y4m_color_space(
    color_space: y4m::Colorspace,
) -> (ChromaSampling, ChromaSamplePosition) {
    use y4m::Colorspace::*;
    use ChromaSamplePosition::*;
    use ChromaSampling::*;
    match color_space {
        Cmono => (Cs400, Unknown),
        C420jpeg | C420paldv => (Cs420, Unknown),
        C420mpeg2 => (Cs420, Vertical),
        C420 | C420p10 | C420p12 => (Cs420, Colocated),
        C422 | C422p10 | C422p12 => (Cs422, Colocated),
        C444 | C444p10 | C444p12 => (Cs444, Colocated),
    }
}

#[allow(unused)]
pub fn read_video_frame<R: Read, T: Pixel>(
    dec: &mut y4m::Decoder<R>,
    cfg: &VideoDetails,
) -> anyhow::Result<Frame<T>> {
    const SB_SIZE_LOG2: usize = 6;
    const SB_SIZE: usize = 1 << SB_SIZE_LOG2;
    const SUBPEL_FILTER_SIZE: usize = 8;
    const FRAME_MARGIN: usize = 16 + SUBPEL_FILTER_SIZE;
    const LUMA_PADDING: usize = SB_SIZE + FRAME_MARGIN;

    let _bytes = dec.get_bytes_per_sample();
    dec.read_frame()
        .map(|frame| {
            let y_plane = frame.get_y_plane();
            Frame {
                planes: [
                    Plane {
                        cfg: PlaneConfig::new(
                            cfg.width,
                            cfg.height,
                            0,
                            0,
                            0,
                            0,
                            std::mem::size_of::<T>(),
                        ),
                        data: todo!(),
                    },
                    Plane::<T>::new(0, 0, 0, 0, 0, 0),
                    Plane::<T>::new(0, 0, 0, 0, 0, 0),
                ],
            }
        })
        .map_err(|e| e.into())
}

impl Default for VideoDetails {
    fn default() -> Self {
        VideoDetails {
            width: 640,
            height: 480,
            bit_depth: 8,
            chroma_sampling: ChromaSampling::Cs420,
            chroma_sample_position: ChromaSamplePosition::Unknown,
            time_base: Rational { num: 30, den: 1 },
            luma_padding: 0,
        }
    }
}
