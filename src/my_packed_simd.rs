extern crate packed_simd;
extern crate rand;

use rand::{Rng, StdRng};
use rand::distributions::{Uniform, Standard, Distribution};
use rand::distributions::uniform::SampleUniform;
use self::packed_simd::*;

pub fn add<T>(a: &[T], b: &[T], c: &mut [T])
    where T: Copy + std::ops::Add<T, Output=T> {
    for ((&a, &b), c) in a.iter().zip(b).zip(c) {
        *c = a + b;
    }
}

pub fn simd_add_unaligned(a: &[u32], b: &[u32], c: &mut [u32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    assert!(a.len() % 16 == 0);
    for i in (0..a.len()).step_by(16) {
        let aa = u32x16::from_slice_unaligned(&a[i..]);
        let bb = u32x16::from_slice_unaligned(&b[i..]);
        let cc = aa + bb;
        cc.write_to_slice_unaligned(&mut c[i..]);
    }
}

pub unsafe fn unsafe_simd_add_unaligned(a: &[u32], b: &[u32], c: &mut [u32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    assert!(a.len() % 16 == 0);
    for i in (0..a.len()).step_by(16) {
        let aa = u32x16::from_slice_unaligned_unchecked(&a[i..]);
        let bb = u32x16::from_slice_unaligned_unchecked(&b[i..]);
        let cc = aa + bb;
        cc.write_to_slice_unaligned_unchecked(&mut c[i..]);
    }
}

pub fn census<T, U>(src: &[T], dst: &mut [U],
                    w: isize, h: isize, window_w: isize, window_h: isize)
    where T: Copy + std::cmp::PartialOrd,
          U: std::ops::Add<U, Output=U> +
             std::ops::Shl<U, Output=U> +
             std::convert::From<u8> {
    assert_eq!(src.len(), dst.len());
    assert_eq!(src.len(), (w * h) as usize);
    assert!(window_w % 2 != 0);
    assert!(window_h % 2 != 0);
    let half_w = window_w / 2;
    let half_h = window_h / 2;

    for y in half_h..(h - half_w) {
        for x in half_w..(w - half_w) {
            let idx = (y * w + x) as usize;
            let center = src[idx];
            let mut val = 0.into();
            for dy in (-half_h)..half_h {
                for dx in (-half_w)..half_w {
                    if dx == 0 && dy == 0 { continue; }
                    let neighbor_idx = ((y + dy) * w + (x + dx)) as usize;
                    let code = if src[neighbor_idx] > center { 1 } else { 0 }; 
                    val = (val << 1.into()) + code.into();
                }
            }
            dst[idx] = val;
        }
    }
}

// 5x5
pub fn simd_census(src: &[f32], dst: &mut [u32], w: isize, h: isize) {
    assert_eq!(src.len(), dst.len());
    assert_eq!(src.len(), (w * h) as usize);
    assert!((w - 4) % 16 == 0);
    let half_w = 2;
    let half_h = 2;
    let zeros = u32x16::splat(0);
    let ones  = u32x16::splat(1);

    for y in half_h..(h - half_w) {
        for x in (half_w..(w - half_w)).step_by(16) {
            let idx = (y * w + x) as usize;
            let center = f32x16::from_slice_unaligned(&src[idx..]);
            let mut val = u32x16::splat(0);
            for dy in (-half_h)..half_h {
                for dx in (-half_w)..half_w {
                    if dx == 0 && dy == 0 { continue; }
                    let neighbor_idx = ((y + dy) * w + (x + dx)) as usize;
                    let neighbor = f32x16::from_slice_unaligned(&src[neighbor_idx..]);
                    let code = neighbor.gt(center).select(ones, zeros);
                    val = (val.rotate_left(ones)) + code;
                }
            }
            val.write_to_slice_unaligned(&mut dst[idx..]);
        }
    }
}

pub unsafe fn unsafe_simd_census(src: &[f32], dst: &mut [u32], w: isize, h: isize) {
    assert_eq!(src.len(), dst.len());
    assert_eq!(src.len(), (w * h) as usize);
    assert!((w - 4) % 16 == 0);
    let half_w = 2;
    let half_h = 2;
    let zeros = u32x16::splat(0);
    let ones  = u32x16::splat(1);

    for y in half_h..(h - half_w) {
        for x in (half_w..(w - half_w)).step_by(16) {
            let idx = (y * w + x) as usize;
            let center = f32x16::from_slice_unaligned_unchecked(&src[idx..]);
            let mut val = u32x16::splat(0);
            for dy in (-half_h)..half_h {
                for dx in (-half_w)..half_w {
                    if dx == 0 && dy == 0 { continue; }
                    let neighbor_idx = ((y + dy) * w + (x + dx)) as usize;
                    let neighbor = f32x16::from_slice_unaligned_unchecked(&src[neighbor_idx..]);
                    let code = neighbor.gt(center).select(ones, zeros);
                    val = val.rotate_left(ones) + code;
                }
            }
            val.write_to_slice_unaligned_unchecked(&mut dst[idx..]);
        }
    }
}

pub fn init_vec<T>(rng: &mut StdRng, size: usize, min_max: Option<(T, T)>) -> Vec<T>
    where T: SampleUniform + std::cmp::PartialOrd,
          Standard: Distribution<T> {
    match min_max {
        Some((min, max)) => {
            assert!(min < max);
            let between = Uniform::from(min..max);
            rng.sample_iter(&between).take(size).collect()
        },
        None => rng.sample_iter(&Standard).take(size).collect()
    }
}
