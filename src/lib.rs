#![feature(test)]

extern crate test;
extern crate rand;
extern crate packed_simd;

use std::ops::{Add, Shl};
use std::cmp::PartialOrd;
use std::convert::From;
use rand::{Rng, StdRng, SeedableRng};
use rand::distributions::{Uniform, Standard, Distribution};
use rand::distributions::uniform::SampleUniform;
use packed_simd::*;

fn add<T>(a: &[T], b: &[T], c: &mut [T])
    where T: Copy + Add<T, Output=T> {
    for ((&a, &b), c) in a.iter().zip(b).zip(c) {
        *c = a + b;
    }
}

fn simd_add_unaligned(a: &[u32], b: &[u32], c: &mut [u32]) {
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

unsafe fn unsafe_simd_add_unaligned(a: &[u32], b: &[u32], c: &mut [u32]) {
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

fn census<T, U>(src: &[T], dst: &mut [U],
                w: isize, h: isize, window_w: isize, window_h: isize)
    where T: Copy + PartialOrd,
          U: Add<U, Output=U> + Shl<U, Output=U> + From<u8> {
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

fn simd_census(src: &[f32], dst: &mut [u64],
               w: isize, h: isize, window_w: isize, window_h: isize) {
    assert_eq!(src.len(), dst.len());
    assert_eq!(src.len(), (w * h) as usize);
    assert!(window_w % 2 != 0);
    assert!(window_h % 2 != 0);
    assert!((w - window_w + 1) % 8 == 0);
    let half_w = window_w / 2;
    let half_h = window_h / 2;
    let zeros = u64x8::splat(0);
    let ones  = u64x8::splat(1);

    for y in half_h..(h - half_w) {
        for x in (half_w..(w - half_w)).step_by(8) {
            let idx = (y * w + x) as usize;
            let center = f32x8::from_slice_unaligned(&src[idx..]);
            let mut val = u64x8::splat(0);
            for dy in (-half_h)..half_h {
                for dx in (-half_w)..half_w {
                    if dx == 0 && dy == 0 { continue; }
                    let neighbor_idx = ((y + dy) * w + (x + dx)) as usize;
                    let neighbor = f32x8::from_slice_unaligned(&src[neighbor_idx..]);
                    let code = neighbor.gt(center).select(ones, zeros);
                    val = (val.rotate_left(ones)) + code;
                }
            }
            val.write_to_slice_unaligned(&mut dst[idx..]);
        }
    }
}

unsafe fn unsafe_simd_census(src: &[f32], dst: &mut [u64],
                             w: isize, h: isize, window_w: isize, window_h: isize) {
    assert_eq!(src.len(), dst.len());
    assert_eq!(src.len(), (w * h) as usize);
    assert!(window_w % 2 != 0);
    assert!(window_h % 2 != 0);
    assert!((w - window_w + 1) % 8 == 0);
    let half_w = window_w / 2;
    let half_h = window_h / 2;
    let zeros = u64x8::splat(0);
    let ones  = u64x8::splat(1);

    for y in half_h..(h - half_w) {
        for x in (half_w..(w - half_w)).step_by(8) {
            let idx = (y * w + x) as usize;
            let center = f32x8::from_slice_unaligned_unchecked(&src[idx..]);
            let mut val = u64x8::splat(0);
            for dy in (-half_h)..half_h {
                for dx in (-half_w)..half_w {
                    if dx == 0 && dy == 0 { continue; }
                    let neighbor_idx = ((y + dy) * w + (x + dx)) as usize;
                    let neighbor = f32x8::from_slice_unaligned_unchecked(&src[neighbor_idx..]);
                    let code = neighbor.gt(center).select(ones, zeros);
                    val = val.rotate_left(ones) + code;
                }
            }
            val.write_to_slice_unaligned_unchecked(&mut dst[idx..]);
        }
    }
}

fn init_vec<R: Rng, T>(rng: &mut R, size: usize, min_max: Option<(T, T)>) -> Vec<T>
    where T: SampleUniform + PartialOrd,
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

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    fn rng() -> StdRng {
        let seed = [57; 32];
        SeedableRng::from_seed(seed)
    }
    
    const ADD_SIZE: usize = 100_000;
    const RANGE_FOR_ADD: (u32, u32) = (0, std::u32::MAX / 10);
    
    #[bench]
    fn bench_add(bm: &mut Bencher) {
        let mut rng = rng();

        let a = init_vec(&mut rng, ADD_SIZE, Some(RANGE_FOR_ADD));
        let b = init_vec(&mut rng, ADD_SIZE, Some(RANGE_FOR_ADD));
        let mut c = vec![0u32; ADD_SIZE];
        bm.iter(|| add(&a, &b, &mut c));
    }

    #[test]
    fn test_simd_add_unaligned() {
        let mut rng = rng();

        let a = init_vec(&mut rng, ADD_SIZE, Some(RANGE_FOR_ADD));
        let b = init_vec(&mut rng, ADD_SIZE, Some(RANGE_FOR_ADD));
        let mut c = vec![0u32; ADD_SIZE];
        let mut d = vec![0u32; ADD_SIZE];
        add(&a, &b, &mut c);
        simd_add_unaligned(&a, &b, &mut d);
        for (c, d) in c.iter().zip(d) {
            assert_eq!(*c, d);
        }
    }

    #[bench]
    fn bench_simd_add_unaligned(bm: &mut Bencher) {
        let mut rng = rng();

        let a = init_vec(&mut rng, ADD_SIZE, Some(RANGE_FOR_ADD));
        let b = init_vec(&mut rng, ADD_SIZE, Some(RANGE_FOR_ADD));
        let mut c = vec![0u32; ADD_SIZE];
        bm.iter(|| simd_add_unaligned(&a, &b, &mut c));
    }
    
    #[test]
    fn test_unsafe_simd_add_unaligned() {
        let mut rng = rng();

        let a = init_vec(&mut rng, ADD_SIZE, Some(RANGE_FOR_ADD));
        let b = init_vec(&mut rng, ADD_SIZE, Some(RANGE_FOR_ADD));
        let mut c = vec![0u32; ADD_SIZE];
        let mut d = vec![0u32; ADD_SIZE];
        add(&a, &b, &mut c);
        unsafe { unsafe_simd_add_unaligned(&a, &b, &mut d); }
        for (c, d) in c.iter().zip(d) {
            assert_eq!(*c, d);
        }
    }

    #[bench]
    fn bench_unsafe_simd_add_unaligned(bm: &mut Bencher) {
        let mut rng = rng();

        let a = init_vec(&mut rng, ADD_SIZE, Some(RANGE_FOR_ADD));
        let b = init_vec(&mut rng, ADD_SIZE, Some(RANGE_FOR_ADD));
        let mut c = vec![0u32; ADD_SIZE];
        bm.iter(|| unsafe { unsafe_simd_add_unaligned(&a, &b, &mut c) });
    }
    
    const WIDTH:  isize = 1024;
    const HEIGHT: isize = 360;
    const CENSUS_SIZE: usize = (WIDTH * HEIGHT) as usize;

    #[bench]
    fn bench_census(bm: &mut Bencher) {
        let mut rng = rng();

        let src = init_vec::<_, f32>(&mut rng, CENSUS_SIZE, None);
        let mut dst = vec![0u64; CENSUS_SIZE];
        bm.iter(|| census(&src, &mut dst, WIDTH, HEIGHT, 9, 7));
    }
    
    #[test]
    fn test_census_simd() {
        let mut rng = rng();

        let src = init_vec::<_, f32>(&mut rng, CENSUS_SIZE, None);
        let mut dst1 = vec![0u64; CENSUS_SIZE];
        let mut dst2 = vec![0u64; CENSUS_SIZE];
        census(&src, &mut dst1, WIDTH, HEIGHT, 9, 7);
        simd_census(&src, &mut dst2, WIDTH, HEIGHT, 9, 7);
        for (dst1, dst2) in dst1.iter().zip(dst2) {
            assert_eq!(*dst1, dst2);
        }
    }


    #[bench]
    fn bench_simd_census(bm: &mut Bencher) {
        let mut rng = rng();

        let src = init_vec::<_, f32>(&mut rng, CENSUS_SIZE, None);
        let mut dst = vec![0u64; CENSUS_SIZE];
        bm.iter(|| simd_census(&src, &mut dst, WIDTH, HEIGHT, 9, 7));
    }
    
       #[test]
    fn test_unsafe_census_simd() {
        let mut rng = rng();

        let src = init_vec::<_, f32>(&mut rng, CENSUS_SIZE, None);
        let mut dst1 = vec![0u64; CENSUS_SIZE];
        let mut dst2 = vec![0u64; CENSUS_SIZE];
        census(&src, &mut dst1, WIDTH, HEIGHT, 9, 7);
        unsafe { unsafe_simd_census(&src, &mut dst2, WIDTH, HEIGHT, 9, 7); }
        for (dst1, dst2) in dst1.iter().zip(dst2) {
            assert_eq!(*dst1, dst2);
        }
    }

    #[bench]
    fn bench_unsafe_simd_census(bm: &mut Bencher) {
        let mut rng = rng();

        let src = init_vec::<_, f32>(&mut rng, CENSUS_SIZE, None);
        let mut dst = vec![0u64; CENSUS_SIZE];
        bm.iter(|| unsafe { unsafe_simd_census(&src, &mut dst, WIDTH, HEIGHT, 9, 7) });
    }
}
