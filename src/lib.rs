#![feature(test)]

pub mod my_packed_simd;

extern crate rand;
extern crate test;

#[cfg(test)]
mod tests {
    use ::my_packed_simd::*;
    use ::rand::{StdRng, SeedableRng};  
    use ::test::Bencher;

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
    
    const WIDTH:  isize = 1012;
    const HEIGHT: isize = 360;
    const CENSUS_SIZE: usize = (WIDTH * HEIGHT) as usize;

    #[bench]
    fn bench_census(bm: &mut Bencher) {
        let mut rng = rng();

        let src = init_vec::<f32>(&mut rng, CENSUS_SIZE, None);
        let mut dst = vec![0u32; CENSUS_SIZE];
        bm.iter(|| census(&src, &mut dst, WIDTH, HEIGHT, 5, 5));
    }
    
    #[test]
    fn test_census_simd() {
        let mut rng = rng();

        let src = init_vec::<f32>(&mut rng, CENSUS_SIZE, None);
        let mut dst1 = vec![0u32; CENSUS_SIZE];
        let mut dst2 = vec![0u32; CENSUS_SIZE];
        census(&src, &mut dst1, WIDTH, HEIGHT, 5, 5);
        simd_census(&src, &mut dst2, WIDTH, HEIGHT);
        for (dst1, dst2) in dst1.iter().zip(dst2) {
            assert_eq!(*dst1, dst2);
        }
    }

    #[bench]
    fn bench_simd_census(bm: &mut Bencher) {
        let mut rng = rng();

        let src = init_vec::<f32>(&mut rng, CENSUS_SIZE, None);
        let mut dst = vec![0u32; CENSUS_SIZE];
        bm.iter(|| simd_census(&src, &mut dst, WIDTH, HEIGHT));
    }
    
    #[test]
    fn test_unsafe_census_simd() {
        let mut rng = rng();

        let src = init_vec::<f32>(&mut rng, CENSUS_SIZE, None);
        let mut dst1 = vec![0u32; CENSUS_SIZE];
        let mut dst2 = vec![0u32; CENSUS_SIZE];
        census(&src, &mut dst1, WIDTH, HEIGHT, 5, 5);
        unsafe { unsafe_simd_census(&src, &mut dst2, WIDTH, HEIGHT); }
        for (dst1, dst2) in dst1.iter().zip(dst2) {
            assert_eq!(*dst1, dst2);
        }
    }

    #[bench]
    fn bench_unsafe_simd_census(bm: &mut Bencher) {
        let mut rng = rng();

        let src = init_vec::<f32>(&mut rng, CENSUS_SIZE, None);
        let mut dst = vec![0u32; CENSUS_SIZE];
        bm.iter(|| unsafe { unsafe_simd_census(&src, &mut dst, WIDTH, HEIGHT) });
    }
}
