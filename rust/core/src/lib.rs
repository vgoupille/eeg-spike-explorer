//! eegspike-core: spike candidate detection, shared by the Python extension
//! (PyO3, `rust/`) and the browser build (WebAssembly, `wasm/`).
//!
//! 1. Teager-Kaiser energy operator: `tk[t] = x[t]^2 - x[t-1] * x[t+1]`
//! 2. Centered moving average over `smooth_n` samples (zero padded)
//! 3. Robust threshold: `median + k * MAD / 0.6745`
//! 4. Local maxima above threshold, with a refractory period
//!    (two peaks closer than `refractory_n` keep only the highest).
//!
//! It mirrors `eegspike.detect._detect_numpy` exactly. No dependencies.

pub fn median(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    let (_, m, _) = v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    let upper = *m;
    if n % 2 == 1 {
        upper
    } else {
        // largest value of the lower half
        let lower = v[..mid]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lower + upper)
    }
}

/// Detect spike candidates on one channel. Returns (sample, score) pairs,
/// where score = energy / threshold (> 1).
pub fn detect_channel(x: &[f64], k: f64, smooth_n: usize, refractory_n: usize) -> Vec<(usize, f64)> {
    let n = x.len();
    if n < 3 {
        return Vec::new();
    }
    // 1. Teager-Kaiser energy
    let mut tk = vec![0.0; n];
    for t in 1..n - 1 {
        tk[t] = x[t] * x[t] - x[t - 1] * x[t + 1];
    }
    // 2. centered moving average (zero padded, like np.convolve(mode="same"))
    let w = smooth_n.max(1);
    let h = w / 2;
    let mut csum = vec![0.0; n + 1];
    for t in 0..n {
        csum[t + 1] = csum[t] + tk[t];
    }
    let mut e = vec![0.0; n];
    for t in 0..n {
        // window [t - h, t - h + w)
        let lo = t as isize - h as isize;
        let hi = lo + w as isize;
        let a = lo.max(0) as usize;
        let b = (hi.min(n as isize)).max(0) as usize;
        e[t] = (csum[b] - csum[a]) / w as f64;
    }
    // 3. robust threshold
    let mut tmp = e.clone();
    let med = median(&mut tmp);
    for (d, v) in tmp.iter_mut().zip(e.iter()) {
        *d = (v - med).abs();
    }
    let mad = median(&mut tmp);
    let thr = med + k * mad / 0.6745;
    // 4. local maxima + refractory period
    let mut peaks: Vec<(usize, f64)> = Vec::new();
    for t in 1..n - 1 {
        let v = e[t];
        if v > thr && v > e[t - 1] && v >= e[t + 1] {
            if let Some(last) = peaks.last_mut() {
                if t - last.0 < refractory_n {
                    if v > last.1 {
                        *last = (t, v);
                    }
                    continue;
                }
            }
            peaks.push((t, v));
        }
    }
    peaks.into_iter().map(|(t, v)| (t, v / thr)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_isolated_spike() {
        let mut x = vec![0.0; 1000];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i as f64) * 0.05).sin() * 10.0;
        }
        x[500] += 200.0;
        let p = detect_channel(&x, 8.0, 5, 50);
        assert_eq!(p.len(), 1);
        assert!((p[0].0 as isize - 500).abs() <= 2);
    }

    #[test]
    fn median_even_odd() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
    }
}
