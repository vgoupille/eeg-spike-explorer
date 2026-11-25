//! `eegspike-core` for the browser, with a plain C ABI (no wasm-bindgen):
//! the page allocates a buffer in WebAssembly memory, writes the signal,
//! calls `detect`, then reads the three result arrays.
//!
//! Same Rust function as the Python extension, so same spikes.

use eegspike_core::detect_channel;
use std::cell::RefCell;

thread_local! {
    static OUT: RefCell<(Vec<u32>, Vec<u32>, Vec<f64>)> = RefCell::new((Vec::new(), Vec::new(), Vec::new()));
}

/// Reserve `n` f64 in WebAssembly memory; the page writes the signal there.
#[no_mangle]
pub extern "C" fn alloc_f64(n: usize) -> *mut f64 {
    let mut v = vec![0.0f64; n];
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Run the detector on `n_ch` channels of `n_t` samples (row major, µV).
/// Returns the number of spike candidates; read them with `out_*`.
#[no_mangle]
pub extern "C" fn detect(x: *const f64, n_ch: usize, n_t: usize, k: f64,
                         smooth_n: usize, refractory_n: usize) -> usize {
    let data = unsafe { std::slice::from_raw_parts(x, n_ch * n_t) };
    OUT.with(|o| {
        let mut o = o.borrow_mut();
        o.0.clear(); o.1.clear(); o.2.clear();
        for c in 0..n_ch {
            for (t, s) in detect_channel(&data[c * n_t..(c + 1) * n_t], k, smooth_n, refractory_n) {
                o.0.push(c as u32);
                o.1.push(t as u32);
                o.2.push(s);
            }
        }
        o.0.len()
    })
}

#[no_mangle]
pub extern "C" fn out_channels() -> *const u32 { OUT.with(|o| o.borrow().0.as_ptr()) }
#[no_mangle]
pub extern "C" fn out_samples() -> *const u32 { OUT.with(|o| o.borrow().1.as_ptr()) }
#[no_mangle]
pub extern "C" fn out_scores() -> *const f64 { OUT.with(|o| o.borrow().2.as_ptr()) }
