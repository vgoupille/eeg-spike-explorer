//! Python bindings (PyO3) for `eegspike-core`.
//!
//! Channels are processed in parallel with scoped threads, without the GIL.
//! The same core is compiled to WebAssembly in `wasm/` for the browser.

use numpy::{PyReadonlyArray2, PyArray1, IntoPyArray};
use pyo3::prelude::*;
use eegspike_core::detect_channel;

/// detect(x, k, smooth_n, refractory_n) -> (channels, samples, scores)
///
/// `x` is a C-contiguous float64 array of shape (n_channels, n_times).
#[pyfunction]
#[pyo3(signature = (x, k=8.0, smooth_n=5, refractory_n=50))]
fn detect<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    k: f64,
    smooth_n: usize,
    refractory_n: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
    let arr = x.as_array();
    let n_ch = arr.shape()[0];
    let rows: Vec<Vec<f64>> = (0..n_ch).map(|c| arr.row(c).to_vec()).collect();

    let results: Vec<Vec<(usize, f64)>> = py.detach(|| {
        std::thread::scope(|s| {
            let handles: Vec<_> = rows
                .iter()
                .map(|r| s.spawn(move || detect_channel(r, k, smooth_n, refractory_n)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    });

    let mut chs = Vec::new();
    let mut samples = Vec::new();
    let mut scores = Vec::new();
    for (c, peaks) in results.into_iter().enumerate() {
        for (t, s) in peaks {
            chs.push(c as i64);
            samples.push(t as i64);
            scores.push(s);
        }
    }
    Ok((chs.into_pyarray(py), samples.into_pyarray(py), scores.into_pyarray(py)))
}

#[pymodule]
fn eegspike_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    Ok(())
}
