"""EE210 (synthetic): a small, hand-written Signals and Systems course used to score the KPIs
before the instructors' real question set exists. Four modules, four pages each. Every fact the
KPI questions ask about is stated on a known page, so answers and citations can be checked.

    python eval/synthetic/ee210_course.py      # writes the four PDFs next to this file (course/)
"""
from __future__ import annotations

from pathlib import Path

COURSE_TITLE = "EE210 Signals and Systems (synthetic evaluation course)"

MODULES: dict[str, list[tuple[str, list[str]]]] = {
    "EE210 Module 1 Signals and Sampling": [
        ("Module 1 - Signals", [
            "1.1 Continuous-time and discrete-time signals",
            "A continuous-time signal x(t) is defined for every real t; a discrete-time signal x[n] is defined",
            "only at integer n.",
            "1.2 Energy and power",
            "The energy of x[n] is E = sum over n of |x[n]|^2. Example: x[n] = {1, 2, 2} has energy 1 + 4 + 4 = 9.",
            "The average power of a periodic signal is the energy in one period divided by the period.",
            "A signal with finite energy has zero average power (an energy signal). A periodic sinusoid has",
            "finite power and infinite energy (a power signal).",
            "1.3 Periodic signals",
            "x(t) is periodic with period T0 if x(t + T0) = x(t) for all t. The fundamental frequency is",
            "f0 = 1/T0 Hz, or w0 = 2 pi f0 rad/s. Example: cos(2 pi 50 t) has period T0 = 20 ms.",
            "1.4 Even and odd parts",
            "Every signal splits into an even part xe(t) = [x(t) + x(-t)]/2 and an odd part",
            "xo(t) = [x(t) - x(-t)]/2, with x(t) = xe(t) + xo(t).",
            "1.5 Unit step and unit impulse",
            "u(t) = 1 for t >= 0 and 0 for t < 0. The unit impulse delta(t) has unit area. Sifting property:",
            "the integral of x(t) delta(t - t0) dt equals x(t0). In discrete time, delta[n] = u[n] - u[n-1].",
        ]),
        ("Module 1 - Transformations and sinusoids", [
            "2.1 Time shift",
            "x(t - t0) is x(t) delayed, that is shifted to the right, by t0 when t0 > 0.",
            "2.2 Time scaling",
            "x(a t) compresses the signal in time when |a| > 1 and stretches it when |a| < 1.",
            "Example: x(2t) plays a recording twice as fast.",
            "2.3 Time reversal",
            "x(-t) mirrors the signal about t = 0.",
            "2.4 Sinusoids",
            "x(t) = A cos(w0 t + phi) has amplitude A, angular frequency w0 = 2 pi f0 and phase phi.",
            "Euler's formula gives cos(w t) = [e^{jwt} + e^{-jwt}] / 2.",
            "2.5 Discrete-time sinusoids",
            "cos(w0 n) is periodic only if w0 / (2 pi) is a rational number m/N; the period is then the smallest",
            "such N. Example: cos(pi n / 4) has period N = 8. cos(n), with w0 = 1 rad, is not periodic because",
            "1 / (2 pi) is not rational.",
            "2.6 Order of operations",
            "For y(t) = x(a t - b), shift first by b and then scale by a; equivalently y(t) = x(a (t - b/a)).",
        ]),
        ("Module 1 - Sampling and aliasing", [
            "3.1 Sampling",
            "Taking x[n] = x(n Ts) with sampling period Ts gives the sampling frequency fs = 1/Ts.",
            "Example: Ts = 0.5 ms gives fs = 2000 Hz.",
            "3.2 Sampling theorem",
            "A signal band-limited to B Hz is recovered exactly from its samples if fs > 2B.",
            "The Nyquist rate is 2B; the Nyquist frequency is fs/2.",
            "3.3 Aliasing",
            "If fs is too low, a tone at f appears at the alias frequency |f - k fs| for the integer k that brings",
            "it into the range 0 to fs/2. Example: a 9 kHz tone sampled at 12 kHz appears at 3 kHz (12 - 9 = 3).",
            "3.4 Anti-aliasing filter",
            "Before sampling, a low-pass filter with cutoff below fs/2 removes components that would alias.",
            "Aliasing cannot be removed after sampling, because the folded components are already mixed",
            "with the real ones.",
            "3.5 Oversampling",
            "Sampling well above the Nyquist rate relaxes the anti-aliasing filter: its transition band can be",
            "much wider, so a simpler, cheaper filter is enough.",
            "EE210 tag: SYN_M1_SAMPLING_P3",
        ]),
        ("Module 1 - Quantization and reconstruction", [
            "4.1 Quantization",
            "An N-bit uniform quantizer has 2^N levels. Over a range from -V to +V the step size is 2V / 2^N.",
            "Example: a 3-bit quantizer over -1 V to +1 V has 8 levels and a step of 0.25 V.",
            "4.2 Signal-to-quantization-noise ratio",
            "For a full-scale sinusoid the SQNR is about 6.02 N + 1.76 dB. A 16-bit converter gives about 98 dB.",
            "Each extra bit adds about 6 dB, because it halves the step size and so halves the error amplitude.",
            "4.3 Reconstruction",
            "An ideal low-pass filter with cutoff fs/2 rebuilds x(t) from its samples (sinc interpolation).",
            "Practical DACs use a zero-order hold followed by a smoothing filter.",
            "4.4 Worked example",
            "x(t) = 3 cos(2 pi 400 t) + cos(2 pi 1500 t). The highest frequency is 1500 Hz, so B = 1500 Hz and",
            "the Nyquist rate is 3000 samples per second. Sampling at 4000 Hz is safe. Sampling at 2000 Hz makes",
            "the 1500 Hz term alias to 2000 - 1500 = 500 Hz, while the 400 Hz term is unaffected.",
        ]),
    ],
    "EE210 Module 2 LTI Systems and Convolution": [
        ("Module 2 - System properties", [
            "1.1 Linearity",
            "A system is linear if it obeys superposition: input a x1 + b x2 gives output a y1 + b y2.",
            "1.2 Time invariance",
            "A system is time-invariant if delaying the input by n0 only delays the output by n0.",
            "1.3 Causality",
            "A system is causal if the output at time n depends only on present and past inputs.",
            "1.4 Memoryless systems",
            "The output depends only on the input at the same instant, for example y[n] = 3 x[n].",
            "1.5 BIBO stability",
            "A system is BIBO stable if every bounded input gives a bounded output.",
            "1.6 Examples",
            "y[n] = x[n] + x[n-1] is linear, time-invariant, causal and stable.",
            "y[n] = n x[n] is linear but time-varying, because the gain n changes with time.",
            "y[n] = x[n]^2 is nonlinear. y(t) = x(t + 1) is non-causal: it needs a future input value.",
            "y[n] = x[n] + 2 is not linear: a zero input gives the output 2, which superposition forbids.",
        ]),
        ("Module 2 - Convolution", [
            "2.1 Impulse response",
            "An LTI system is fully described by its impulse response h[n], the output for the input delta[n].",
            "2.2 Convolution",
            "y[n] = sum over k of x[k] h[n - k]. In continuous time, y(t) = integral of x(tau) h(t - tau) d tau.",
            "2.3 Output length",
            "Convolving a length-L sequence with a length-M sequence gives L + M - 1 samples.",
            "2.4 Worked example",
            "x[n] = {1, 2, 3} and h[n] = {1, 1} give y[n] = {1, 3, 5, 3}.",
            "2.5 Properties",
            "Convolution is commutative, associative and distributive. Systems in series (cascade) have the",
            "overall impulse response h1 * h2 (their convolution); systems in parallel have h1 + h2.",
            "2.6 Continuous-time example",
            "u(t) convolved with e^{-2t} u(t) gives y(t) = (1/2)(1 - e^{-2t}) u(t).",
        ]),
        ("Module 2 - Causality and stability from h", [
            "3.1 Causal LTI systems",
            "An LTI system is causal if and only if h[n] = 0 for n < 0.",
            "3.2 Stable LTI systems",
            "An LTI system is BIBO stable if and only if the sum of |h[n]| is finite (h is absolutely summable).",
            "3.3 Examples",
            "h[n] = (0.5)^n u[n] is causal and stable: the sum of |h[n]| is 1 / (1 - 0.5) = 2.",
            "h[n] = 2^n u[n] is unstable: its terms grow without bound, so the sum diverges.",
            "3.4 Step response",
            "The step response is s[n] = sum of h[k] for k <= n. For h[n] = (0.5)^n u[n], s[n] approaches 2.",
            "3.5 Poles",
            "A causal LTI system with a rational transfer function is stable if all its poles lie inside the",
            "unit circle, |z| < 1. For continuous-time systems the poles must lie in the left half of the s-plane.",
            "EE210 tag: SYN_M2_STABILITY_P3",
        ]),
        ("Module 2 - Difference equations, FIR and IIR", [
            "4.1 Difference equations",
            "y[n] - 0.5 y[n-1] = x[n] has the impulse response h[n] = (0.5)^n u[n] (a pole at z = 0.5).",
            "4.2 Transfer function",
            "H(z) = Y(z) / X(z). For y[n] - 0.5 y[n-1] = x[n], H(z) = 1 / (1 - 0.5 z^-1).",
            "4.3 FIR and IIR",
            "A finite impulse response (FIR) system has a finite-length h[n] and no feedback. An infinite",
            "impulse response (IIR) system uses feedback of past outputs, so h[n] never ends.",
            "4.4 Moving average",
            "y[n] = (x[n] + x[n-1] + x[n-2]) / 3 is a 3-point FIR moving average, h[n] = {1/3, 1/3, 1/3}.",
            "4.5 Worked example",
            "For the 3-point moving average and a unit step input the output is 1/3 at n = 0, 2/3 at n = 1",
            "and 1 for every n >= 2.",
        ]),
    ],
    "EE210 Module 3 Fourier Analysis DFT and Filters": [
        ("Module 3 - Fourier transform", [
            "1.1 Fourier series",
            "A periodic signal with fundamental frequency f0 is a sum of harmonics at k f0.",
            "1.2 Fourier transform",
            "X(f) = integral of x(t) e^{-j 2 pi f t} dt.",
            "1.3 Rectangular pulse",
            "A pulse of height 1 and width T centred at 0 has the transform T sinc(f T), where",
            "sinc(x) = sin(pi x) / (pi x). Its first zero crossings are at f = +1/T and -1/T, so a wider pulse",
            "has a narrower spectrum.",
            "1.4 Other pairs",
            "delta(t) <-> 1 (a flat spectrum). e^{-a t} u(t) <-> 1 / (a + j 2 pi f) for a > 0.",
            "1.5 Properties",
            "A time shift by t0 multiplies X(f) by e^{-j 2 pi f t0}: the magnitude is unchanged, only the phase",
            "changes. Convolution in time is multiplication in frequency. Parseval: the energy is the same",
            "in time and in frequency.",
        ]),
        ("Module 3 - The DFT", [
            "2.1 Definition",
            "X[k] = sum over n = 0..N-1 of x[n] e^{-j 2 pi k n / N}, for k = 0..N-1.",
            "2.2 Bin frequency and resolution",
            "Bin k corresponds to the frequency f = k fs / N. The frequency resolution (bin spacing) is fs / N.",
            "2.3 Worked example",
            "fs = 8000 Hz and N = 256 give 31.25 Hz per bin; a 1000 Hz tone falls in bin 1000 / 31.25 = 32.",
            "2.4 Zero-padding",
            "Zero-padding interpolates the spectrum (more points) but does not improve the true resolution,",
            "which depends on the duration of the recorded signal.",
            "2.5 Symmetry",
            "For a real signal X[N - k] is the complex conjugate of X[k]. X[0] is the sum of all samples (DC).",
            "EE210 tag: SYN_M3_DFT_P2",
        ]),
        ("Module 3 - FFT and windows", [
            "3.1 FFT",
            "A radix-2 FFT needs about (N/2) log2 N complex multiplications instead of N^2 for the direct DFT.",
            "For N = 1024 that is 5,120 instead of 1,048,576, about 205 times fewer. N must be a power of 2.",
            "3.2 Spectral leakage",
            "When a tone does not fall exactly on a bin, its energy leaks into neighbouring bins.",
            "3.3 Windows",
            "Multiplying by a window such as the Hann window w[n] = 0.5 - 0.5 cos(2 pi n / (N - 1)) reduces",
            "leakage, at the cost of a wider main lobe.",
            "3.4 Circular convolution",
            "Multiplying two N-point DFTs gives circular convolution. To get the linear convolution of",
            "sequences of lengths L and M, zero-pad both to at least L + M - 1 points.",
        ]),
        ("Module 3 - Filters", [
            "4.1 Ideal filters",
            "A low-pass filter passes frequencies below its cutoff fc and blocks those above.",
            "4.2 RC low-pass filter",
            "A first-order RC low-pass filter has the cutoff fc = 1 / (2 pi R C).",
            "Example: R = 1 kOhm and C = 1 uF give fc = 159 Hz (approximately). At fc the gain is 1/sqrt(2),",
            "that is -3 dB.",
            "4.3 Decibels",
            "Gain in dB is 20 log10(|H|). A gain of 0.1 is -20 dB and a gain of 10 is +20 dB.",
            "4.4 FIR versus IIR filters",
            "FIR filters are always stable and have exactly linear phase when their coefficients are symmetric.",
            "IIR filters reach a sharp response with a much lower order, but they can be unstable and do not",
            "have linear phase.",
            "4.5 Choosing",
            "Use FIR when phase matters (audio, data communications); use IIR when computation must be minimal.",
        ]),
    ],
    "EE210 Module 4 DSP in Python Solved Exercises": [
        ("Module 4 - Python: energy, power, moving average, convolution", [
            "Solved exercise P1: signal energy and power in Python with NumPy",
            "    import numpy as np",
            "    def energy(x):",
            "        x = np.asarray(x)",
            "        return float(np.sum(np.abs(x) ** 2))",
            "    def average_power(x):",
            "        return float(np.mean(np.abs(np.asarray(x)) ** 2))",
            "The RMS value is the square root of the average power: np.sqrt(np.mean(np.abs(x) ** 2)).",
            "Solved exercise P2: causal moving average with np.convolve (output has the same length as x)",
            "    def moving_average(x, M):",
            "        h = np.ones(M) / M",
            "        return np.convolve(x, h)[:len(x)]",
            "Solved exercise P3: linear convolution",
            "np.convolve(x, h) returns the linear convolution; it has len(x) + len(h) - 1 samples.",
        ]),
        ("Module 4 - Python: aliasing and the DFT", [
            "Solved exercise P4: alias frequency of a sampled tone",
            "    def alias_frequency(f, fs):",
            "        f = f % fs",
            "        return min(f, fs - f)      # folded into 0 .. fs/2",
            "The Nyquist rate of a signal with components freqs is 2 * max(freqs). Sampling is safe (no",
            "aliasing) when fs > 2 * fmax.",
            "Solved exercise P5: the DFT by its definition",
            "    def dft(x):",
            "        x = np.asarray(x, dtype=complex)",
            "        N = len(x)",
            "        n = np.arange(N)",
            "        k = n.reshape((N, 1))",
            "        return np.exp(-2j * np.pi * k * n / N) @ x",
            "Bin frequency: k * fs / N. Frequency resolution: fs / N. Dominant frequency of a real signal:",
            "the index of the largest value of np.abs(np.fft.rfft(x)) multiplied by fs / N.",
        ]),
        ("Module 4 - Python: quantization, decibels, filters", [
            "Solved exercise P6: uniform quantizer (mid-point of each step)",
            "    def quantize(x, bits, xmin=-1.0, xmax=1.0):",
            "        levels = 2 ** bits",
            "        step = (xmax - xmin) / levels",
            "        idx = np.clip(np.floor((np.asarray(x) - xmin) / step), 0, levels - 1)",
            "        return xmin + (idx + 0.5) * step",
            "SQNR in dB for a full-scale sine: 6.02 * bits + 1.76. Gain in decibels: 20 * np.log10(abs(gain)).",
            "SNR in dB: 10 * np.log10(P_signal / P_noise), with P the average power.",
            "Solved exercise P7: first-order IIR filter y[n] = x[n] + a y[n-1]",
            "    def first_order_iir(x, a):",
            "        y = np.zeros(len(x))",
            "        for n in range(len(x)):",
            "            y[n] = x[n] + (a * y[n - 1] if n > 0 else 0.0)",
            "        return y",
            "General difference equation with a[0] = 1: y[n] = sum_k b[k] x[n-k] - sum_{k>=1} a[k] y[n-k]",
            "(the same as scipy.signal.lfilter(b, a, x)). The impulse response a^n u[n] is stable when |a| < 1.",
        ]),
        ("Module 4 - Python: resampling, windows and other helpers", [
            "Solved exercise P8: down-sampling and up-sampling",
            "    def downsample(x, M):",
            "        return np.asarray(x)[::M]",
            "    def upsample(x, L):",
            "        y = np.zeros(len(x) * L)",
            "        y[::L] = x",
            "        return y",
            "Zero crossings: count the n where x[n-1] * x[n] < 0. Normalize to peak 1: x / np.max(np.abs(x)).",
            "Autocorrelation at lag l: the sum over n of x[n] x[n + l].",
            "Hann window: 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n = 0..N-1.",
            "Even part of a sequence centred at n = 0: (x + x[::-1]) / 2; odd part: (x - x[::-1]) / 2.",
            "RC cutoff: 1 / (2 * np.pi * R * C). FFT saving: N**2 / ((N / 2) * np.log2(N)).",
            "Moving-average magnitude response: |sin(pi f M / fs) / (M sin(pi f / fs))|, equal to 1 at f = 0.",
        ]),
    ],
}


def write_pdfs(out_dir: Path | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42  # real, extractable text
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_dir = out_dir or Path(__file__).with_name("course")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for title, pages in MODULES.items():
        path = out_dir / (title.replace(" ", "_") + ".pdf")
        with PdfPages(path) as pdf:
            for i, (heading, lines) in enumerate(pages, 1):
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.text(0.07, 0.95, heading, fontsize=13, weight="bold")
                y = 0.91
                for line in lines:
                    fig.text(0.07, y, line, fontsize=8.6, family="monospace" if line.startswith("    ") else "sans-serif")
                    y -= 0.026
                fig.text(0.07, 0.03, f"{title} - page {i}", fontsize=8, color="gray")
                pdf.savefig(fig)
                plt.close(fig)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in write_pdfs():
        print(p)
