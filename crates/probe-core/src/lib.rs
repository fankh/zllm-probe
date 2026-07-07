//! Engine-agnostic LLM output-distribution instrumentation.
//!
//! Per-token uncertainty signals (predictive entropy, top-token probability,
//! top-1/top-2 margin, chosen-token log-prob) aggregated into a response-level
//! hallucination-risk report. Two input modes:
//!
//! - [`Detector::observe_logits`] — full logit vector (in-process adapters
//!   with engine access: zllm native, vLLM logits-processor).
//! - [`Detector::observe_top_logprobs`] — the `logprobs`/`top_logprobs` field
//!   every OpenAI-compatible server returns (llama.cpp `llama-server`, vLLM,
//!   zllm behind the proxy). Entropy is then an **approximation**: exact over
//!   the observed top-K mass, with the unobserved tail folded into a single
//!   outcome (a lower bound on true entropy; documented per field).
//!
//! Honest framing: these are **uncertainty proxies**, not calibrated
//! hallucination oracles. High uncertainty correlates with confabulation;
//! treat `risk_score` as a relative flag and `peak_token` as the pointer to
//! the risky span. (Origin: the zllm engine's live-verified detector —
//! factual prompt 0.32 unflagged vs confabulated-fact 0.69 flagged on
//! Llama-3.2-1B.)

use serde::Serialize;

/// Per-generated-token uncertainty.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct TokenUncertainty {
    /// Predictive entropy in nats. Exact for `observe_logits`; a lower-bound
    /// approximation for `observe_top_logprobs` (tail folded to one outcome).
    pub entropy: f32,
    /// Probability of the most likely token.
    pub max_prob: f32,
    /// `p(top1) − p(top2)`.
    pub margin: f32,
    /// `ln p(chosen)` — log-prob of the token actually emitted.
    pub chosen_logprob: f32,
}

impl TokenUncertainty {
    /// Per-token risk in `[0,1]`: probability mass the model put on something
    /// other than its top choice.
    pub fn risk(&self) -> f32 {
        (1.0 - self.max_prob).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DetectorConfig {
    /// A token is counted "risky" when its entropy exceeds this (nats)…
    pub entropy_threshold: f32,
    /// …or when its top-token probability falls below this.
    pub max_prob_threshold: f32,
    /// Response-level `flagged` bar on the mean risk.
    pub risk_flag_threshold: f32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        // ~2.0 nats ≈ choosing among ~7+ tokens; <40% top mass. Conservative.
        Self { entropy_threshold: 2.0, max_prob_threshold: 0.4, risk_flag_threshold: 0.5 }
    }
}

/// Response-level summary plus per-token detail.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// Mean per-token `risk()` in `[0,1]`.
    pub risk_score: f32,
    /// Mean predictive entropy (nats). Approximate in top-logprobs mode.
    pub mean_entropy: f32,
    /// Fraction of tokens that tripped a threshold.
    pub risky_fraction: f32,
    /// Index of the single most uncertain token.
    pub peak_token: Option<usize>,
    pub n_tokens: usize,
    /// `risk_score >= risk_flag_threshold`.
    pub flagged: bool,
    /// `"logits"` (exact) or `"top_logprobs"` (approximate entropy).
    pub mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_token: Option<Vec<TokenUncertainty>>,
}

/// Accumulates per-token uncertainty across one generation.
pub struct Detector {
    cfg: DetectorConfig,
    tokens: Vec<TokenUncertainty>,
    risky: usize,
    approx: bool,
}

impl Detector {
    pub fn new(cfg: DetectorConfig) -> Self {
        Self { cfg, tokens: Vec::new(), risky: 0, approx: false }
    }

    /// Feed one decode step's full logits and the emitted token id.
    pub fn observe_logits(&mut self, logits: &[f32], chosen: usize) {
        let u = uncertainty_from_logits(logits, chosen);
        self.push(u);
    }

    /// Feed one decode step's OpenAI-style logprobs: the chosen token's
    /// logprob and the `top_logprobs` values (natural log, descending or not).
    pub fn observe_top_logprobs(&mut self, chosen_logprob: f32, top_logprobs: &[f32]) {
        let u = uncertainty_from_top_logprobs(chosen_logprob, top_logprobs);
        self.approx = true;
        self.push(u);
    }

    fn push(&mut self, u: TokenUncertainty) {
        if u.entropy > self.cfg.entropy_threshold || u.max_prob < self.cfg.max_prob_threshold {
            self.risky += 1;
        }
        self.tokens.push(u);
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Produce the report. `include_per_token` controls the detail payload.
    pub fn report(self, include_per_token: bool) -> Report {
        let n = self.tokens.len();
        if n == 0 {
            return Report {
                risk_score: 0.0, mean_entropy: 0.0, risky_fraction: 0.0,
                peak_token: None, n_tokens: 0, flagged: false,
                mode: if self.approx { "top_logprobs" } else { "logits" },
                per_token: include_per_token.then(Vec::new),
            };
        }
        let mean_entropy = self.tokens.iter().map(|t| t.entropy).sum::<f32>() / n as f32;
        let risk_score = self.tokens.iter().map(|t| t.risk()).sum::<f32>() / n as f32;
        let peak_token = self.tokens.iter().enumerate()
            .max_by(|a, b| a.1.entropy.total_cmp(&b.1.entropy))
            .map(|(i, _)| i);
        Report {
            risk_score,
            mean_entropy,
            risky_fraction: self.risky as f32 / n as f32,
            peak_token,
            n_tokens: n,
            flagged: risk_score >= self.cfg.risk_flag_threshold,
            mode: if self.approx { "top_logprobs" } else { "logits" },
            per_token: include_per_token.then_some(self.tokens),
        }
    }
}

/// Exact entropy/max/margin from one full logit vector (max-subtracted
/// softmax; numerically stable).
pub fn uncertainty_from_logits(logits: &[f32], chosen: usize) -> TokenUncertainty {
    if logits.is_empty() {
        return TokenUncertainty { entropy: 0.0, max_prob: 0.0, margin: 0.0, chosen_logprob: f32::NEG_INFINITY };
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &l in logits {
        sum += (l - max).exp();
    }
    let inv = 1.0 / sum;
    let ln_sum = sum.ln();
    let mut entropy = 0.0f32;
    let (mut top1, mut top2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &l in logits {
        let p = (l - max).exp() * inv;
        if p > 0.0 {
            entropy -= p * ((l - max) - ln_sum);
        }
        if p > top1 {
            top2 = top1;
            top1 = p;
        } else if p > top2 {
            top2 = p;
        }
    }
    let chosen_p = logits.get(chosen).map(|&l| (l - max).exp() * inv).unwrap_or(0.0);
    TokenUncertainty {
        entropy: entropy.max(0.0),
        max_prob: top1.max(0.0),
        margin: (top1 - top2).max(0.0),
        chosen_logprob: if chosen_p > 0.0 { chosen_p.ln() } else { f32::NEG_INFINITY },
    }
}

/// Approximate entropy/max/margin from OpenAI-style top-K logprobs. The
/// unobserved tail mass `1 − Σ p_i` is folded into a single outcome —
/// a LOWER BOUND on the true entropy (the tail's internal spread is unseen).
pub fn uncertainty_from_top_logprobs(chosen_logprob: f32, top_logprobs: &[f32]) -> TokenUncertainty {
    if top_logprobs.is_empty() {
        // Only the chosen token's logprob available: max_prob >= p(chosen) is
        // unknown; use p(chosen) as the (under)estimate of top mass.
        let p = chosen_logprob.exp().clamp(0.0, 1.0);
        let tail = (1.0 - p).max(0.0);
        let mut entropy = 0.0;
        if p > 0.0 { entropy -= p * p.ln(); }
        if tail > 0.0 { entropy -= tail * tail.ln(); }
        return TokenUncertainty { entropy, max_prob: p, margin: 0.0, chosen_logprob };
    }
    let mut probs: Vec<f32> = top_logprobs.iter().map(|lp| lp.exp()).collect();
    probs.sort_by(|a, b| b.total_cmp(a));
    let observed: f32 = probs.iter().sum::<f32>().min(1.0);
    let mut entropy = 0.0f32;
    for &p in &probs {
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    let tail = (1.0 - observed).max(0.0);
    if tail > 1e-6 {
        entropy -= tail * tail.ln();
    }
    let top1 = probs[0];
    let top2 = probs.get(1).copied().unwrap_or(0.0);
    TokenUncertainty {
        entropy: entropy.max(0.0),
        max_prob: top1.clamp(0.0, 1.0),
        margin: (top1 - top2).max(0.0),
        chosen_logprob,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn logits_one_hot_is_certain() {
        let mut l = vec![0.0f32; 64];
        l[7] = 50.0;
        let u = uncertainty_from_logits(&l, 7);
        assert!(u.entropy < 1e-3);
        assert!(approx_eq(u.max_prob, 1.0, 1e-3));
        assert!(u.risk() < 1e-3);
    }

    #[test]
    fn logits_uniform_is_maximally_uncertain() {
        let v = 64usize;
        let l = vec![0.0f32; v];
        let u = uncertainty_from_logits(&l, 0);
        assert!(approx_eq(u.entropy, (v as f32).ln(), 1e-3));
        assert!(approx_eq(u.max_prob, 1.0 / v as f32, 1e-4));
    }

    #[test]
    fn top_logprobs_matches_exact_when_mass_covered() {
        // A distribution fully described by its top-4: exact and approx agree.
        let probs = [0.7f32, 0.2, 0.07, 0.03];
        let lps: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let ua = uncertainty_from_top_logprobs(lps[0], &lps);
        let exact: f32 = probs.iter().map(|p| -p * p.ln()).sum();
        assert!(approx_eq(ua.entropy, exact, 1e-3), "{} vs {exact}", ua.entropy);
        assert!(approx_eq(ua.max_prob, 0.7, 1e-4));
        assert!(approx_eq(ua.margin, 0.5, 1e-4));
    }

    #[test]
    fn top_logprobs_tail_is_lower_bound() {
        // Top-2 of a wide distribution: approx entropy must be <= true entropy
        // of (0.4, 0.3, uniform 30% tail over many tokens).
        let lps = [0.4f32.ln(), 0.3f32.ln()];
        let ua = uncertainty_from_top_logprobs(lps[0], &lps);
        let true_wide: f32 = -(0.4f32 * 0.4f32.ln() + 0.3 * 0.3f32.ln())
            + 0.3 * (100.0f32).ln() - 0.3 * 0.3f32.ln(); // tail spread over 100
        assert!(ua.entropy <= true_wide);
        assert!(ua.entropy > 0.5); // and still clearly nonzero
    }

    #[test]
    fn detector_flags_uncertain_generations() {
        let mut d = Detector::new(DetectorConfig::default());
        let mut sharp = vec![0.0f32; 32];
        sharp[1] = 40.0;
        d.observe_logits(&sharp, 1);
        d.observe_logits(&vec![0.0f32; 32], 0); // uniform → risky
        let r = d.report(true);
        assert_eq!(r.n_tokens, 2);
        assert_eq!(r.peak_token, Some(1));
        assert!(approx_eq(r.risky_fraction, 0.5, 1e-5));
        assert_eq!(r.mode, "logits");
        assert!(r.per_token.is_some());
    }

    #[test]
    fn detector_top_logprobs_mode_marked_approximate() {
        let mut d = Detector::new(DetectorConfig::default());
        d.observe_top_logprobs(0.9f32.ln(), &[0.9f32.ln(), 0.05f32.ln()]);
        let r = d.report(false);
        assert_eq!(r.mode, "top_logprobs");
        assert!(!r.flagged);
        assert!(r.per_token.is_none());
    }

    #[test]
    fn empty_report_is_zero() {
        let d = Detector::new(DetectorConfig::default());
        let r = d.report(false);
        assert_eq!(r.n_tokens, 0);
        assert!(!r.flagged);
    }
}
