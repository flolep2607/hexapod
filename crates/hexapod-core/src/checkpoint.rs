//! Text format for a trained policy — the file `train-all` writes and both
//! `eval-all` and the browser read.
//!
//! It lives here rather than in the CLI because the dashboard loads the same
//! checkpoints: one parser, so a file that evaluates natively cannot be
//! rejected or, worse, silently misread in the page. No serde — the format is
//! eight lines of `key=values`, and pulling a derive machinery into a wasm
//! module to read it would cost more than it saves.

use crate::policy::{n_theta, Normalizer, Policy, MAX_OBS};
use crate::robot::{Frame, MAX_LEGS};

pub const MAGIC: &str = "hexapod-policy-v1";

/// Serialise at full `f64` precision: a checkpoint that does not reproduce its
/// evaluation is not a checkpoint.
pub fn to_text(policy: &Policy) -> String {
    let values = |xs: &[f64]| {
        xs.iter()
            .map(|v| format!("{v:.17e}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    format!(
        "{MAGIC}\nframe={}\nfeedback={:.17e}\nbase_offsets={}\nnorm_n={:.17e}\nnorm_mean={}\nnorm_m2={}\ntheta={}\n",
        policy.frame.legs(),
        policy.feedback,
        values(&policy.base_offsets),
        policy.norm.n,
        values(&policy.norm.mean),
        values(&policy.norm.m2),
        values(&policy.theta),
    )
}

/// Read the leg count alone, without validating the rest. The dashboard needs
/// it before loading, because adopting a different frame is a different
/// machine and throws away everything else.
pub fn frame_of(text: &str) -> Option<Frame> {
    let legs = text
        .lines()
        .skip(1)
        .find_map(|line| line.strip_prefix("frame="))?
        .trim()
        .parse::<usize>()
        .ok()?;
    let frame = Frame::new(legs);
    (frame.legs() == legs).then_some(frame)
}

pub fn from_text(text: &str) -> Result<Policy, String> {
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some(MAGIC) {
        return Err(format!("not a {MAGIC} checkpoint"));
    }
    let mut fields = std::collections::HashMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("invalid checkpoint line {line:?}"));
        };
        fields.insert(key.trim(), value);
    }
    let field = |name: &str| {
        fields
            .get(name)
            .copied()
            .ok_or_else(|| format!("checkpoint is missing {name}"))
    };
    let scalar = |name: &str| -> Result<f64, String> {
        field(name)?
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("invalid {name}"))
    };
    let numbers = |name: &str| -> Result<Vec<f64>, String> {
        field(name)?
            .split_whitespace()
            .map(|v| {
                v.parse::<f64>()
                    .map_err(|_| format!("invalid number {v:?} in {name}"))
            })
            .collect()
    };
    let legs = field("frame")?
        .trim()
        .parse::<usize>()
        .map_err(|_| "invalid frame".to_string())?;
    let frame = Frame::new(legs);
    if frame.legs() != legs {
        return Err(format!("unsupported frame with {legs} legs"));
    }
    let feedback = scalar("feedback")?;
    let norm_n = scalar("norm_n")?;
    let offsets = numbers("base_offsets")?;
    let means = numbers("norm_mean")?;
    let m2 = numbers("norm_m2")?;
    let theta = numbers("theta")?;
    if offsets.len() != MAX_LEGS {
        return Err(format!("expected {MAX_LEGS} base offsets"));
    }
    if means.len() != MAX_OBS || m2.len() != MAX_OBS {
        return Err(format!("expected {MAX_OBS} normalizer entries"));
    }
    if theta.len() != n_theta(frame) {
        return Err(format!(
            "expected {} parameters for {} legs, found {}",
            n_theta(frame),
            legs,
            theta.len()
        ));
    }
    if !offsets
        .iter()
        .chain(&means)
        .chain(&m2)
        .chain(&theta)
        .chain([feedback, norm_n].iter())
        .all(|v| v.is_finite())
    {
        return Err("checkpoint contains NaN or infinity".to_string());
    }
    let mut base_offsets = [0.0; MAX_LEGS];
    base_offsets.copy_from_slice(&offsets);
    let mut mean = [0.0; MAX_OBS];
    mean.copy_from_slice(&means);
    let mut norm_m2 = [0.0; MAX_OBS];
    norm_m2.copy_from_slice(&m2);
    Ok(Policy {
        frame,
        theta,
        base_offsets,
        norm: Normalizer {
            n: norm_n,
            mean,
            m2: norm_m2,
            frozen: true,
        },
        feedback,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::Preset;

    fn sample() -> Policy {
        let frame = Frame::new(6);
        let mut p = Policy::seeded(Preset::default_for(frame), frame);
        p.feedback = 0.75;
        p.norm.n = 1234.5;
        p.norm.mean[3] = -0.125;
        p.theta[7] = 0.5;
        p
    }

    #[test]
    fn a_checkpoint_round_trips_exactly() {
        let p = sample();
        let back = from_text(&to_text(&p)).expect("round trip");
        assert_eq!(back.frame, p.frame);
        assert_eq!(back.theta, p.theta);
        assert_eq!(back.base_offsets, p.base_offsets);
        assert_eq!(back.norm.mean, p.norm.mean);
        assert_eq!(back.norm.m2, p.norm.m2);
        assert_eq!(back.norm.n, p.norm.n);
        assert_eq!(back.feedback, p.feedback);
        assert!(
            back.norm.frozen,
            "a loaded normaliser is evidence, not state"
        );
    }

    #[test]
    fn the_frame_is_readable_without_a_full_parse() {
        assert_eq!(frame_of(&to_text(&sample())), Some(Frame::new(6)));
        assert_eq!(frame_of("nonsense"), None);
    }

    #[test]
    fn junk_is_rejected_rather_than_half_read() {
        assert!(from_text("hello\nframe=6\n").is_err());
        let text = to_text(&sample()).replace("frame=6", "frame=7");
        assert!(from_text(&text).is_err(), "odd leg counts do not exist");
        let text = to_text(&sample()).replace("feedback=", "feedback_x=");
        let Err(error) = from_text(&text) else {
            panic!("a checkpoint without a feedback scale is not one");
        };
        assert!(error.contains("missing feedback"), "{error}");
    }
}
