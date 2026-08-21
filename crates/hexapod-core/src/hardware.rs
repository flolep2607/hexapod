//! Sizing the real thing: turn a simulated gait into a joint-torque
//! requirement, then into a shortlist of servos you can actually buy.
//!
//! # Torque model
//!
//! For a foot carrying a vertical load `F`, the static torque about a
//! horizontal joint is `F` times the *horizontal* distance from that joint to
//! the foot — the lever arm in the leg's working plane. That is the Jacobian
//! transpose specialised to a vertical force, and it is what actually sizes a
//! walking robot's servos. The coxa joint rotates about a vertical axis, so a
//! vertical load produces no moment on it; it is sized by the horizontal
//! traction force instead.
//!
//! Because the lever arm grows with stance width, a policy that learns to
//! stand wider for stability directly increases the torque bill. The hardware
//! tab exists to make that trade visible.
//!
//! # Prices
//!
//! Every price here is a recorded observation with a source and a date, not a
//! live quote. Marketplace bands (AliExpress and similar) and western
//! distributor prices are kept apart because they routinely differ by 3-5x for
//! the same part. Verify before ordering.

use crate::dynamics::{joint_torques, Actuator, LegMass, Physics};
use crate::robot::{fk_body, MAX_LEGS};
use crate::sim::Sim;

pub use crate::dynamics::NM_TO_KGCM;
use crate::dynamics::G;

/// Physical build the simulated gait is being mapped onto.
#[derive(Clone, Copy, Debug)]
pub struct Build {
    /// Linear scale from simulator units to metres. At 0.10 the 0.80-unit
    /// femur becomes an 80 mm link and the robot stands ~28 cm wide.
    pub scale: f64,
    /// All-up mass in kilograms, including servos, chassis and battery.
    pub mass_kg: f64,
    /// Multiplier covering gait transients, landing impact and the fact that
    /// load is never shared perfectly between the legs on the ground.
    pub dynamic_factor: f64,
    /// Fraction of the vertical foot load assumed to act horizontally
    /// (traction and lateral scuffing), which is what loads the coxa joint.
    pub traction_ratio: f64,
    /// Required headroom over peak demand before a servo counts as adequate.
    pub safety: f64,
}

impl Default for Build {
    fn default() -> Self {
        Build {
            scale: 0.10,
            // Same all-up mass as `Physics::default`; the two must agree or the
            // servo is being costed against a robot the simulator is not running.
            mass_kg: 2.3,
            dynamic_factor: 1.5,
            traction_ratio: 0.30,
            safety: 1.35,
        }
    }
}

impl Build {
    /// The physics this build implies, given the servo driving its joints.
    ///
    /// This is the join between the two halves of the project: the servo you
    /// are costing is the servo the simulator runs on, at this build's mass
    /// and scale.
    ///
    /// Both halves of it. A servo's torque-speed line and a servo's *weight*
    /// are the same choice, and the legs are half the machine — taking the
    /// line but leaving the default leg mass simulated a 9 g SG90 swinging
    /// links built around a 74.5 g one, which loaded it with inertia it would
    /// never carry and made the comparison between catalogue entries unfair in
    /// the heavy servo's favour.
    pub fn physics(&self, servo: Option<&Servo>) -> Physics {
        Physics {
            mass_kg: self.mass_kg,
            scale: self.scale,
            dynamic: self.dynamic_factor,
            actuator: match servo {
                Some(s) => s.actuator(),
                None => Actuator::default(),
            },
            leg: match servo {
                Some(s) => LegMass::from_servo(s.mass_g / 1000.0),
                None => Physics::default().leg,
            },
            ..Physics::default()
        }
    }

    /// Femur, tibia and coxa lengths in millimetres.
    pub fn link_mm(&self) -> [f64; 3] {
        [
            crate::robot::COXA * self.scale * 1000.0,
            crate::robot::FEMUR * self.scale * 1000.0,
            crate::robot::TIBIA * self.scale * 1000.0,
        ]
    }
}

/// Peak joint torques observed over a gait, in newton-metres.
#[derive(Clone, Copy, Debug, Default)]
pub struct TorqueMeter {
    /// Peak per joint: coxa, femur, tibia.
    pub peak: [f64; 3],
    /// Peak per leg, worst joint.
    pub peak_leg: [f64; MAX_LEGS],
    /// Running mean of the femur joint, the usual sizing case.
    pub mean_femur: f64,
    pub samples: f64,
    /// Largest single-foot load seen, newtons.
    pub peak_foot_load: f64,
}

impl TorqueMeter {
    /// Fold one simulator step into the running peaks.
    pub fn observe(&mut self, sim: &Sim, build: &Build) {
        let weight = build.mass_kg * G;
        self.samples += 1.0;

        for leg in 0..sim.frame.legs() {
            if !sim.feet[leg].stance {
                continue;
            }

            // Vertical load on this foot, with the transient allowance.
            let f_v = weight * sim.feet[leg].load * build.dynamic_factor;
            let f_h = f_v * build.traction_ratio;
            if f_v > self.peak_foot_load {
                self.peak_foot_load = f_v;
            }

            // Lever arms come from the pose, in simulator units, then scale.
            // Same formula the simulator drives the servos with.
            let j = fk_body(sim.frame, leg, sim.q[leg]);
            let t = joint_torques(&j, f_v, f_h, build.scale);

            for k in 0..3 {
                if t[k] > self.peak[k] {
                    self.peak[k] = t[k];
                }
            }
            let worst = t[0].max(t[1]).max(t[2]);
            if worst > self.peak_leg[leg] {
                self.peak_leg[leg] = worst;
            }
            self.mean_femur += (t[1] - self.mean_femur) / self.samples;
        }
    }

    /// Worst joint torque anywhere in the robot, newton-metres.
    pub fn worst(&self) -> f64 {
        self.peak[0].max(self.peak[1]).max(self.peak[2])
    }

    /// What a servo must be rated for, in kg-cm, including the safety factor.
    pub fn required_kgcm(&self, build: &Build) -> f64 {
        self.worst() * NM_TO_KGCM * build.safety
    }

    pub fn peak_kgcm(&self) -> [f64; 3] {
        [
            self.peak[0] * NM_TO_KGCM,
            self.peak[1] * NM_TO_KGCM,
            self.peak[2] * NM_TO_KGCM,
        ]
    }
}

/// How a price was obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// Read off a named vendor's product page on `checked`.
    Vendor,
    /// Typical street price band across marketplace listings. Indicative only.
    Marketplace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bus {
    /// Standard 50 Hz PWM. One wire per servo, no feedback.
    Pwm,
    /// Addressable serial bus, daisy-chained, with position/load feedback.
    Serial,
}

impl Bus {
    pub fn name(self) -> &'static str {
        match self {
            Bus::Pwm => "PWM",
            Bus::Serial => "SERIAL BUS",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Servo {
    pub part: &'static str,
    pub maker: &'static str,
    /// Stall torque in kg-cm at `at_volts`.
    pub stall_kgcm: f64,
    pub at_volts: f64,
    /// No-load speed in seconds per 60 degrees, at `at_volts`. With
    /// `stall_kgcm` this fixes the servo's torque-speed line, which is what
    /// the simulator drives the joints with. Carries the same provenance as
    /// the electrical figures: where `elec_checked` is false it is a
    /// commonly-repeated number, not one read off a datasheet.
    pub speed_s60: f64,
    pub mass_g: f64,
    /// Stall current at `at_volts`, amps.
    pub stall_amps: f64,
    /// Running current with no load, amps.
    pub noload_amps: f64,
    /// Quiescent draw while merely holding position, amps.
    pub idle_amps: f64,
    /// True when the electrical figures came from a manufacturer datasheet
    /// rather than commonly-repeated community values.
    pub elec_checked: bool,
    pub bus: Bus,
    pub metal_gear: bool,
    /// Position/load feedback back to the controller.
    pub feedback: bool,
    /// Low and high of the typical marketplace band, USD per unit.
    pub market_usd: (f64, f64),
    /// Named-distributor price, USD per unit, when one was checked.
    pub vendor_usd: Option<f64>,
    pub vendor_name: &'static str,
    pub source: &'static str,
    pub provenance: Provenance,
    pub note: &'static str,
}

impl Servo {
    /// Cheapest defensible unit price: the marketplace low if there is a band,
    /// otherwise the distributor price.
    pub fn unit_low(&self) -> f64 {
        self.market_usd.0
    }

    pub fn unit_high(&self) -> f64 {
        self.vendor_usd.unwrap_or(self.market_usd.1).max(self.market_usd.1)
    }

    /// Cost of a full set: three servos per leg.
    pub fn build_cost(&self, joints: usize) -> (f64, f64) {
        let n = joints as f64;
        (self.unit_low() * n, self.unit_high() * n)
    }

    /// Total servo mass for a full set, kilograms.
    pub fn set_mass_kg(&self, joints: usize) -> f64 {
        self.mass_g * joints as f64 / 1000.0
    }

    /// Current drawn while holding `tau` newton-metres, given the servo's
    /// stall torque in the same units.
    ///
    /// A brushed DC motor's torque is proportional to its current, so this is
    /// linear between the idle and stall points. It ignores gearbox friction
    /// and reversal inrush, so treat it as a floor.
    #[inline]
    pub fn amps_at(&self, tau_nm: f64, stall_nm: f64) -> f64 {
        if stall_nm <= 0.0 {
            return self.idle_amps;
        }
        let frac = crate::math::clamp(tau_nm / stall_nm, 0.0, 1.0);
        self.idle_amps + (self.stall_amps - self.idle_amps) * frac
    }

    /// Watts at the servo bus while holding `tau`.
    #[inline]
    pub fn watts_at(&self, tau_nm: f64, stall_nm: f64) -> f64 {
        self.amps_at(tau_nm, stall_nm) * self.at_volts
    }

    /// The joint model this servo implies: its torque-speed line.
    pub fn actuator(&self) -> Actuator {
        Actuator::from_rating(self.speed_s60, self.stall_kgcm)
    }

    pub fn meets(&self, required_kgcm: f64) -> bool {
        self.stall_kgcm >= required_kgcm
    }

    /// Stall torque divided by requirement. Above 1.0 clears it.
    pub fn headroom(&self, required_kgcm: f64) -> f64 {
        if required_kgcm <= 0.0 {
            return f64::INFINITY;
        }
        self.stall_kgcm / required_kgcm
    }
}

/// Date the vendor prices below were read, ISO 8601.
pub const PRICES_CHECKED: &str = "2026-08-16";

/// Servos commonly used to build hexapods, cheapest first.
///
/// Torque figures are manufacturer stall ratings — real continuous torque is
/// far lower, which is what the safety factor in [`Build`] is for.
pub const SERVOS: &[Servo] = &[
    Servo {
        part: "SG90",
        maker: "TowerPro",
        stall_kgcm: 1.8,
        at_volts: 4.8,
        speed_s60: 0.1,
        mass_g: 9.0,
        stall_amps: 0.7,
        noload_amps: 0.22,
        idle_amps: 0.006,
        elec_checked: false,
        bus: Bus::Pwm,
        metal_gear: false,
        feedback: false,
        market_usd: (0.80, 2.50),
        vendor_usd: None,
        vendor_name: "",
        source: "https://www.aliexpress.com/w/wholesale-sg90-servo.html",
        provenance: Provenance::Marketplace,
        note: "Plastic gears strip under side load. Micro builds only.",
    },
    Servo {
        part: "MG90S",
        maker: "TowerPro",
        stall_kgcm: 2.2,
        at_volts: 6.0,
        speed_s60: 0.08,
        mass_g: 13.4,
        stall_amps: 0.75,
        noload_amps: 0.25,
        idle_amps: 0.008,
        elec_checked: false,
        bus: Bus::Pwm,
        metal_gear: true,
        feedback: false,
        market_usd: (1.50, 3.50),
        vendor_usd: None,
        vendor_name: "",
        source: "https://www.aliexpress.com/w/wholesale-mg90s-servo.html",
        provenance: Provenance::Marketplace,
        note: "Metal-gear SG90 upgrade. Fine for sub-kilogram machines.",
    },
    Servo {
        part: "MG995",
        maker: "TowerPro",
        stall_kgcm: 10.0,
        at_volts: 6.0,
        speed_s60: 0.16,
        mass_g: 62.4,
        stall_amps: 1.2,
        noload_amps: 0.35,
        idle_amps: 0.01,
        elec_checked: false,
        bus: Bus::Pwm,
        metal_gear: true,
        feedback: false,
        market_usd: (3.00, 6.00),
        vendor_usd: Some(19.95),
        vendor_name: "Adafruit",
        source: "https://www.adafruit.com/product/1142",
        provenance: Provenance::Vendor,
        note: "Coarse deadband; noticeably jittery holding a pose.",
    },
    Servo {
        part: "MG996R",
        maker: "TowerPro",
        stall_kgcm: 11.0,
        at_volts: 6.0,
        speed_s60: 0.14,
        mass_g: 55.0,
        stall_amps: 2.5,
        noload_amps: 0.17,
        idle_amps: 0.01,
        elec_checked: true,
        bus: Bus::Pwm,
        metal_gear: true,
        feedback: false,
        market_usd: (3.00, 6.50),
        vendor_usd: Some(10.95),
        vendor_name: "JSumo",
        source: "https://www.jsumo.com/mg996r-servo-motor-digital",
        provenance: Provenance::Vendor,
        note: "The default hexapod servo. Clone quality varies a lot.",
    },
    Servo {
        part: "DS3218MG",
        maker: "DSServo",
        stall_kgcm: 20.0,
        at_volts: 6.8,
        speed_s60: 0.16,
        mass_g: 60.0,
        stall_amps: 2.5,
        noload_amps: 0.35,
        idle_amps: 0.01,
        elec_checked: false,
        bus: Bus::Pwm,
        metal_gear: true,
        feedback: false,
        market_usd: (8.00, 18.00),
        vendor_usd: None,
        vendor_name: "",
        source: "https://www.amazon.com/ANNIMOS-Digital-Waterproof-DS3218MG-Control/dp/B076CNKQX4",
        provenance: Provenance::Marketplace,
        note: "IP66 digital. Big torque per dollar, but a current hog.",
    },
    Servo {
        part: "LX-16A",
        maker: "Hiwonder",
        stall_kgcm: 17.0,
        at_volts: 6.0,
        speed_s60: 0.22,
        mass_g: 52.0,
        stall_amps: 1.5,
        noload_amps: 0.1,
        idle_amps: 0.01,
        elec_checked: false,
        bus: Bus::Serial,
        metal_gear: true,
        feedback: true,
        market_usd: (16.00, 22.00),
        vendor_usd: None,
        vendor_name: "",
        source: "https://www.hiwonder.com/products/lx-16a",
        provenance: Provenance::Marketplace,
        note: "Daisy-chains; position and temperature feedback. 19.5 kg-cm at 7.4 V.",
    },
    Servo {
        part: "STS3215",
        maker: "Feetech",
        stall_kgcm: 19.5,
        at_volts: 7.4,
        speed_s60: 0.24,
        mass_g: 60.0,
        stall_amps: 2.7,
        noload_amps: 0.18,
        idle_amps: 0.01,
        elec_checked: false,
        bus: Bus::Serial,
        metal_gear: true,
        feedback: true,
        market_usd: (18.00, 24.00),
        vendor_usd: Some(21.99),
        vendor_name: "Seeed Studio",
        source: "https://www.seeedstudio.com/STS3215-19kg-cm-7-4V-Serial-Servo-p-6338.html",
        provenance: Provenance::Vendor,
        note: "12-bit magnetic encoder, 1:345 metal box. $20.99 each at 10+.",
    },
    Servo {
        part: "AX-12A",
        maker: "Robotis Dynamixel",
        stall_kgcm: 15.3,
        at_volts: 12.0,
        speed_s60: 0.17,
        mass_g: 54.6,
        stall_amps: 1.5,
        noload_amps: 0.06,
        idle_amps: 0.05,
        elec_checked: false,
        bus: Bus::Serial,
        metal_gear: true,
        feedback: true,
        market_usd: (45.00, 60.00),
        vendor_usd: Some(57.39),
        vendor_name: "Robotis",
        source: "https://www.robotis.us/dynamixel-ax-12a/",
        provenance: Provenance::Vendor,
        note: "What the PhantomX hexapod runs. Rated (not stall) torque is ~0.2 N-m.",
    },
    Servo {
        part: "STS3250",
        maker: "Feetech",
        // Stall, which is where the torque-speed line crosses zero speed and so
        // the right number for `actuator()`. It is NOT what the joint can hold:
        // a bench run measured 48 kg-cm for a split second, 25 kg-cm sustained
        // before the overload trip, against a manufacturer rated figure of 16.
        // Feetech trips over-current above 4.85 A for 2 s, overload above 80% of
        // stall for 2.5 s, and disables torque at 70 C -- which a 40% load
        // reaches in eight minutes at 3.75 C/min. Size a walk on 25 kg-cm.
        stall_kgcm: 50.0,
        at_volts: 12.0,
        speed_s60: 0.133,
        mass_g: 74.5,
        stall_amps: 4.2,
        noload_amps: 0.28,
        // Not published; every other figure here is off the manufacturer table.
        idle_amps: 0.01,
        elec_checked: true,
        bus: Bus::Serial,
        metal_gear: true,
        feedback: true,
        market_usd: (45.50, 48.70),
        vendor_usd: Some(45.50),
        vendor_name: "Shenzhen Feite on Alibaba",
        source: "https://www.alibaba.com/product-detail/FEETECH-STS3250-12V-50KG-Double-Shaft_1601527075848.html",
        provenance: Provenance::Vendor,
        note: "ST-3250-C001. 12-bit encoder, 1:345, 45.2x24.7x35 mm, 25T, 0.43 deg \
               backlash, 91 N-m/rad torsional. 18 stalling together draw 75.6 A at \
               12 V; the rated-current budget is 25.2 A. C001 is coreless, C002 \
               carbon-brush -- not the same part. Feetech spec 2024-01-16 ed. A/0.",
    },
];

/// Servos that clear `required_kgcm`, cheapest set first.
pub fn shortlist(required_kgcm: f64) -> Vec<&'static Servo> {
    let mut v: Vec<&Servo> = SERVOS.iter().filter(|s| s.meets(required_kgcm)).collect();
    // Set size is the same for every candidate, so ordering by unit price
    // orders by set price.
    v.sort_by(|a, b| {
        a.unit_low()
            .partial_cmp(&b.unit_low())
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Policy, Preset};
    use crate::sim::{Cmd, DT};
    use crate::terrain::{Course, Terrain};

    fn measure(build: Build) -> TorqueMeter {
        let terrain = Terrain::new(Course::Flat, 1);
        let p = Policy::seeded(Preset::Tripod, crate::robot::Frame::default());
        let g = p.gait();
        // Ideal joints, so the trajectory is identical for every build and
        // the only thing varying between cases is the sizing arithmetic.
        let phys = crate::dynamics::Physics::ideal();
        let mut s = Sim::default();
        s.reset(&terrain, &g, &phys);
        let mut m = TorqueMeter::default();
        for _ in 0..600 {
            s.step(&terrain, &p, &g, DT, Cmd::at(g.nominal_speed()));
            m.observe(&s, &build);
        }
        m
    }

    #[test]
    fn torque_lands_in_a_plausible_range_for_a_2kg_machine() {
        let m = measure(Build::default());
        let req = m.required_kgcm(&Build::default());
        // A 2 kg, 28 cm hexapod is squarely MG996R/DS3218 territory.
        assert!(
            (3.0..30.0).contains(&req),
            "required torque {req:.1} kg-cm is not physically sensible"
        );
        // The femur joint should be the sizing case, not the coxa.
        let k = m.peak_kgcm();
        assert!(k[1] > k[0], "femur {:.2} should exceed coxa {:.2}", k[1], k[0]);
    }

    #[test]
    fn torque_scales_linearly_with_mass() {
        let a = measure(Build {
            mass_kg: 2.0,
            ..Build::default()
        });
        let b = measure(Build {
            mass_kg: 4.0,
            ..Build::default()
        });
        let ratio = b.worst() / a.worst();
        assert!((ratio - 2.0).abs() < 1e-6, "ratio {ratio}");
    }

    #[test]
    fn torque_scales_linearly_with_size() {
        let a = measure(Build {
            scale: 0.10,
            ..Build::default()
        });
        let b = measure(Build {
            scale: 0.20,
            ..Build::default()
        });
        let ratio = b.worst() / a.worst();
        assert!((ratio - 2.0).abs() < 1e-6, "ratio {ratio}");
    }

    #[test]
    fn shortlist_only_returns_servos_that_clear_the_requirement() {
        let need = 12.0;
        for s in shortlist(need) {
            assert!(s.stall_kgcm >= need, "{} does not meet {need}", s.part);
            assert!(s.headroom(need) >= 1.0);
        }
        // Micro servos must be excluded at this torque.
        assert!(!shortlist(need).iter().any(|s| s.part == "SG90"));
        // And an impossible requirement returns nothing rather than guessing.
        assert!(shortlist(500.0).is_empty());
    }

    #[test]
    fn shortlist_is_ordered_by_build_cost() {
        let v = shortlist(10.0);
        for w in v.windows(2) {
            assert!(w[0].build_cost(18).0 <= w[1].build_cost(18).0);
        }
    }

    #[test]
    fn catalogue_entries_are_internally_consistent() {
        for s in SERVOS {
            assert!(s.market_usd.0 > 0.0 && s.market_usd.1 >= s.market_usd.0, "{}", s.part);
            assert!(s.stall_kgcm > 0.0 && s.mass_g > 0.0, "{}", s.part);
            assert!(
                s.stall_amps > s.noload_amps && s.noload_amps > s.idle_amps,
                "{} has an incoherent current curve",
                s.part
            );
            assert!(s.source.starts_with("https://"), "{} has no source", s.part);
            // A vendor-checked price must name its vendor, and vice versa.
            assert_eq!(
                s.vendor_usd.is_some(),
                s.provenance == Provenance::Vendor,
                "{} provenance disagrees with its price",
                s.part
            );
            if s.vendor_usd.is_some() {
                assert!(!s.vendor_name.is_empty(), "{} has no vendor name", s.part);
            }
        }
    }

    #[test]
    fn a_set_of_servos_weighs_a_real_fraction_of_the_robot() {
        let mg996r = SERVOS.iter().find(|s| s.part == "MG996R").unwrap();
        // 18 x 55 g is about a kilogram: half the default 2 kg budget.
        assert!((mg996r.set_mass_kg(18) - 0.99).abs() < 0.01);
        // Three servos per leg, so an octopod's set weighs a third more.
        assert!((mg996r.set_mass_kg(24) / mg996r.set_mass_kg(18) - 4.0 / 3.0).abs() < 1e-9);
    }
}
