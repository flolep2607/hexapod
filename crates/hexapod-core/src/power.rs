//! From a gait to a complete parts list: current draw, battery, regulator,
//! controller and sensors.
//!
//! # The loop that actually matters
//!
//! Battery mass is part of all-up mass, all-up mass sets joint torque, torque
//! sets current, and current sets the battery you need. That is a fixed point,
//! and it does not always have one: past a certain runtime a build cannot
//! carry the battery it needs to achieve that runtime. [`solve`] iterates it
//! and reports honestly when it diverges.
//!
//! # Current model
//!
//! A brushed DC motor's torque is proportional to its current, so a servo
//! holding `tau` against a stall rating of `tau_stall` draws roughly
//! `idle + noload*moving + (stall - noload) * tau/tau_stall`. That is a model,
//! not a datasheet lookup — real servos add gearbox friction, PWM losses and
//! a large inrush on direction reversal. It is deliberately optimistic, which
//! is why [`Sizing`] applies headroom before choosing parts.
//!
//! # Sensors are not decoration
//!
//! The learned policy reads body pitch, roll, stability margin and a terrain
//! height under each leg's predicted touchdown. On a real robot each of those
//! is a sensor with a range, a rate and a resolution requirement, and
//! [`SensingNeed`] derives them from the gait rather than guessing.

use crate::hardware::{Build, Servo, NM_TO_KGCM};
use crate::robot::MAX_LEGS;
use crate::dynamics::DT;
use crate::terrain::Terrain;

/// Joints on the robot.
/// Ceiling for the per-tick torque row. The live width is the frame's.
pub const MAX_JOINTS: usize = MAX_LEGS * 3;

/// Pack-level specific energy for hobby lithium polymer, Wh/kg. A 3S 5000 mAh
/// pack is about 55 Wh and weighs about 400 g, so this is the real figure
/// including case, wiring and connector — not the cell datasheet number.
pub const LIPO_WH_PER_KG: f64 = 140.0;

/// Nominal volts per lithium polymer cell.
pub const CELL_V: f64 = 3.7;

/// Fraction of pack capacity actually usable before voltage sag bites.
pub const USABLE_FRACTION: f64 = 0.80;

/// Switching regulator efficiency used when converting pack watts to servo
/// watts. Mid-range for a hobby buck module under real load.
pub const CONVERTER_EFF: f64 = 0.85;

/* ------------------------------------------------------------- torque trace */

/// Per-tick, per-joint torque for a gait, normalised to a 1 kg robot.
///
/// Torque is exactly linear in mass, so recording it once lets the fixed point
/// re-evaluate current for any candidate mass without re-running the
/// simulation.
pub struct TorqueTrace {
    /// `ticks * joints` newton-metres per kilogram of all-up mass.
    pub per_kg: Vec<f64>,
    pub ticks: usize,
    /// Joints on this machine: three per leg.
    pub joints: usize,
    pub frame: crate::robot::Frame,
    /// Standing geometry, which is what the sensing requirement needs.
    pub stance: crate::robot::Stance,
    /// Metres of real length per simulator unit.
    pub scale: f64,
}

impl TorqueTrace {
    /// Fold one control tick into the trace.
    ///
    /// Torques come in per kilogram of all-up mass so the sizing loop can price
    /// any candidate mass without re-simulating. Whatever drives the joints
    /// supplies them — there is no built-in recorder any more, because there is
    /// no built-in gait to drive.
    pub fn observe(&mut self, per_kg: &[f64], joints: usize) {
        debug_assert_eq!(joints, self.joints);
        self.per_kg.extend_from_slice(&per_kg[..joints]);
        self.ticks += 1;
    }

    pub fn peak_kgcm(&self, mass_kg: f64) -> f64 {
        self.per_kg
            .iter()
            .fold(0.0f64, |a, b| a.max(*b))
            * mass_kg
            * NM_TO_KGCM
    }

    /// Current draw for a candidate mass and servo, on both sides of the
    /// regulator. The regulator is rated on its *output*, so sizing it against
    /// the pack-side figure would under-spec it whenever the pack sits above
    /// the servo bus voltage.
    pub fn current(&self, mass_kg: f64, servo: &Servo, pack_v: f64) -> Draw {
        if self.ticks == 0 {
            return Draw::default();
        }
        let stall_nm = servo.stall_kgcm / NM_TO_KGCM;
        let mut sum_pack = 0.0;
        let mut sum_servo = 0.0;
        let mut peak_pack = 0.0f64;
        let mut peak_servo = 0.0f64;

        for t in 0..self.ticks {
            let row = &self.per_kg[t * self.joints..(t + 1) * self.joints];
            let mut amps = 0.0;
            for &tau_per_kg in row {
                let tau = tau_per_kg * mass_kg;
                amps += servo.amps_at(tau, stall_nm);
            }
            let pack_a = amps * servo.at_volts / (pack_v * CONVERTER_EFF);
            sum_servo += amps;
            sum_pack += pack_a;
            peak_servo = peak_servo.max(amps);
            peak_pack = peak_pack.max(pack_a);
        }
        let n = self.ticks as f64;
        Draw {
            mean_pack: sum_pack / n,
            peak_pack,
            mean_servo: sum_servo / n,
            peak_servo,
        }
    }
}

/// Current draw on both sides of the regulator, amps.
#[derive(Clone, Copy, Debug, Default)]
pub struct Draw {
    pub mean_pack: f64,
    pub peak_pack: f64,
    pub mean_servo: f64,
    pub peak_servo: f64,
}

/// Static torque about each joint, newton-metres, for the current pose.
/// Mirrors [`crate::hardware::TorqueMeter`] but keeps the joints separate.

/* ------------------------------------------------------------- the catalogue */

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Battery,
    Regulator,
    ServoDriver,
    BusAdapter,
    Compute,
    Ranger,
    Imu,
    Support,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Battery => "BATTERY",
            Kind::Regulator => "REGULATOR",
            Kind::ServoDriver => "SERVO DRIVER",
            Kind::BusAdapter => "BUS ADAPTER",
            Kind::Compute => "COMPUTE",
            Kind::Ranger => "RANGEFINDER",
            Kind::Imu => "IMU",
            Kind::Support => "SUPPORT",
        }
    }
}

/// A non-servo component. Capability fields are interpreted per [`Kind`].
#[derive(Clone, Copy, Debug)]
pub struct Part {
    pub name: &'static str,
    pub maker: &'static str,
    pub kind: Kind,
    pub mass_g: f64,
    pub market_usd: (f64, f64),
    pub vendor_usd: Option<f64>,
    pub vendor_name: &'static str,
    pub source: &'static str,
    pub note: &'static str,

    /// Ranger: typical ranging accuracy, millimetres. Zero when not a sensor.
    pub accuracy_mm: f64,
    /// Battery: cell count. Regulator: output volts. Otherwise unused.
    pub volts: f64,
    /// Battery: capacity mAh. Regulator: continuous amps. Ranger: range in mm.
    pub capacity: f64,
    /// Battery: C rating. Driver: channels. Ranger: max update Hz.
    pub rating: f64,
}

impl Part {
    pub fn unit_price(&self) -> f64 {
        self.vendor_usd.unwrap_or((self.market_usd.0 + self.market_usd.1) * 0.5)
    }

    /// Nominal pack voltage for a battery.
    pub fn pack_volts(&self) -> f64 {
        self.volts * CELL_V
    }

    /// Usable energy in watt-hours.
    pub fn watt_hours(&self) -> f64 {
        self.pack_volts() * self.capacity / 1000.0 * USABLE_FRACTION
    }

    /// Sustained current a pack can deliver at its C rating, amps.
    pub fn max_amps(&self) -> f64 {
        self.capacity / 1000.0 * self.rating
    }
}

pub const PARTS: &[Part] = &[
    // --- batteries -------------------------------------------------------
    Part {
        name: "2S 2200 mAh 25C LiPo",
        maker: "generic",
        kind: Kind::Battery,
        mass_g: 125.0,
        market_usd: (12.0, 20.0),
        vendor_usd: None,
        vendor_name: "",
        source: "https://hobbyking.com/en_us/batteries-chargers/battery-packs.html",
        note: "7.4 V direct to HV servos; no regulator needed.",
        accuracy_mm: 0.0,
        volts: 2.0,
        capacity: 2200.0,
        rating: 25.0,
    },
    Part {
        name: "2S 5000 mAh 50C LiPo",
        maker: "generic",
        kind: Kind::Battery,
        mass_g: 260.0,
        market_usd: (22.0, 34.0),
        vendor_usd: None,
        vendor_name: "",
        source: "https://hobbyking.com/en_us/batteries-chargers/battery-packs.html",
        note: "The usual choice for 7.4 V bus servos.",
        accuracy_mm: 0.0,
        volts: 2.0,
        capacity: 5000.0,
        rating: 50.0,
    },
    Part {
        name: "3S 2200 mAh 30C LiPo",
        maker: "generic",
        kind: Kind::Battery,
        mass_g: 185.0,
        market_usd: (16.0, 26.0),
        vendor_usd: None,
        vendor_name: "",
        source: "https://hobbyking.com/en_us/batteries-chargers/battery-packs.html",
        note: "11.1 V; needs a regulator for 6 V servos.",
        accuracy_mm: 0.0,
        volts: 3.0,
        capacity: 2200.0,
        rating: 30.0,
    },
    Part {
        name: "3S 5000 mAh 50C LiPo",
        maker: "generic",
        kind: Kind::Battery,
        mass_g: 395.0,
        market_usd: (30.0, 48.0),
        vendor_usd: None,
        vendor_name: "",
        source: "https://hobbyking.com/en_us/batteries-chargers/battery-packs.html",
        note: "Headroom for an 18-servo machine with compute on board.",
        accuracy_mm: 0.0,
        volts: 3.0,
        capacity: 5000.0,
        rating: 50.0,
    },
    Part {
        name: "4S 5000 mAh 50C LiPo",
        maker: "generic",
        kind: Kind::Battery,
        mass_g: 530.0,
        market_usd: (45.0, 70.0),
        vendor_usd: None,
        vendor_name: "",
        source: "https://hobbyking.com/en_us/batteries-chargers/battery-packs.html",
        note: "14.8 V. Only worth it for 12 V bus servos like the AX-12A.",
        accuracy_mm: 0.0,
        volts: 4.0,
        capacity: 5000.0,
        rating: 50.0,
    },
    // --- regulators ------------------------------------------------------
    Part {
        name: "S18V20F6",
        maker: "Pololu",
        kind: Kind::Regulator,
        mass_g: 12.0,
        market_usd: (29.0, 33.0),
        vendor_usd: Some(29.95),
        vendor_name: "Pololu",
        source: "https://www.pololu.com/product/2575",
        note: "6 V step-up/step-down, but only ~2 A continuous — far short of a walking hexapod.",
        accuracy_mm: 0.0,
        volts: 6.0,
        capacity: 2.0,
        rating: 0.0,
    },
    Part {
        name: "CC BEC 2.0",
        maker: "Castle Creations",
        kind: Kind::Regulator,
        mass_g: 27.0,
        market_usd: (45.0, 60.0),
        vendor_usd: None,
        vendor_name: "",
        source: "https://home.castlecreations.com/product/cc-bec-2-0/",
        note: "14 A continuous, 20 A peak, output adjustable 4.8-12.6 V. The standard answer for servo-heavy robots.",
        accuracy_mm: 0.0,
        volts: 6.0,
        capacity: 14.0,
        rating: 0.0,
    },
    Part {
        name: "20 A adjustable buck (XL4016 class)",
        maker: "generic",
        kind: Kind::Regulator,
        mass_g: 70.0,
        market_usd: (12.0, 22.0),
        vendor_usd: None,
        vendor_name: "",
        source: "https://www.aliexpress.com/w/wholesale-xl4016-buck-converter.html",
        note: "Cheap and adequate if you add a heatsink and fan. Ratings are optimistic.",
        accuracy_mm: 0.0,
        volts: 6.0,
        capacity: 12.0,
        rating: 0.0,
    },
    // --- driving the servos ----------------------------------------------
    Part {
        name: "PCA9685 16-channel driver",
        maker: "Adafruit",
        kind: Kind::ServoDriver,
        mass_g: 8.0,
        market_usd: (3.0, 15.0),
        vendor_usd: Some(14.95),
        vendor_name: "Adafruit",
        source: "https://www.adafruit.com/product/815",
        note: "I2C, 16 channels — an 18-servo robot needs two. Clones are a few dollars.",
        accuracy_mm: 0.0,
        volts: 5.0,
        capacity: 0.0,
        rating: 16.0,
    },
    Part {
        name: "Mini Maestro 24",
        maker: "Pololu",
        kind: Kind::ServoDriver,
        mass_g: 11.0,
        market_usd: (52.0, 58.0),
        vendor_usd: Some(54.95),
        vendor_name: "Pololu",
        source: "https://www.pololu.com/product/1356",
        note: "24 channels, 333 Hz, 0.25 us resolution, onboard scripting. One board covers the robot.",
        accuracy_mm: 0.0,
        volts: 5.0,
        capacity: 0.0,
        rating: 24.0,
    },
    Part {
        name: "FE-URT-1 bus adapter",
        maker: "Feetech",
        kind: Kind::BusAdapter,
        mass_g: 9.0,
        market_usd: (6.0, 14.0),
        vendor_usd: None,
        vendor_name: "",
        source: "https://www.seeedstudio.com/STS3215-19kg-cm-7-4V-Serial-Servo-p-6338.html",
        note: "Serial-bus servos daisy-chain: one adapter replaces all 18 PWM channels.",
        accuracy_mm: 0.0,
        volts: 5.0,
        capacity: 0.0,
        rating: 253.0,
    },
    // --- compute ----------------------------------------------------------
    Part {
        name: "Teensy 4.1",
        maker: "PJRC",
        kind: Kind::Compute,
        mass_g: 10.0,
        market_usd: (30.0, 38.0),
        vendor_usd: None,
        vendor_name: "",
        source: "https://www.pjrc.com/store/teensy41.html",
        note: "600 MHz Cortex-M7 with a hardware FPU. Runs the IK and the policy with room to spare.",
        accuracy_mm: 0.0,
        volts: 5.0,
        capacity: 0.0,
        rating: 600.0,
    },
    Part {
        name: "ESP32 DevKitC",
        maker: "Espressif",
        kind: Kind::Compute,
        mass_g: 9.0,
        market_usd: (5.0, 12.0),
        vendor_usd: None,
        vendor_name: "",
        source: "https://www.espressif.com/en/products/devkits",
        note: "240 MHz dual core with single-precision FPU, plus Wi-Fi for telemetry.",
        accuracy_mm: 0.0,
        volts: 5.0,
        capacity: 0.0,
        rating: 240.0,
    },
    Part {
        name: "Raspberry Pi Zero 2 W",
        maker: "Raspberry Pi",
        kind: Kind::Compute,
        mass_g: 11.0,
        market_usd: (15.0, 25.0),
        vendor_usd: None,
        vendor_name: "",
        source: "https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/",
        note: "Only if you want to retrain on the robot; a microcontroller is better at hard real time.",
        accuracy_mm: 0.0,
        volts: 5.0,
        capacity: 0.0,
        rating: 1000.0,
    },
    // --- sensing ----------------------------------------------------------
    Part {
        name: "VL53L1X ToF ranger",
        maker: "Adafruit",
        kind: Kind::Ranger,
        mass_g: 2.0,
        market_usd: (4.0, 15.0),
        vendor_usd: Some(14.95),
        vendor_name: "Adafruit",
        source: "https://www.adafruit.com/product/3967",
        note: "30-4000 mm, up to 50 Hz, ~940 nm eye-safe laser with a narrow cone. One per leg.",
        // ST quotes roughly +-5 mm at short range in the dark; worse in
        // sunlight and against dark or angled surfaces.
        accuracy_mm: 5.0,
        volts: 3.3,
        capacity: 4000.0,
        rating: 50.0,
    },
    Part {
        name: "TCA9548A I2C multiplexer",
        maker: "Adafruit",
        kind: Kind::Support,
        mass_g: 3.0,
        market_usd: (2.0, 8.0),
        vendor_usd: None,
        vendor_name: "",
        source: "https://www.adafruit.com/product/2717",
        note: "Every VL53L1X boots at the same I2C address, so six of them need a mux or XSHUT sequencing.",
        accuracy_mm: 0.0,
        volts: 3.3,
        capacity: 0.0,
        rating: 8.0,
    },
    Part {
        name: "BNO055 9-DOF IMU",
        maker: "Adafruit",
        kind: Kind::Imu,
        mass_g: 3.0,
        market_usd: (12.0, 35.0),
        vendor_usd: Some(29.95),
        vendor_name: "Adafruit",
        source: "https://www.adafruit.com/product/4646",
        note: "Onboard fusion outputs absolute orientation at 100 Hz — pitch and roll without writing a filter.",
        accuracy_mm: 0.0,
        volts: 3.3,
        capacity: 0.0,
        rating: 100.0,
    },
    Part {
        name: "MPU-6050 6-DOF IMU",
        maker: "generic",
        kind: Kind::Imu,
        mass_g: 2.0,
        market_usd: (1.5, 5.0),
        vendor_usd: None,
        vendor_name: "",
        source: "https://www.aliexpress.com/w/wholesale-mpu6050.html",
        note: "Cheap, but you write the fusion and it drifts in yaw. Fine here: the policy only needs pitch and roll.",
        accuracy_mm: 0.0,
        volts: 3.3,
        capacity: 0.0,
        rating: 1000.0,
    },
];

pub fn parts_of(kind: Kind) -> impl Iterator<Item = &'static Part> {
    PARTS.iter().filter(move |p| p.kind == kind)
}

/* ------------------------------------------------------------------ sensing */

/// What the policy's observation vector demands of real hardware.
#[derive(Clone, Copy, Debug)]
pub struct SensingNeed {
    /// Distance from the body to a predicted touchdown point, metres.
    pub lookahead_m: f64,
    /// Minimum sample rate to see a landing spot before committing to it.
    pub min_rate_hz: f64,
    /// Height difference that has to be distinguishable, millimetres.
    pub resolution_mm: f64,
    /// One per leg — the policy has six independent lookahead inputs.
    pub rangers: usize,
    /// Pitch and roll are observations 1 and 2.
    pub needs_imu: bool,
    /// The stability margin needs to know which feet are loaded.
    pub needs_contact: bool,
    /// True when the chosen servo reports load over its bus, making separate
    /// contact sensors unnecessary.
    pub contact_from_bus: bool,
}

impl SensingNeed {
    /// Whether a rangefinder actually satisfies this requirement, as
    /// (range, rate, resolution). Resolution is the one that usually fails:
    /// the terrain detail a policy trained in simulation relies on can be
    /// finer than a cheap time-of-flight sensor can resolve.
    pub fn ranger_verdict(&self, part: &Part) -> (bool, bool, bool) {
        (
            part.capacity / 1000.0 >= self.lookahead_m,
            part.rating >= self.min_rate_hz,
            part.accuracy_mm > 0.0 && part.accuracy_mm <= self.resolution_mm,
        )
    }
}

impl SensingNeed {
    pub fn derive(
        frame: crate::robot::Frame,
        stance: &crate::robot::Stance,
        scale: f64,
        servo: &Servo,
    ) -> SensingNeed {
        // The probe sits half a stance sweep ahead of the neutral foot.
        let horizontal = (stance.stance_w * 0.5 + NOMINAL_STRIDE * 0.5) * scale;
        let ride = stance.body_h * scale;
        let swing_s = NOMINAL_SWING_S;

        SensingNeed {
            lookahead_m: (horizontal * horizontal + ride * ride).sqrt(),
            // Four samples per swing is the minimum to place a foot usefully.
            min_rate_hz: 4.0 / swing_s.max(1e-3),
            // The smallest obstacle the courses generate is 10 sim-cm tall;
            // resolve a quarter of that.
            resolution_mm: 0.10 * scale * 1000.0 * 0.25,
            rangers: frame.legs(),
            needs_imu: true,
            needs_contact: true,
            contact_from_bus: servo.feedback,
        }
    }
}

/* -------------------------------------------------------------- the solution */

/// Stride the sensing requirement is stated against, simulator units.
///
/// A learned controller has no fixed stride or duty factor, so the lookahead a
/// terrain sensor needs cannot be read off a gait any more. These are the
/// figures the hand-written gaits produced, kept as the reference the
/// requirement is quoted against rather than silently dropped.
pub const NOMINAL_STRIDE: f64 = 1.08;

/// Swing duration the sampling rate is stated against, seconds.
pub const NOMINAL_SWING_S: f64 = 0.235;

/// Inputs that are not derived from the gait.
#[derive(Clone, Copy, Debug)]
pub struct Sizing {
    /// Frame, brackets, fasteners and wiring, kilograms.
    pub chassis_kg: f64,
    /// Target walking endurance, minutes.
    pub runtime_min: f64,
    /// Extra current for compute and sensors, amps at the pack.
    pub electronics_a: f64,
    /// Multiplier applied to mean current before choosing a battery.
    pub margin: f64,
    /// Torque headroom a servo must clear, matching the servo-sizing tab.
    pub safety: f64,
}

impl Default for Sizing {
    fn default() -> Self {
        Sizing {
            chassis_kg: 0.45,
            runtime_min: 20.0,
            electronics_a: 0.35,
            margin: 1.25,
            safety: 1.35,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub converged: bool,
    pub iterations: usize,
    /// Why no fixed point exists, when `converged` is false.
    pub failure: &'static str,

    pub all_up_kg: f64,
    pub servo_kg: f64,
    pub battery_kg: f64,
    pub chassis_kg: f64,
    pub electronics_kg: f64,

    pub peak_torque_kgcm: f64,
    pub servo_ok: bool,

    pub mean_amps: f64,
    pub peak_amps: f64,
    /// Regulator-output-side current, which is what sizes the regulator.
    pub mean_servo_amps: f64,
    pub peak_servo_amps: f64,
    /// Peak torque including the safety factor — what a servo must clear.
    pub required_kgcm: f64,
    pub mean_watts: f64,
    pub required_wh: f64,
    pub runtime_min: f64,

    pub battery: Option<&'static Part>,
    pub regulator: Option<&'static Part>,
    pub driver: Option<&'static Part>,
    pub driver_count: usize,
    pub compute: Option<&'static Part>,

    pub sensing: SensingNeed,
    pub cost_usd: f64,
    pub cost_servos: f64,
}

/// Solve the mass / torque / current / battery loop for a given servo.
pub fn solve(trace: &TorqueTrace, servo: &Servo, sizing: &Sizing) -> Solution {
    let servo_kg = servo.set_mass_kg(trace.joints);
    // Fixed electronics: driver, compute, IMU, one ranger per leg, mux, wiring.
    let electronics_kg = 0.075;

    let mut battery: Option<&'static Part> = None;
    let mut mass = sizing.chassis_kg + servo_kg + electronics_kg + 0.25;
    let mut mean_a = 0.0;
    let mut peak_a = 0.0;
    let mut draw = Draw::default();
    let mut required_wh = 0.0;
    let mut converged = false;
    let mut failure = "";
    let mut iterations = 0;

    for i in 0..80 {
        iterations = i + 1;

        // Pick the pack voltage from the servo: bus servos above 7 V want a
        // higher-cell pack, 6 V servos run from a regulated 2S or 3S.
        let cells = if servo.at_volts > 9.0 {
            4.0
        } else if servo.at_volts > 7.0 {
            2.0
        } else {
            3.0
        };
        let pack_v = cells * CELL_V;

        draw = trace.current(mass, servo, pack_v);
        // The margin covers what the torque-to-current model leaves out —
        // gearbox friction and reversal inrush — and those are present at the
        // peak as much as at the mean, so both sides carry it. Applying it to
        // only one lets a smooth gait report a peak below its own mean.
        mean_a = draw.mean_pack * sizing.margin + sizing.electronics_a;
        peak_a = draw.peak_pack * sizing.margin + sizing.electronics_a;
        required_wh = mean_a * pack_v * (sizing.runtime_min / 60.0);

        // Smallest pack of the right cell count that carries the energy and
        // can deliver the peak.
        let pick = parts_of(Kind::Battery)
            .filter(|b| (b.volts - cells).abs() < 0.5)
            .filter(|b| b.watt_hours() >= required_wh && b.max_amps() >= peak_a)
            .min_by(|a, b| a.mass_g.partial_cmp(&b.mass_g).unwrap());

        let batt_kg = match pick {
            Some(b) => b.mass_g / 1000.0,
            None => {
                // No stock pack fits; fall back to the ideal mass implied by
                // specific energy so the loop can still report a verdict.
                required_wh / LIPO_WH_PER_KG
            }
        };
        battery = pick;

        let next = sizing.chassis_kg + servo_kg + electronics_kg + batt_kg;
        if (next - mass).abs() < 1e-4 {
            mass = next;
            converged = true;
            break;
        }
        // Damped so an oscillating pick cannot ring forever.
        mass += (next - mass) * 0.6;

        if mass > 60.0 {
            failure = "diverged: the robot cannot carry the battery this runtime needs";
            break;
        }
    }

    if converged && battery.is_none() {
        converged = false;
        failure = "no stock pack has both the energy and the peak-current rating";
    }

    let peak_torque = trace.peak_kgcm(mass);
    let required_kgcm = peak_torque * sizing.safety;
    let servo_ok = servo.stall_kgcm >= required_kgcm;
    if converged && !servo_ok {
        failure = "servo is under-torqued at the converged mass";
    }

    // Regulator only when the servo bus voltage differs from the pack.
    let needs_reg = !(servo.at_volts > 7.0 && servo.at_volts < 8.0);
    let regulator = if needs_reg {
        parts_of(Kind::Regulator)
            .filter(|r| r.capacity >= draw.peak_servo * 0.8)
            .min_by(|a, b| a.unit_price().partial_cmp(&b.unit_price()).unwrap())
    } else {
        None
    };

    // Serial-bus servos need one adapter; PWM servos need enough channels.
    let (driver, driver_count) = if servo.bus == crate::hardware::Bus::Serial {
        (parts_of(Kind::BusAdapter).next(), 1)
    } else {
        let best = parts_of(Kind::ServoDriver)
            .map(|d| {
                let n = (trace.joints as f64 / d.rating).ceil() as usize;
                (d, n, d.unit_price() * n as f64)
            })
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        match best {
            Some((d, n, _)) => (Some(d), n),
            None => (None, 0),
        }
    };

    let compute = parts_of(Kind::Compute).find(|c| c.name == "Teensy 4.1");
    let sensing = SensingNeed::derive(trace.frame, &trace.stance, trace.scale, servo);

    let mut cost = servo.unit_low() * trace.joints as f64;
    let cost_servos = cost;
    if let Some(b) = battery {
        cost += b.unit_price();
    }
    if let Some(r) = regulator {
        cost += r.unit_price();
    }
    if let Some(d) = driver {
        cost += d.unit_price() * driver_count as f64;
    }
    if let Some(c) = compute {
        cost += c.unit_price();
    }
    // Sensing: one ranger per leg, a mux to address them, and an IMU.
    if let Some(r) = parts_of(Kind::Ranger).next() {
        cost += r.unit_price() * sensing.rangers as f64;
    }
    if let Some(m) = parts_of(Kind::Support).next() {
        cost += m.unit_price();
    }
    if let Some(i) = parts_of(Kind::Imu).next() {
        cost += i.unit_price();
    }

    let cells = if servo.at_volts > 9.0 {
        4.0
    } else if servo.at_volts > 7.0 {
        2.0
    } else {
        3.0
    };
    let pack_v = cells * CELL_V;
    let actual_runtime = match battery {
        Some(b) if mean_a > 1e-6 => b.watt_hours() / (mean_a * pack_v) * 60.0,
        _ => 0.0,
    };

    Solution {
        converged,
        iterations,
        failure,
        all_up_kg: mass,
        servo_kg,
        battery_kg: battery.map(|b| b.mass_g / 1000.0).unwrap_or(0.0),
        chassis_kg: sizing.chassis_kg,
        electronics_kg,
        peak_torque_kgcm: peak_torque,
        servo_ok,
        mean_amps: mean_a,
        peak_amps: peak_a,
        mean_servo_amps: draw.mean_servo,
        peak_servo_amps: draw.peak_servo,
        required_kgcm,
        mean_watts: mean_a * pack_v,
        required_wh,
        runtime_min: actual_runtime,
        battery,
        regulator,
        driver,
        driver_count,
        compute,
        sensing,
        cost_usd: cost,
        cost_servos,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::SERVOS;
    /// A trace with a known shape. The sizing loop only ever reads per-kilogram
    /// torques out of this, so a synthetic one exercises the same arithmetic as
    /// a simulated one and pins the numbers exactly.
    fn trace() -> TorqueTrace {
        let frame = crate::robot::Frame::default();
        let joints = frame.legs() * 3;
        let mut t = TorqueTrace {
            per_kg: Vec::new(),
            ticks: 0,
            joints,
            frame,
            stance: crate::robot::Stance::default(),
            scale: 0.10,
        };
        // A rising ramp across the joints, so the peak is unambiguous, held for
        // enough ticks that a mean is meaningful.
        let tick: Vec<f64> = (0..joints).map(|j| 0.02 + 0.002 * j as f64).collect();
        for _ in 0..600 {
            t.observe(&tick, joints);
        }
        t
    }

    fn servo(part: &str) -> &'static Servo {
        SERVOS.iter().find(|s| s.part == part).unwrap()
    }

    #[test]
    fn trace_torque_is_linear_in_mass() {
        let t = trace();
        let a = t.peak_kgcm(2.0);
        let b = t.peak_kgcm(4.0);
        assert!((b / a - 2.0).abs() < 1e-9, "ratio {}", b / a);
    }

    #[test]
    fn a_sensible_build_converges_to_a_sensible_mass() {
        let t = trace();
        let s = solve(&t, servo("DS3218MG"), &Sizing::default());
        assert!(s.converged, "did not converge: {}", s.failure);
        assert!(
            (1.5..6.0).contains(&s.all_up_kg),
            "all-up mass {:.2} kg is not plausible",
            s.all_up_kg
        );
        assert!(s.battery.is_some());
        // Servos dominate the mass budget of a machine this size.
        assert!(s.servo_kg > s.chassis_kg);
    }

    #[test]
    fn regulator_is_sized_on_output_current_not_pack_current() {
        let t = trace();
        let s = solve(&t, servo("MG996R"), &Sizing::default());
        // A 3S pack at 11.1 V feeding 6 V servos means more amps out than in.
        assert!(
            s.peak_servo_amps > s.peak_amps,
            "servo-side {:.2} A should exceed pack-side {:.2} A",
            s.peak_servo_amps,
            s.peak_amps
        );
    }

    #[test]
    fn servo_verdict_uses_the_same_safety_factor_as_the_torque_tab() {
        let t = trace();
        let s = solve(&t, servo("MG996R"), &Sizing::default());
        assert!((s.required_kgcm / s.peak_torque_kgcm - 1.35).abs() < 1e-9);
        assert_eq!(s.servo_ok, servo("MG996R").stall_kgcm >= s.required_kgcm);
    }

    #[test]
    fn current_is_positive_and_peak_exceeds_mean() {
        let t = trace();
        let s = solve(&t, servo("MG996R"), &Sizing::default());
        assert!(s.mean_amps > 0.1, "mean {}", s.mean_amps);
        assert!(s.peak_amps >= s.mean_amps, "{} < {}", s.peak_amps, s.mean_amps);
        assert!(s.mean_watts > 1.0);
    }

    #[test]
    fn heavier_robots_draw_more_current() {
        let t = trace();
        let light = solve(
            &t,
            servo("DS3218MG"),
            &Sizing {
                chassis_kg: 0.3,
                ..Sizing::default()
            },
        );
        let heavy = solve(
            &t,
            servo("DS3218MG"),
            &Sizing {
                chassis_kg: 3.0,
                ..Sizing::default()
            },
        );
        assert!(
            heavy.mean_amps > light.mean_amps,
            "{} !> {}",
            heavy.mean_amps,
            light.mean_amps
        );
        assert!(heavy.all_up_kg > light.all_up_kg);
    }

    #[test]
    fn demanding_an_absurd_runtime_fails_instead_of_lying() {
        let t = trace();
        let s = solve(
            &t,
            servo("DS3218MG"),
            &Sizing {
                runtime_min: 6000.0,
                ..Sizing::default()
            },
        );
        assert!(!s.converged, "claimed to solve a 100-hour hexapod");
        assert!(!s.failure.is_empty(), "failure must be explained");
    }

    #[test]
    fn serial_bus_servos_need_an_adapter_not_pwm_channels() {
        let t = trace();
        let bus = solve(&t, servo("STS3215"), &Sizing::default());
        assert_eq!(bus.driver.unwrap().kind, Kind::BusAdapter);
        assert_eq!(bus.driver_count, 1);
        assert!(bus.sensing.contact_from_bus, "bus servos report load");

        let pwm = solve(&t, servo("DS3218MG"), &Sizing::default());
        assert_eq!(pwm.driver.unwrap().kind, Kind::ServoDriver);
        // 18 joints cannot fit on one 16-channel board.
        let ch = pwm.driver.unwrap().rating as usize;
        assert!(ch * pwm.driver_count >= t.joints);
        assert!(!pwm.sensing.contact_from_bus, "PWM servos report nothing");
    }

    #[test]
    fn seven_point_four_volt_servos_skip_the_regulator() {
        let t = trace();
        let hv = solve(&t, servo("STS3215"), &Sizing::default());
        assert!(hv.regulator.is_none(), "2S feeds 7.4 V servos directly");

        let lv = solve(&t, servo("MG996R"), &Sizing::default());
        assert!(lv.regulator.is_some(), "6 V servos need a regulator from 3S");
        // The 2 A Pololu part must not be chosen for an 18-servo machine.
        assert!(
            lv.regulator.unwrap().capacity >= lv.peak_servo_amps * 0.8,
            "picked a {} A regulator for {:.1} A of servos",
            lv.regulator.unwrap().capacity,
            lv.peak_servo_amps
        );
        // The 2 A Pololu part cannot serve eighteen servos.
        assert_ne!(lv.regulator.unwrap().name, "S18V20F6");
    }

    #[test]
    fn sensing_requirements_follow_the_gait() {
        let t = trace();
        let s = solve(&t, servo("DS3218MG"), &Sizing::default());
        let n = s.sensing;
        assert_eq!(n.rangers, 6, "one lookahead input per leg");
        assert!(n.needs_imu && n.needs_contact);
        // A 28 cm robot probes a few tens of centimetres ahead.
        assert!(
            (0.05..1.5).contains(&n.lookahead_m),
            "lookahead {:.3} m",
            n.lookahead_m
        );
        // And it has to sample faster than its own swing phase.
        assert!(n.min_rate_hz > 5.0 && n.min_rate_hz < 400.0, "{}", n.min_rate_hz);
        // The VL53L1X in the catalogue must actually satisfy it.
        let r = parts_of(Kind::Ranger).next().unwrap();
        let (range_ok, rate_ok, res_ok) = n.ranger_verdict(r);
        assert!(range_ok, "ranger too short");
        assert!(rate_ok, "ranger too slow: {} Hz", r.rating);
        // A VL53L1X is +-5 mm; the gait wants finer than that at this scale.
        // The point of the check is that the tool says so instead of pretending.
        assert!(!res_ok, "expected the resolution gap to be reported, not hidden");
    }

    #[test]
    fn catalogue_entries_are_internally_consistent() {
        for p in PARTS {
            assert!(p.market_usd.0 > 0.0 && p.market_usd.1 >= p.market_usd.0, "{}", p.name);
            assert!(p.mass_g > 0.0, "{}", p.name);
            assert!(p.source.starts_with("https://"), "{} has no source", p.name);
            if p.vendor_usd.is_some() {
                assert!(!p.vendor_name.is_empty(), "{} has no vendor", p.name);
            }
        }
        // Every category the solver reaches for must be populated.
        for k in [
            Kind::Battery,
            Kind::Regulator,
            Kind::ServoDriver,
            Kind::BusAdapter,
            Kind::Compute,
            Kind::Ranger,
            Kind::Imu,
            Kind::Support,
        ] {
            assert!(parts_of(k).next().is_some(), "{k:?} is empty");
        }
    }

    #[test]
    fn part_names_are_unique() {
        // Consumers identify parts by name because `PARTS` is a const and
        // pointer identity is not stable across crates.
        for (i, a) in PARTS.iter().enumerate() {
            for b in PARTS.iter().skip(i + 1) {
                assert_ne!(a.name, b.name, "duplicate part name");
            }
        }
    }

    #[test]
    fn battery_energy_and_current_ratings_are_sane() {
        let b = parts_of(Kind::Battery)
            .find(|b| b.name.starts_with("3S 5000"))
            .unwrap();
        // 11.1 V x 5 Ah x 0.8 usable is about 44 Wh.
        assert!((b.watt_hours() - 44.4).abs() < 1.0, "{}", b.watt_hours());
        // 50C on 5 Ah is a lot of amps; it must not be the binding constraint.
        assert!(b.max_amps() > 100.0);
    }
}
