# Hexapod Gait Lab

A legged-locomotion simulator with a policy-search trainer and a hardware
sizer. Four legs to ten; six is the default and the one everything is tuned
at. The simulator, the learner and the kinematics are Rust; the browser only
draws.

Start it from `dist/hexapod-simulator.html` — one self-contained file, no
server, no network, no dependencies.

```
./build.sh          # test, compile wasm, emit dist/hexapod-simulator.html
node test/smoke.mjs # drive the built page in a real browser
```

## What it does

The dashboard runs a hand-tuned tripod gait over a generated obstacle course at
a speed you command, following a route the course lays out for it. Press
**Train** and an ARS learner starts from an exact copy of that gait and searches
for something better — live, in the page. The **Servos** and **System** tabs
turn whatever gait is running into a list of parts you can actually buy, and the
servo you pick there is the servo the simulator drives the joints with.

## The simulator

A **centroidal rigid-body** model with Coulomb contact and actuator limits. Not
a full articulated multibody engine, and the About tab says so in the page too.

**Contact.** A stance foot transmits at most `mu * N` horizontally. Past that it
skids: the body keeps the momentum it already had, and the contact point moves
in the world, corrupting the support polygon and the plane fit for every tick
that follows. Grip varies by surface — loose rubble is worth about two thirds of
firm ground, a trench floor three quarters.

**Actuators.** Joints are integrated toward the inverse-kinematics solution at a
rate that falls linearly to zero at stall — a brushed motor's torque-speed line
— and past stall they back-drive, the leg folds, and the chassis sags until the
joints run out of mechanical travel. The stall torque and no-load speed come
from the servo selected in the catalogue.

**Momentum.** The chassis carries linear and angular velocity. It cannot change
speed faster than traction allows, it coasts when a foot slips, and its own
acceleration throws the centre of mass around — braking hard pitches the mass
forward over the toes, which is what actually tips a legged robot that stops in
a hurry.

**Leg mass.** The femur and tibia assemblies weigh something, and on a machine
this size they are about half of it. A swinging leg holds itself up against
gravity and has to be accelerated and stopped; its joints pay for both in
torque, and the reaction goes back into the chassis and eats the same friction
budget as everything else. Swing used to be free — a leg in the air carried no
load, so it moved at the servo's no-load speed however fast it was asked to.
Only two of the three servos swing: the coxa is bolted to the chassis.

Turning leg mass off (`--leg-mass 0`) recovers the old model exactly, and it is
worth **15 %** of the hand-tuned baseline's reward on `MIXED`: 68.4 with
weightless legs against 58.2 with real ones.

**Walls.** The corridor is fenced by two invisible walls at its edges, and some
of what is inside it cannot be climbed either — a slalom wall is nearly twice
the length of a leg. Both answer the same obstruction test, because there is no
reason for them to be two different things, and so do the legs: a femur that
would enter a wall stops the chassis the same way the disc around the body does.
Hitting one takes the velocity component into it and keeps the component along
it, so a machine that arrives square stops dead and one that arrives at an angle
slides along and gets round. That difference is the entire reason a steering
action is worth having.

Feet get the same treatment from the other end: a step aimed at ground higher
than a leg can reach is deflected outward to the nearest foothold it can
actually use, and if there is none the step lands short and pays the usual
stubbing penalty. Without that, the kinematics get asked to plant a foot two
metres in the air, half succeed, and tip the support plane over.

Also modelled, as before: a foot that cannot reach its planted position brakes
the chassis, a foot that swings too low catches on terrain, and the chassis
collides with terrain taller than its clearance — as do the feet, which used
to be planted inside a slalom wall because a six-point sample of the chassis
disc missed it and the foothold fallback left them there.

What is still kinematic, stated plainly: links are rigid, contact is resolved
once per tick rather than by an impulse solver, and the leg-inertia terms are
the diagonal ones — each joint sees the mass below it, without the off-diagonal
coupling or the Coriolis terms of a full mass matrix.

Legs are solved with analytic 3-DOF inverse kinematics at 100 Hz. About 7.0 µs
per step natively for a hexapod's 18 joints — 1400× real time, which is what
keeps in-browser training practical. (5.9 µs before the forward scan and the
route bookkeeping; the extra microsecond buys the ability to steer.)

## The reward

The reward contains **no distance term at all**. The task is to hold a commanded
speed, and distance is what happens when you succeed.

That is the second attempt at this. The first rewarded raw distance, and the
optimiser correctly pinned cycle time and stride to their bounds and ran flat
out having learned nothing about terrain. The obvious patch — reward holding one
fixed cruise speed — fixes the symptom and leaves the disease: the policy still
specialises, just on a number chosen by hand rather than a bound.

So the command is an **input**. It is sampled fresh from 1.5–6.0 m/s for every
rollout, shared between both sides of each finite difference, and handed to the
policy as an observation. Evaluation averages three speeds. A gait that only
works at one speed cannot score, and there is no longer a hard-coded speed
anywhere in the simulator.

For that to be a solvable problem the policy needs the controls a real legged
controller has, so three of its actions scale **cycle time, stride and duty
factor** online, every tick. Beyond speed tracking it pays for mechanical
work in joules, foot skid in metres, terrain clipped by a swinging foot, time
spent asking a servo for more torque than it has, and going over.

Navigation adds three more terms, and the same discipline applies to all of
them. Bearing error to the next waypoint is cheap — the machine is asked to
*get* there, not to point at it every instant, or a detour round an obstacle
would cost more than walking into one. Progress toward it is paid per metre
closed **and capped at the metres the commanded speed would have covered
anyway**, so past the command there is nothing more to earn and all it can buy
is pointing the right way. Reaching one is worth a small fixed bonus; a large
one would be mileage again, since arrivals come at whatever rate the route
happens to be spaced at. The first version of this had an uncapped progress
term and a bonus twice the size, and the result was exactly the failure this
reward was written to avoid — the walk-to-run duty trend below flattened out
while the optimiser went shopping for distance.

### What it learned

500 iterations on `MIXED`, horizon 12 s
(`hexapod speed --iters 500 --dirs 24 --top 8`):

| commanded | achieved | cycle time | stride | duty factor |
| --------: | -------: | ---------: | -----: | ----------: |
|  2.00 m/s |     1.93 |      0.520 |  0.923 |   **0.581** |
|  2.75 m/s |     2.67 |      0.476 |  0.973 |       0.546 |
|  3.50 m/s |     3.36 |      0.507 |  0.927 |       0.540 |
|  4.25 m/s |     4.05 |      0.477 |  0.957 |       0.516 |
|  5.00 m/s |     4.71 |      0.461 |  0.957 |       0.500 |
|  5.75 m/s |     5.37 |      0.461 |  0.974 |   **0.487** |

Walking slowly it keeps more feet on the ground; running it drops to a tripod.
That is the walk-to-run transition, it is monotonic across the whole range, and
nothing in the reward mentions duty factor, stride or gait at all. Mean speed
error is 0.20 m/s against the hand-tuned gait's 0.26.

### It needed a bigger search, and finding that out was the whole exercise

Adding a forward scan and something to steer toward took the policy from 252
parameters to 428. Run at the settings that used to work
(`--dirs 10 --top 4`), the walk-to-run trend **disappears**: duty comes out
flat and slightly *rising* with speed, and mean speed error lands at 0.36,
worse than the hand-tuned gait it started from.

A thousand extra iterations do not fix it — 1500 returns the identical policy,
because the best-so-far is found early and never beaten. Zeroing the three
navigation reward terms does not fix it either; the trend stays flat, which
rules the new reward out as the cause. What fixes it is **more directions per
iteration**: 24 instead of 10, and the table above is what comes back.

That is what ARS is: the update is estimated from the spread of returns across
sampled directions, so the population needed grows with the parameter count.
Seventy per cent more parameters needed a bigger sample, not a longer run. The
shipped default moved from 8 directions to 16 because of this, and it is the
kind of thing that is very easy to ship silently as "the learner got worse
after we added perception".

Reward on `MIXED` goes **58.2 → 126.7 (+118 %)** over 400 iterations. Absolute
rewards are not comparable across changes to the reward function — this one has
navigation terms in it that the earlier numbers in this file's history did
not — so what is worth reading is the gap between the two columns of the same
run, never a number on its own.

It generalises: trained on `MIXED`, it is better on all nine other courses,
including one full of walls it never saw. The sweep is in the next section.

## Where it goes

Walking is only half a locomotion problem. The other half is that the way
forward is sometimes not forward, and a controller with no steering and nothing
to steer toward cannot express that at all.

So every course carries a **route**: waypoints in order, ending at the far end
of the corridor. On the open courses the route runs straight down the middle
and having it changes nothing, which is the point — the old behaviour is a
special case, not a thing that was replaced. On the courses with something in
the way, the generator lays the route through the gaps as it builds them, so
there is no search and no chance of a route through a wall.

The policy gets one new action, **steer**, and four new kinds of observation:
bearing and range to the next waypoint, where it sits between the two walls, and
a six-point **forward scan** of how much higher the ground is one and three
metres ahead, at three bearings across the body. The per-leg lookaheads it
already had only ever see the next footfall, which is far too late to turn on.
The scan sees the invisible walls too — they are invisible, not undetectable,
and a policy that cannot sense a fence cannot avoid one.

Nine courses, all generated from a seed:

| course | what is in it |
| --- | --- |
| `FLAT` | Nothing. The reference case |
| `STEPS` | Staircases across the corridor, 16–34 cm per riser |
| `RUBBLE` | Scattered debris 10–58 cm tall |
| `GAPS` | Trenches 45–105 cm wide, 90 cm deep |
| `MIXED` | Rubble, stairs, trenches, rubble. The default |
| `RAMPS` | Grades up to 1.3 m, about half of them banked across the corridor |
| `SLALOM` | Walls with a 3.5 m gate in each, staggered left and right |
| `SLICK` | Ice at a fifth of the grip of the ground around it |
| `GAUNTLET` | All of the above in one run |

`RAMPS` is a different problem from `STEPS`: a staircase is a sequence of
shocks, a ramp is a sustained tilt, and a banked one rolls the support plane and
slides the machine sideways the whole way up. `SLICK` is the course the traction
meter was built for. `SLALOM` is the one that needs the route.

### Steering is worth more than gait tuning, where the way forward is not forward

400 iterations, horizon 8 s, on the two courses that need a route:

| course | hand-tuned | learned | distance |
| --- | ---: | ---: | ---: |
| `SLALOM` | −32.1 (falls at the first wall) | **74.9** | 7.3 m → **18.3 m** |
| `GAUNTLET` | 21.3 (falls) | **120.9** | 23.3 m → **26.9 m** |

The hand-tuned gait on `SLALOM` walks 7.3 m, meets a wall, and goes over. It has
no steering action, so there is nothing it could have done. Several hundred
per cent is not a gait improvement — it is the difference between a controller
that can express *go round that* and one that cannot.

It works in the browser too, which is where anybody will actually try it: two
and a half minutes of in-page training on `SLALOM` is about 1300 iterations,
and takes the machine from −32 to +54 and from stuck at the first wall to four
waypoints down the course.

And it transfers. Trained on `MIXED` seed 1 for 300 iterations, then scored on
all nine courses (`hexapod sweep`):

| course | hand-tuned | learned | waypoints reached |
| --- | ---: | ---: | ---: |
| `FLAT` | 86.5 | **138.3** | 11 → 11 |
| `STEPS`\* | 61.9 | **63.5** | 11 → 6 |
| `RUBBLE`\* | 59.8 | **113.3** | 11 → 11 |
| `GAPS`\* | −43.6 | **−26.2** | 9 → 9 |
| `MIXED` | 58.2 | **121.6** | 10 → 10 |
| `MIXED`\* | 52.5 | **104.7** | 11 → 10 |
| `RAMPS`\* | 81.1 | **131.2** | 11 → 10 |
| `SLALOM`\* | −36.6 | **42.2** | 0 → 9 |
| `SLICK`\* | 81.0 | **122.1** | 11 → 10 |
| `GAUNTLET`\* | 26.8 | **85.6** | 8 → 11 |

\* = a course seed the policy never trained on.

The row worth staring at is `SLALOM`. The policy never trained on it and never
saw a wall — and it still goes from zero waypoints and a faceplant at 7.3 m to
nine waypoints and 14.8 m. There is nothing on `MIXED` to steer *around*, but
there is plenty to be pushed off line by, so bearing error is never zero for
long and the steering row of the feedback matrix gets a gradient anyway. What
generalises is "reduce the bearing to where you are going", which is exactly
the right thing to have learned.

That result is also a correction. Measured at the old ten-direction population,
this same sweep left `SLALOM` flat at −34, and the tidy conclusion on offer was
"perception you never need is perception you never learn". It was wrong, and it
was wrong because the search was undersized rather than because the claim was
true. Two of the three interesting findings in this section only appeared after
fixing that.

`STEPS` is the one case where reward barely moves and distance halves: the
learned policy runs, and running at a staircase gets it 18.5 m and a fall where
walking got 29.1 m and none. Reward is not distance, and here it is nearly a
wash.

## How many legs

The frame is a value, not a constant. Any even count from four to ten: the
chassis grows, the leg mounts spread evenly front to back, the presets are
closed forms in the pair index, and the policy is reshaped around the new
observation and action counts — `17 + n` observations, `2n + 6` actions. At
three pairs every formula reproduces the original hand-built hexapod exactly,
down to the phase offsets, which is what the tests check.

Same course, same reward, 300 iterations each, all at a fixed 2 kg all-up:

| legs | swinging leg mass | hand-tuned | learned |
| ---: | ---: | ---: | ---: |
| 4 | 648 g (32 %) | **108.5** | 128.6 |
| 6 | 972 g (49 %) | 58.2 | **121.6** |
| 8 | 1296 g (65 %) | 35.3 | **111.5** |
| 10 | 1620 g (81 %) | 4.0 | **90.8** |

The hand-tuned column falls off a cliff: 108.5 down to 4.0 as legs are added.
The reason is the leg-inertia model — at a fixed all-up mass, every leg added is
another 162 g that has to be swung, and by ten legs four fifths of the robot is
legs. Without leg mass in the simulator, extra legs would be free stability.
They are not, and the two features had to land together for either to say
anything true.

The learned column falls off much more gently, 128.6 to 90.8, which is the more
interesting half: a hand-tuned gait scaled to ten legs is nearly useless, and
most of that is recoverable by searching for a gait that suits the frame rather
than transplanting the hexapod's.

The sizing tool agrees from the other direction. More feet share the load, so
the torque per joint *falls*, but the machine gets heavier, thirstier and dearer:

| legs | peak joint | all-up | mean draw | endurance | total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 9.9 kg·cm | 1.43 kg | 2.8 A | 37 min | $278 |
| 6 | 13.3 | 1.79 kg | 3.0 A | 36 min | $371 |
| 8 | 12.8 | 2.15 kg | 3.8 A | 28 min | $448 |
| 10 | 8.0 | 2.51 kg | 4.1 A | 26 min | $526 |

### A quadruped will not trot

Four legs alternating is a trot, which stands on two diagonal feet. Two feet are
a line, not a polygon, so the centre of mass is never inside the support and the
robot goes over — in this simulator. Real quadrupeds trot perfectly well,
because a trot is *dynamically* stable and stability here is judged statically,
by where the centre of mass projects relative to the support polygon.

So a four-legged frame starts on the crawl instead, the dashboard marks the trot
button with a warning rather than hiding it, and there is a test that asserts
the trot falls and the crawl does not. Picking the wrong answer quietly would
have been easy; saying which model is being run is the useful thing.

## The servo is a simulator input

This is the part that changed most. The servo used to be a post-hoc sizing
decision; now its torque-speed line drives the joints, so it changes what the
optimiser converges to. `hexapod servo` trains the same course once per servo:

| servo | stall | no-load | hand-tuned | learned | peak / stall |
| --- | ---: | ---: | ---: | ---: | ---: |
| SG90 | 1.8 kg·cm | 100 rpm | −156.5 | −42.5 | **11.8×** |
| MG90S | 2.2 | 125 | −214.2 | −44.0 | **8.8×** |
| MG995 | 10.0 | 62 | 43.0 | 107.5 | 1.97× |
| MG996R | 11.0 | 71 | 57.7 | **119.4** | 1.41× |
| DS3218MG | 20.0 | 62 | 58.2 | 110.5 | **0.89×** |
| LX-16A | 17.0 | 45 | 27.3 | 98.1 | 1.11× |
| STS3215 | 19.5 | 42 | 18.1 | **116.7** | **0.84×** |
| AX-12A | 15.3 | 59 | 48.8 | 110.5 | 1.35× |

The micro servos are driven nine to twelve times past their rating: the legs
fold, the chassis sits on the ground, and the reward is deeply negative — the
robot does not walk, in the simulator, for the same reason it would not walk on
a bench. The useful part is that this is a **second, independent** route to the
same conclusion the static sizing arithmetic reaches: the torque calculation
says the build needs about 18 kg·cm, and the only two servos that stay under
stall in simulation are the 19.5 and 20 kg·cm ones. Two different models
agreeing is worth more than either alone.

Note also that the *learned* column is much flatter than the hand-tuned one.
Given a search, most of these servos end up somewhere between 98 and 119 —
the learner adapts the gait to the actuator, which is exactly what it is for,
and exactly why the servo has to be a simulator input rather than a sizing
decision made afterwards.

## The learner

Augmented Random Search ([Mania, Guy & Recht,
2018](https://arxiv.org/abs/1803.07055)), the V2-t variant: finite-difference
estimates along random directions, scaled by the spread of the returns, using
only the best-scoring directions, over a linear policy with normalised
observations. Derivative-free, so the simulator never has to be differentiable.

428 parameters on six legs, and `8 + n + n_act * n_obs` in general:

- 6 gait scalars (cycle time, stride, step height, body height, stance width, duty)
- one phase offset per leg — so the coordination pattern itself is learned
- 2 lateral stance trims, on the outermost pairs
- an 18×23 linear feedback matrix (18 actions, 23 observations)

Observations, `17 + n` of them: body height error, pitch, roll, stability
margin, gait phase (sin/cos), speed error against the command, a terrain-height
lookahead under each leg's predicted touchdown, the commanded speed, bearing and
range to the next waypoint, position between the two walls, and a six-point
forward terrain scan. Actions, `2n + 6`: per-leg step height and touchdown
offset, body height and pitch trim, the three gait modulations, and steering.

The feedback block starts at zero, so **iteration 0 is exactly the hand-tuned
gait** — including the modulation actions, which sit at their nominal values
until something is learned. The comparison in the UI is honest by construction,
and there is a test for it.

The default population is **16 directions**, up from 8. ARS estimates its
update from the spread of returns across sampled directions, so the sample it
needs scales with the parameter count, and the parameter count grew by seventy
per cent when the policy gained something to look at and somewhere to go. See
the section above for what happens if you leave it at 8.

## Servo sizing

For a foot carrying vertical load `F`, the static torque about a horizontal
joint is `F` times the horizontal distance from that joint to the foot — the
Jacobian transpose specialised to a vertical force. The coxa rotates about a
vertical axis, so it is sized by traction instead. One function computes this,
and the simulator, the torque meter and the power model all call it.

```
hexapod bom --mass 2.0 --scale 0.10 --iters 200
```

On the default 2 kg / 28 cm build the hand-tuned gait needs **12.6 kg·cm** and
the learned one **17.8 kg·cm** — the peak load on a single foot goes from 9.8 N
to 17.0 N. An earlier version of this file claimed the opposite, that the
learned gait was cheaper to build as well as better at walking; that was true of
the old kinematic simulator and is not true now. Dropping the duty factor to run
fast means fewer feet on the ground at the moment of peak load, and somebody has
to pay for it. The tool's job is to show the trade, not to have a favourite.

## Whole-machine sizing

`hexapod system` (and the **System** tab) sizes the rest of the robot, and the
interesting part is that it cannot be done in one pass. Battery mass is part of
all-up mass, all-up mass sets joint torque, torque sets current, and current
sets the battery you need. That is a fixed point, solved by iteration —
typically nine rounds — and **it does not always have one**: ask for two hours
of endurance and the solver reports that the robot cannot carry the battery its
own runtime demands rather than inventing a number.

Current comes from the torque trace, not a rule of thumb: a brushed DC motor's
torque is proportional to its current, so each servo draws
`idle + (stall - idle) * tau/tau_stall` at every tick of the gait. The model
ignores gearbox friction and reversal inrush, so it reads optimistic — hence the
headroom applied before parts are chosen, on the peak as well as the mean.

Sizing every catalogue servo as a complete robot, on the default 20-minute,
0.45 kg-chassis build with the learned gait:

| servo | all-up | needs | stall | battery | mean | endurance | total |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| MG996R | 1.70 kg | 15.0 | 11.0 | 3S 2200 | 3.9 A | — | under-torqued |
| DS3218MG | 1.79 kg | 15.8 | 20.0 | 3S 2200 | 2.8 A | 38 min | **$371** |
| LX-16A | 1.65 kg | 14.5 | 17.0 | 3S 2200 | 1.8 A | 59 min | $495 |
| STS3215 | 1.73 kg | 15.2 | 19.5 | 2S 5000 | 4.6 A | 23 min | $509 |
| AX-12A | 2.04 kg | 18.0 | 15.3 | 4S 5000 | 4.0 A | — | under-torqued |

Two things fall out that are easy to miss by hand. Pack voltage is chosen by the
servo: a 7.4 V bus servo runs straight off 2S and needs **no regulator at all**,
while 6 V servos need one sized on its *output* current, which is higher than
the pack current whenever the pack sits above the bus. And the LX-16A costs $124
more than the DS3218MG but adds twenty minutes of endurance — it draws far less
current for the same torque.

## The sensors are dictated by the policy

The learned policy reads a specific observation vector every control tick, and
on a real robot each entry is a measurement with a range, a rate and a
resolution:

| observation | becomes | requirement |
| --- | --- | --- |
| 6× terrain height under predicted touchdowns | one rangefinder per leg | reach, rate and resolution derived from the gait |
| 6× forward terrain scan, 1.4 m and 3.0 m ahead | a forward-facing sensor with a field of view | longer reach than the per-leg rangefinders |
| bearing and range to the next waypoint | odometry or a fix | drift over the length of a course |
| body pitch, roll | IMU | control rate |
| stability margin | which feet are loaded | contact sensing |

The three per-leg numbers are derived from the gait, not chosen: reach is half a
stance sweep plus half a stance width at the build's scale, the rate is four
samples per swing phase, and the resolution is a quarter of the smallest
obstacle the courses generate.

The two navigation rows are honest debts, not solved problems: the parts list
sizes the per-leg rangefinders and does not yet size a forward sensor or say
where the pose fix comes from. On a real machine those are the hard parts, and
pretending otherwise would make the bill of materials a fiction.

Two of those requirements do not pass, and the tool says so:

- A **VL53L1X** covers the range and the rate easily, but it is about ±5 mm —
  coarser than the gait wants at 28 cm scale, and worse in sunlight or against
  dark and angled surfaces. Either scale the robot up so terrain features are
  larger relative to sensor noise, or filter the lookahead and accept the lag.
- **Foot contact** is free on a serial-bus servo, which reports load over the
  same wire, and needs six extra sensors on a PWM servo. That is a real reason
  the LX-16A and STS3215 are worth their price premium, and it is not visible
  from a torque calculation alone.

### Prices

Eighteen non-servo components — batteries, regulators, servo drivers, a bus
adapter, compute boards, rangefinders, an I²C multiplexer and IMUs — sit
alongside eight servos commonly used for hexapods. Every entry carries a source
URL and a provenance flag, and the catalogue lives in
`crates/hexapod-core/src/hardware.rs` — the browser reads it as JSON generated at
build time, so there is one source of truth.

Prices are recorded observations, not live quotes:

- **Distributor** prices were read from the linked vendor page on 2026-08-16:
  MG995 $19.95 (Adafruit), MG996R $10.95 (JSumo), STS3215 $21.99 (Seeed
  Studio), AX-12A $57.39 (Robotis).
- **Marketplace** bands are indicative street prices for AliExpress-grade
  listings and are *not* quotes.

Stall torque and no-load speed are manufacturer ratings, and only the MG996R's
electrical figures are datasheet-sourced; the rest are flagged as
commonly-repeated values. Stall torque is a ceiling, not a duty point —
continuous torque is far lower, which is what the safety factor is for.

## Layout

```
crates/hexapod-core    simulator, dynamics, ARS trainer, hardware. 112 tests.
crates/hexapod-cli     train, bench, sweep, speed, servo, bom, system, courses
crates/hexapod-wasm    C-ABI bridge; no wasm-bindgen, ~158 kB of wasm
web/                   dashboard: renderer, panels, styling
build.sh               inlines wasm as base64 into one HTML file
test/smoke.mjs         Playwright end-to-end check of the built page
```

## Reading the stage

| Mark | Meaning |
| --- | --- |
| Solid red ring | Foot planted; grows with its share of the load |
| Dashed grey ring | Where a swinging leg intends to touch down |
| Dashed red polygon | Support polygon — convex hull of the planted feet |
| Crosshair | Centre of mass; turns red near the polygon edge |
| Clay block | Unclimbable wall. Turns red, with a red outline, when the chassis or a link is inside it |
| Slate slab | Staircase tread |
| Sage strip | Ramp slab |
| Sand block | Rubble |
| Pale cyan sheet | Ice. Thin, and slippery |
| Recessed grey box | Pit. Falling in usually ends the run |
| Red chassis / leg | That body is currently in a wall |
| Red ring with a post | The waypoint being chased right now |
| Small dashed rings | The rest of the route, faded once passed |
| `1` / `2` / `3` | Orbit, top and side cameras. Drag the stage to take orbit back |
| Dashed red line at ±5 m | An invisible wall. Uprights fade in as you close on it |

Drag to orbit, scroll to zoom, `WASD`/`QE` to drive, `X` to stop, `space` to
pause, `1`/`2`/`3` for orbit / top / side. The **Frame** slider sets the leg count, the **Commanded speed** dial is
the number the reward tracks, and the **Machine** selector picks the servo the
joints are driven with. Changing any of them is a different robot, so anything
learned for the old one is discarded. **Follow route** hands steering to the
policy; the turn keys take it back for as long as you hold them.

### The gait pattern panel is a measurement

It used to draw the gait's *schedule* — phase offsets out of the parameter
vector, laid out as bars. It was right by construction and therefore said
nothing: it read TRIPOD whether or not the machine was walking one.

It now records what the feet actually did. Every frame it takes each leg's
stance flag and load share out of the telemetry buffer, and the panel is that
recording: one row per leg over the last four seconds, dark where the foot was
down and darker the more weight it carried. Cycle time comes from the median
interval between one leg's footfalls. Duty comes from counting them. Each leg's
phase offset is recovered by reading the gait clock at its rising edges, and the
pattern is named by matching those against the preset tables — asked for over
the wasm bridge rather than kept as a second copy in the JavaScript. A
coordination the learner invented and nobody has a name for reads as
`IRREGULAR`, and one that started as a tripod and drifted into a wave says so.

That last case is not hypothetical: the end-to-end test found the learner had
re-phased a hexapod tripod into a clean six-slot wave, and the panel reported it
before anybody looked for it.

## Tests

`cargo test` — 112 tests.

Geometry, on every frame from four legs to ten: IK round-trips against forward
kinematics, hips that never collide however many legs there are, mirrored pairs
marching front to back, joint travel that contains the neutral stance, and the
six-legged frame reproducing the original hand-built yaw angles, body radius and
preset phase offsets exactly — because if that drifts, none of the numbers
recorded above stay comparable.

Dynamics: the torque-speed line runs from no-load to stall, a joint only gives
way past stall, the ideal joint is deadbeat at the control rate, torque grows
with lever arm and scale, leg inertia scales as length squared, a leg in the air
still loads the two pitch joints and not the coxa, and the collapse direction
matches a numerical derivative of forward kinematics.

Behaviour, which is where the interesting ones are: stance feet do not move at
all when there is traction to spare, the same gait skids five times further on
ice than on grip, a weaker servo scores worse *and* is driven past stall, an
overloaded servo visibly sags the chassis, one tick of "stop" cannot remove
5 m/s but a second of it can, swinging legs push back on the chassis with a real
but sub-weight force, heavier legs cost more work and load the servos harder,
weightless legs cost exactly nothing, every leg count walks and holds its
commanded speed, ten legs are more stable than four, a trotting quadruped falls
over and a crawling one does not, cost of transport lands in the range legged
machines actually occupy, the body holds the speed it is commanded regardless of
what its stride is set to, the three gait modulations are wired to the command,
and the baseline gait is untouched by them so iteration 0 really is the
hand-tuned gait.

Terrain and navigation: every course generates something and carries a route
that runs its whole length in order and never passes through a wall, a ramp is
a sustained grade with a banked section somewhere on it, ice is slippery
without being an obstacle, the corridor walls obstruct from both sides and so
does a slalom wall, a disc that only just overlaps a wall is still obstructed,
bearing is zero exactly when the waypoint is dead ahead, waypoints are reached
one at a time and the route never runs out, the forward scan sees a wall three
metres before the feet do and sees the invisible fence too, no foot is ever
planted on top of a wall or inside one, a machine walking into a slalom wall
keeps its feet and chassis out of the block, and — the one that matters — a
machine steering toward its waypoints gets through a slalom that the same
machine walking straight ahead does not.

Hardware: torque scaling with mass and size against a fixed trajectory, a
converging mass/current loop, regulator sizing on output rather than pack
current, bus servos taking an adapter instead of PWM channels, 7.4 V servos
skipping the regulator, and an absurd runtime failing loudly instead of
returning a plausible-looking answer.

`node test/smoke.mjs` — 65 checks in Chromium: the wasm starts, the sim
advances, the stage paints, training improves the reward, the learned policy
becomes selectable, courses switch, the torque requirement responds to mass, the
commanded-speed dial actually changes how fast the robot goes, contact and
actuator telemetry is live, leg weight shows up as joint torque and as a kick
into the chassis, picking an undersized servo drives it past stall and sags the
chassis, and switching to four or ten legs rebuilds the per-leg readouts,
repaints the stage and still walks. Then: every canvas's backing store matches
the box CSS lays it out in, so nothing on the page is drawn stretched; the
footfall recorder holds real samples and its measured cycle, duty and pattern
agree with the gait actually running; every course the simulator knows has a
button; the slalom loads with a route that leaves the centreline; and the
autopilot can be switched off and hands steering back.
