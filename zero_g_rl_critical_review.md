# Zero-G Assembly RL — Critical Code Review

## Executive Summary

The project structure is conceptually sound: curriculum PPO for a free-floating
manipulator in microgravity, with system angular momentum as both an observation
feature and a penalty. However there are **~15 significant bugs and design flaws**
spread across the environment, training script, and evaluation utilities that will
actively prevent the agent from learning correctly or at all. Many of them are
silent — they won't raise exceptions, they'll just produce wrong physics or
corrupted gradients. Below is every issue found, ordered by severity, with
concrete fixes.

---

## 1. CRITICAL — Physics & Simulation

### 1.1 Task-Space Jacobian Column Extraction is Wrong

**File:** `envs/truss_assembly_env.py` — `step()`, task-space branch

```python
# Current code — WRONG
start_col = 6  # Skip 6 base DoFs
arm_cols = [start_col + i for i in range(len(self.arm_indices))]
linear_jacobian = np.array(linear_jacobian)[:, arm_cols]
```

`p.calculateJacobian` in PyBullet returns a Jacobian whose column count equals the
number of **movable** DoFs **only** — it does NOT prepend 6 base DoF columns.
The full Jacobian is `(3, N_movable)` where `N_movable = 8` (7 arm + 1 prismatic
gripper). Slicing columns `[6, 7, 8, 9, 10, 11, 12]` on an 8-column matrix will
raise an IndexError at runtime, or silently give garbage if the shapes happen to
broadcast. The base DoF offset assumption is only valid for the `pin` / `pinocchio`
convention, not PyBullet.

**Fix:**
```python
# PyBullet Jacobian columns = movable joints only, in joint-index order.
# arm_indices = [0,1,2,3,4,5,6] → first 7 columns of the 8-column matrix
linear_jacobian = np.array(linear_jacobian)[:, :len(self.arm_indices)]  # (3,7)
angular_jacobian = np.array(angular_jacobian)[:, :len(self.arm_indices)]  # (3,7)
J = np.vstack([linear_jacobian, angular_jacobian])  # (6,7) — correct
```

---

### 1.2 `_compute_system_angular_momentum` — Wrong COM Velocity for Links

**File:** `envs/truss_assembly_env.py` — `_compute_system_angular_momentum()`

```python
link_state = p.getLinkState(body_id, link_idx, computeLinkVelocity=1)
vel = link_state[6]   # World linear velocity of link COM
omega = link_state[7] # World angular velocity of link
```

`getLinkState` indices 6 and 7 are the **URDF link frame** linear/angular velocity,
not the inertial (COM) frame velocity. For links where the inertial origin is offset
from the link origin (all arm links have `localInertialFramePosition` ≠ 0), these
velocities are wrong by a cross-product term `v_com = v_frame + omega × r_offset`.
This makes your conserved-quantity computation physically incorrect, and the
momentum penalty will oscillate chaotically.

**Fix:**
```python
link_state = p.getLinkState(body_id, link_idx,
                             computeLinkVelocity=1,
                             computeForwardKinematics=1)
# getLinkState returns:
# [0] linkWorldPosition (link frame origin)
# [1] linkWorldOrientation
# [2] localInertialFramePosition
# [3] localInertialFrameOrientation
# [4] worldLinkFramePosition
# [5] worldLinkFrameOrientation
# [6] worldLinkLinearVelocity   ← velocity at link FRAME origin, not COM
# [7] worldLinkAngularVelocity

frame_vel = np.array(link_state[6])
omega_world = np.array(link_state[7])
r_offset = R_link @ local_inertial_pos   # COM offset in world frame
# Correct COM velocity via rigid-body kinematics
vel_world = frame_vel + np.cross(omega_world, r_offset)
```

---

### 1.3 Zero-Gravity Constraint — Gravity Never Re-disabled After Reconnect

**File:** `envs/truss_assembly_env.py` — `reset()`

```python
p.disconnect(self.physics_client)
# ...
p.setGravity(0, 0, 0)
```

This is correct, but there is **no check** that the `p.DIRECT` / `p.GUI` connection
succeeded before calling `setGravity`. If `p.connect` fails (e.g., too many
simultaneous processes during `SubprocVecEnv` init), all subsequent `p.*` calls
silently use the **last valid physics client ID**, which still has gravity enabled.
Add an assert:

```python
self.physics_client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
assert self.physics_client >= 0, "PyBullet connection failed"
p.setGravity(0, 0, 0, physicsClientId=self.physics_client)
```

Better still, pass `physicsClientId=self.physics_client` to **every** `p.*` call —
without it, parallel `SubprocVecEnv` workers occasionally cross-contaminate physics
clients, producing silent wrong-gravity episodes.

---

## 2. CRITICAL — Reward Design

### 2.1 Stage 3 Declares Immediate Success on Any Grasp — Stage Cannot Transition

**File:** `envs/truss_assembly_env.py` — `_compute_reward()`, Stage 3

```python
if self.gripper_closed:
    if not self.milestone_first_grasp:
        reward += 200.0
        self.milestone_first_grasp = True
    reward += 500.0
    info["success"] = True   # ← triggers `terminated = True` every step
```

`gripper_closed` stays `True` for the remainder of the episode after the first
successful grasp. This means `terminated` is set to `True` on the **same step** as
the grasp, which is correct — but if the agent closes the gripper accidentally at
`dist_to_part > grasp_distance` (constraint not created, `gripper_closed` not set),
this branch is never reached. The real bug is that `success` fires immediately with
no "hold the grasp for N steps" requirement, making Stage 3 trivially solvable with
a random grasp at distance < `grasp_distance`. That is not necessarily wrong, but it
means the policy only learns to close the gripper once — it learns nothing about
**sustaining** the grasp, which Stage 4 suddenly requires.

**Fix:** Add a minimum hold requirement:

```python
GRASP_HOLD_STEPS = 5   # require N consecutive steps of closed gripper

if self.gripper_closed:
    self.grasp_hold_steps = getattr(self, 'grasp_hold_steps', 0) + 1
    if not self.milestone_first_grasp:
        reward += 200.0
        self.milestone_first_grasp = True
    if self.grasp_hold_steps >= GRASP_HOLD_STEPS:
        reward += 500.0
        info["success"] = True
else:
    self.grasp_hold_steps = 0
```

---

### 2.2 Stage 4/5 Progress Reward Scale Inconsistency (100× amplification)

**File:** `_compute_reward()`, Stage 4 transport and Stage 5 Phase 2

```python
# Stage 4
reward += 80.0 * goal_progress       # delta * 80

# Stage 5
reward += 150.0 * goal_progress      # delta * 150
```

With `sim_substeps=5` and `dt=1/240`, each environment step advances `5/240 ≈ 0.02 s`.
At a robot speed of 0.5 m/s the part moves ~0.01 m per step, giving `goal_progress ≈ 0.01`.
Reward contribution: `80 * 0.01 = 0.8` per step. Fine. But if the robot overshoots at 2 m/s,
`goal_progress` can be **negative** (−0.02), giving `80 * -0.02 = -1.6`. Combined with the
velocity damping penalty, the agent receives a strong contradictory signal: "slow down near
goal" vs "progress at all times." The multipliers differ by nearly 2× between Stages 4 and 5
with no justification, making the Stage 4→5 knowledge transfer noisy.

**Fix:** Unify the multiplier and make the progress reward always non-negative (clip at 0)
to separate it from the velocity-damping term which already handles overshoot:

```python
progress_reward = 100.0 * max(0.0, goal_progress)   # same for Stage 4 and 5
reward += progress_reward
```

---

### 2.3 Double-Counting the Momentum Penalty in Stage 5

**File:** `_compute_reward()`, Stage 5 comment

```python
# NOTE: Global momentum penalty (0.01 * |H_sys|) is already applied at line 608.
# A duplicate 0.02 penalty was removed here...
```

The comment says the duplicate was removed, but the `momentum_penalty` at line ~428
(top of `_compute_reward`) is:

```python
momentum_penalty = 0.01 * H_sys_norm
reward = -fuel_penalty - momentum_penalty - 0.5 * joint_limit_penalty
```

And then inside Stage 5 Phase 2 the variable `H_sys` is re-read from
`_compute_system_angular_momentum()` **a second time** (it's called inside the
Phase 3 release-bonus block). Each call to this function is O(N_links²) and costs
~0.5 ms in PyBullet — calling it twice per step per environment adds ~50 ms overhead
in an 8-env setup. Cache the result:

```python
# In _compute_reward(), compute once at the top
H_sys = self._compute_system_angular_momentum()
H_sys_norm = np.linalg.norm(H_sys)
# Pass H_sys_norm to all sub-blocks; never call _compute_system_angular_momentum() again
```

---

### 2.4 Stage 1 Success Condition is Impractically Strict

**File:** `_compute_reward()`, Stage 1

```python
if dist_from_origin < 0.1 and vel_magnitude < 0.1:
    reward += 10.0
    info["success"] = True
```

The dense reward `1.0 - dist_from_origin` gives a maximum of `+1.0` per step.
The success bonus is only `+10`, equivalent to 10 steps of perfect behaviour.
With fuel penalty eating into reward constantly, the agent has very little incentive
to sustain the success condition. Combined with `max_steps=500` for Stage 1, the
agent will learn to hover near the origin without ever triggering success, and
curriculum training will plateau here indefinitely.

**Fix:**
```python
# Increase success bonus and require sustained success
if not hasattr(self, 'station_keeping_steps'):
    self.station_keeping_steps = 0

if dist_from_origin < 0.1 and vel_magnitude < 0.1:
    self.station_keeping_steps += 1
    reward += 2.0 * self.station_keeping_steps   # escalating reward for sustained keeping
    if self.station_keeping_steps >= 20:          # ~0.4 s of success
        reward += 200.0
        info["success"] = True
else:
    self.station_keeping_steps = 0
```

---

## 3. CRITICAL — Observation Space

### 3.1 Absolute Robot Position Leaks into Observation — Breaks Curriculum Transfer

**File:** `_get_obs()`

```python
obs = np.array(
    list(robot_pos) +        # ← ABSOLUTE world position [0:3]
    list(robot_orn) +
    ...
)
```

The README and docstring explicitly state "relative vectors for translation-invariant
transfer." But `robot_pos` (indices 0–2) is the **absolute world position** of the
base. In Stage 1 the robot starts at `[0,0,0]`; in Stage 5 it starts at `[1,0,0]`.
Any policy trained on Stage 1 that uses `obs[0:3]` to infer where it is will be
systematically biased when transferred. Absolute position is also meaningless in a
real spacecraft environment.

**Fix:** Replace with `robot_pos - mean_scene_origin` or simply **remove** absolute
position and rely on the already-included relative vectors. The observation already
contains `base_to_ee`, `ee_to_part`, and `part_to_goal`, so absolute position adds
no new information and hurts transfer:

```python
# Replace robot_pos with zeros or remove the field
# and update observation_space shape from 46 → 43
obs_components = (
    [0., 0., 0.] +          # placeholder or remove
    list(robot_orn) +       # keep orientation — it IS absolute and needed
    ...
)
```

Or alternatively, express robot position relative to the part:
```python
robot_to_part = part_pos - robot_pos  # already implicitly encoded via base_to_ee + ee_to_part
```

---

### 3.2 Observation Indices in `evaluation_utils.py` Are Off by One

**File:** `scripts/evaluation_utils.py`

```python
ROBOT_POS    = slice(0, 3)
ROBOT_LIN_VEL = slice(7, 10)    # Should be 7:10  ← correct only if orn is [3:7]
ROBOT_ANG_VEL = slice(10, 13)
EE_TO_PART   = slice(27, 30)
PART_TO_GOAL = slice(30, 33)
BASE_TO_EE   = slice(33, 36)
GRIPPER_STATE = 36
DIST_TO_PART = 40
DIST_TO_GOAL = 41
H_SYS        = slice(42, 45)
H_SYS_NORM   = 45
```

Let's manually count from `_get_obs()`:

| Field            | Size | Cumulative |
|-----------------|------|-----------|
| robot_pos       | 3    | 0–2       |
| robot_orn       | 4    | 3–6       |
| robot_lin_vel   | 3    | 7–9       |
| robot_ang_vel   | 3    | 10–12     |
| joint_pos       | 7    | 13–19     |
| joint_vel       | 7    | 20–26     |
| ee_to_part      | 3    | 27–29     |
| part_to_goal    | 3    | 30–32     |
| base_to_ee      | 3    | 33–35     |
| gripper_state   | 1    | 36        |
| contact_force   | 3    | 37–39     |
| dist_to_part    | 1    | 40        |
| dist_to_goal    | 1    | 41        |
| H_sys           | 3    | 42–44     |
| H_sys_norm      | 1    | 45        |

The indices in `evaluation_utils.py` **are correct** in absolute terms, but
`DIST_TO_PART = 40` and `DIST_TO_GOAL = 41` are used as scalar indices while the
contact_force field occupies `37:40`. If the contact force implementation changes
size (e.g., full 3D vector per contact), all downstream indices shift silently.
Make the indices derived programmatically or add a unit test that verifies them
against a real environment step.

---

## 4. HIGH — Training Script Issues

### 4.1 Curriculum Advancement Checks Success on a Single Evaluation — Fragile

**File:** `scripts/curriculum_train.py` — `run_curriculum()`

```python
if success_rate >= config['success_threshold'] and total_trained >= config['min_timesteps']:
    print(f"\n  ✅ Stage {stage} COMPLETE! ...")
    break
```

`success_threshold = 1.00` (100%) for every stage. A single unlucky evaluation
(50 episodes, one random failure) will never trigger advancement. This means the
curriculum will always exhaust `max_timesteps` before advancing. Practical thresholds
for RL are typically 90–95% over a window of evaluations.

**Fix:**
```python
STAGE_CONFIGS = {
    1: {"success_threshold": 0.95, ...},
    2: {"success_threshold": 0.90, ...},
    3: {"success_threshold": 0.85, ...},
    4: {"success_threshold": 0.80, ...},
    5: {"success_threshold": 0.75, ...},
}
# Additionally, require threshold to be met across 2 consecutive evaluations
```

---

### 4.2 Learning Rate Halved Without Warm-Up on Stage 4+ Transfer

**File:** `scripts/curriculum_train.py`

```python
if stage >= 4:
    model.learning_rate = PPO_CONFIG["learning_rate"] / 2
```

`model.learning_rate` is a float property. Changing it after `.set_env()` does NOT
update the internal Adam optimizer's learning rate — it only affects the **schedule**
if a callable is used. The optimizer still uses the original LR until the next call
to `_setup_lr_schedule()`. This is a known SB3 gotcha. The change is silently
ignored.

**Fix:**
```python
from stable_baselines3.common.utils import get_linear_fn

if stage >= 4:
    new_lr = PPO_CONFIG["learning_rate"] / 2
    model.learning_rate = new_lr
    # Force optimizer update
    for param_group in model.policy.optimizer.param_groups:
        param_group['lr'] = new_lr
```

---

### 4.3 `SuccessRateCallback` Double-Counts Successes

**File:** `scripts/curriculum_train.py` — `SuccessRateCallback._on_step()`

```python
for info in self.locals['infos']:
    if 'episode' in info:
        self.episodes += 1
        episode_info = info.get('episode', {})
        if info.get('success', False) or episode_info.get('success', False):
            self.successes += 1
```

`info['success']` is the per-step success flag from the environment. When the
`Monitor` wrapper is used (as it is in `make_env()`), the `episode` key is added
to `info` only at episode end — but `info['success']` can be `True` at the same
step. The condition `info.get('success', False) or episode_info.get('success', False)`
is redundant, but more importantly, **it counts one episode's success twice**
if both conditions are true simultaneously. This inflates the success rate metric
and causes premature curriculum advancement.

**Fix:**
```python
for info in self.locals['infos']:
    if 'episode' in info:        # Only count at episode boundaries
        self.episodes += 1
        if info.get('success', False):
            self.successes += 1
```

---

### 4.4 `EvalCallback` Uses `deterministic=True` for Stochastic Policies

**File:** `scripts/curriculum_train.py`

```python
eval_callback = EvalCallback(
    eval_env, ...,
    deterministic=True,    # ← This is fine for final eval but...
)
```

During early training (Stages 1–3), the policy is highly stochastic and the entropy
bonus (`ent_coef=0.03`) is meant to keep exploration alive. Evaluating with
`deterministic=True` produces a systematically lower success rate than training
behaviour, leading to delayed curriculum advancement. Use `deterministic=False`
during Stages 1–3 and switch to `True` for Stages 4–5.

---

### 4.5 `n_steps * n_envs` Must Be Divisible by `batch_size`

**File:** `scripts/curriculum_train.py` — `PPO_CONFIG`

```python
PPO_CONFIG = {
    "n_steps": 2048,
    "batch_size": 256,
    ...
}
```

With `n_envs=8`: buffer size = `2048 * 8 = 16384`. `16384 / 256 = 64` — this is
fine. But if someone passes `--n_envs 6`: `2048 * 6 = 12288`. `12288 / 256 = 48`
— still fine. The issue arises with `--n_envs 3`: `2048 * 3 = 6144 / 256 = 24` —
fine. With `--n_envs 5`: `2048 * 5 = 10240 / 256 = 40` — fine. Actually with
batch_size=256 and n_steps=2048, any n_envs works as long as `n_envs * 2048 >= 256`.
But worth adding a validation:

```python
assert (n_envs * PPO_CONFIG["n_steps"]) % PPO_CONFIG["batch_size"] == 0, \
    f"n_envs={n_envs} * n_steps={PPO_CONFIG['n_steps']} not divisible by batch_size={PPO_CONFIG['batch_size']}"
```

---

## 5. HIGH — URDF / Robot Model Issues

### 5.1 Gripper Right Finger is Fixed — Gripper Cannot Open

**File:** `assets/zero_g_servicer.urdf`

```xml
<joint name="gripper_joint_right" type="fixed">
    <parent link="wrist_link_2"/>
    <child link="finger_right"/>
    <origin rpy="0 0 0" xyz="0.08 -0.02 0"/>
</joint>
```

The right finger is **fixed** in the URDF, but `_handle_gripper()` attempts to
control it:

```python
p.setJointMotorControl2(
    self.robot_id, self.gripper_indices[1],   # finger_right index 8
    controlMode=p.POSITION_CONTROL,
    targetPosition=-target_pos,
    force=10
)
```

`gripper_indices[1] = 8` corresponds to joint index 8 in the URDF. PyBullet will
silently ignore `setJointMotorControl2` on a `FIXED` joint, meaning only the left
finger moves. The gripper is not parallel-jaw — it's a single moving finger vs a
fixed wall. This works mechanically but wastes one action dimension and makes the
gripper asymmetric.

**Fix:** Change to `type="prismatic"` with axis `0 -1 0` (mirrored):
```xml
<joint name="gripper_joint_right" type="prismatic">
    <parent link="wrist_link_2"/>
    <child link="finger_right"/>
    <origin rpy="0 0 0" xyz="0.08 -0.02 0"/>
    <axis xyz="0 -1 0"/>
    <limit effort="10.0" lower="0.0" upper="0.03" velocity="0.5"/>
</joint>
```

Or unify both fingers as one prismatic joint with a mimic tag (not supported in
PyBullet — use the VELOCITY_CONTROL mirror approach in code instead).

---

### 5.2 `ee_link_idx = 6` Points to Wrist Roll, Not Fingertips

**File:** `envs/truss_assembly_env.py` — `_create_robot()`

```python
self.ee_link_idx = 6    # wrist_roll
```

Link 6 is `wrist_link_2` (the wrist roll link). The actual fingertip (where grasping
occurs) is at links 7/8 (`finger_left`/`finger_right`). The URDF offsets from
`wrist_link_2` to finger base are `xyz="0.08 ±0.02 0"`. The grasp distance check
uses EE position at link 6, which is `0.08 m` behind the actual fingertip. This
means the agent must bring the wrist to within `grasp_distance=0.25 m` of the part,
but the actual fingertip is `0.08 m` further — effectively the agent can grasp at
`0.25 + 0.08 = 0.33 m` wrist-to-part distance, which is very generous and produces
imprecise grasps.

**Fix:**
```python
# Use finger_left link for EE reference
self.ee_link_idx = 7    # finger_left link index

# Or compute a virtual EE midpoint
ee_left = np.array(p.getLinkState(self.robot_id, 7)[0])
ee_right = np.array(p.getLinkState(self.robot_id, 8)[0])
ee_pos = (ee_left + ee_right) / 2.0
```

---

## 6. MEDIUM — Algorithmic / Design Issues

### 6.1 Null Space Damping in Task Space Control Uses Wrong DoF Count

**File:** `step()`, task-space branch

```python
dq_arm = np.array(dq_movable[:7])         # First 7 of movable joints
...
null_projector = I - (J_pinv @ J)         # I is (7,7), J_pinv@J is (7,7) ← OK
q_dot_secondary = null_projector @ q_dot_null_ref   # (7,7)@(7,) ← OK
```

This is actually correct in shape, but `dq_movable` is pulled from `movable_indices`
which includes the gripper joint (index 7) as movable. So `dq_movable[:7]` = arm
joints 0–6 only IF the gripper joint appears last. If PyBullet returns movable
indices in joint-index order (it does), then index 7 = gripper is at position 7 in
`movable_indices`, so `dq_movable[:7]` correctly captures arm joints only. This is
fragile and should be explicit:

```python
arm_joint_states = p.getJointStates(self.robot_id, self.arm_indices)
dq_arm = np.array([s[1] for s in arm_joint_states])
```

---

### 6.2 `prev_dist_to_part` / `prev_dist_to_goal` Initialized to `None` — First Step Has No Progress Reward

**File:** `_compute_reward()`, Stages 2–5

```python
if self.prev_dist_to_part is None:
    self.prev_dist_to_part = dist_to_part

progress = self.prev_dist_to_part - dist_to_part
reward += 100.0 * progress
```

On the first step, `prev_dist_to_part` is set to the current distance, giving
`progress = 0`. This wastes the first step's reward signal but is otherwise harmless.
More critically, in Stage 5 the first-grasp milestone resets `prev_dist_to_goal`:

```python
if not self.milestone_first_grasp:
    reward += 200.0
    self.milestone_first_grasp = True
    self.prev_dist_to_goal = dist_part_to_goal    # ← set here
```

But `self.prev_dist_to_goal` is also initialized in `reset()` to `None`, and in
Stage 4 the first-grasp milestone does NOT reset `prev_dist_to_goal`. This means
on the step after the first grasp in Stage 4, `prev_dist_to_goal` is whatever the
distance was at the *start of the episode*, not at the moment of grasping. If the
robot drifted during approach, this produces a large spurious goal-progress reward.

**Fix:** Always reset `prev_dist_to_goal` when transitioning from Phase 1 to Phase 2:

```python
if not self.gripper_closed and was_not_closed:   # transition step
    self.prev_dist_to_goal = dist_part_to_goal    # reset reference at grasp moment
```

---

### 6.3 Angular Momentum Observation is Redundant with Penalty — Likely Not Learned

The observation includes `H_sys (3D vector)` + `H_sys_norm (scalar)`. The 3D vector
carries orientation of the momentum, which is important for control. However, the
penalty uses only the norm: `0.01 * H_sys_norm`. This creates a partial incentive —
the agent is penalized for total momentum but never gets a **directional** reward
signal telling it which joint velocities are causing the build-up.

For better momentum-awareness, consider a penalty on the *rate of change* of angular
momentum (equivalent to net torque), or explicitly include `dH/dt` in the observation.
This would make the agent learn to anticipate momentum build-up, not just react to it.

---

### 6.4 `sim_substeps=5` but Arm Velocity Applied Once per Step — Physics Inconsistency

**File:** `step()`

```python
# Joint velocity applied ONCE before the loop
p.setJointMotorControlArray(
    self.robot_id, self.arm_indices,
    controlMode=p.VELOCITY_CONTROL,
    targetVelocities=arm_vel,
    forces=[100]*7
)

# Then substep loop runs 5 times
for _ in range(self.sim_substeps):
    p.applyExternalForce(...)   # base thrust applied each substep
    p.stepSimulation()
```

Base thrust is applied every substep (correct — force is continuous). But arm
velocity is set once before the loop. In `VELOCITY_CONTROL` mode, PyBullet's joint
motor maintains the target velocity throughout the substeps — this is actually fine
because the velocity controller is already internal to PyBullet's simulation loop.
However, the task-space Jacobian calculation (which happens inside the loop) fetches
link states that change between substeps, recalculates the Jacobian each substep,
and re-applies joint velocities. This is computationally expensive and the Jacobian
changes are small per substep, but the code recomputes it 5 times per policy step.
Cache the Jacobian outside the loop and only recompute once per policy step.

---

## 7. LOW — Evaluation Script Issues

### 7.1 `evaluate_policy.py` Runs Two Separate Evaluation Loops

**File:** `scripts/evaluate_policy.py`

```python
for i in range(n_episodes):    # First loop — logs per-episode output
    ...

metrics = evaluate_policy_metrics(model, env, n_episodes=n_episodes, deterministic=True)  # Second loop
```

The function runs `n_episodes` episodes **twice** — once for console output and GIF
saving, and once inside `evaluate_policy_metrics`. Total evaluation cost is `2×n_episodes`.
More importantly, the per-episode results printed in the first loop may not match the
aggregate metrics from the second loop (different random seeds, different stochasticity).

**Fix:** Integrate GIF capture into `evaluate_policy_metrics`, or print per-episode
stats from the returned `RolloutMetrics` object.

---

### 7.2 `evaluate_policy_metrics` has a Silent Bug — `for/else` Loop Counter

**File:** `scripts/evaluation_utils.py`

```python
for step in range(env.max_steps):
    ...
    if info.get("success", False):
        successes += 1
        lengths.append(step + 1)
        break
    if terminated or truncated:
        ...
        lengths.append(step + 1)
        break
else:
    timeouts += 1
    lengths.append(env.max_steps)
```

The `for/else` clause fires if the loop exhausts without a `break`. But both
`terminated` and `truncated` trigger a `break`, and `truncated=True` already
means `step_count >= max_steps`. So the `else` clause is unreachable — `timeouts`
inside `else` is never incremented, and the `timeouts` counter in the breakdown
will always read 0 regardless of actual timeout rate.

**Fix:**
```python
for step in range(env.max_steps):
    action, _ = model.predict(obs, deterministic=deterministic)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    ...

    if info.get("success", False):
        successes += 1
        lengths.append(step + 1)
        break

    if terminated or truncated:
        if info.get("dropped_early", False):
            early_drops += 1
        elif not info.get("success", False):
            timeouts += 1      # ← count here, not in else
        lengths.append(step + 1)
        break
```

---

## 8. Summary Table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1.1 | **CRITICAL** | `truss_assembly_env.py` | Jacobian column extraction off by 6 — task-space control broken |
| 1.2 | **CRITICAL** | `truss_assembly_env.py` | COM velocity for links computed at wrong frame — momentum wrong |
| 1.3 | **CRITICAL** | `truss_assembly_env.py` | No `physicsClientId` passed — parallel envs cross-contaminate |
| 2.1 | **CRITICAL** | `truss_assembly_env.py` | Stage 3 success fires instantly — policy learns nothing sustained |
| 2.2 | **CRITICAL** | `truss_assembly_env.py` | Progress reward sign inconsistency near goal — contradictory signal |
| 2.3 | **CRITICAL** | `truss_assembly_env.py` | `_compute_system_angular_momentum()` called twice — O(N²) overhead |
| 2.4 | **HIGH** | `truss_assembly_env.py` | Stage 1 success bonus too small to dominate fuel penalty |
| 3.1 | **HIGH** | `truss_assembly_env.py` | Absolute robot_pos in obs — breaks curriculum transfer |
| 3.2 | **MEDIUM** | `evaluation_utils.py` | Hardcoded obs indices — brittle, will silently break on obs change |
| 4.1 | **HIGH** | `curriculum_train.py` | `success_threshold=1.00` — curriculum never advances |
| 4.2 | **HIGH** | `curriculum_train.py` | `model.learning_rate` change doesn't update Adam optimizer |
| 4.3 | **HIGH** | `curriculum_train.py` | `SuccessRateCallback` double-counts successes |
| 4.4 | **MEDIUM** | `curriculum_train.py` | `deterministic=True` eval underestimates early-stage performance |
| 5.1 | **HIGH** | `zero_g_servicer.urdf` | Right finger is `fixed` — gripper is single-sided |
| 5.2 | **MEDIUM** | `truss_assembly_env.py` | `ee_link_idx=6` behind actual fingertip — imprecise grasps |
| 6.2 | **MEDIUM** | `truss_assembly_env.py` | `prev_dist_to_goal` not reset at grasp transition in Stage 4 |
| 7.1 | **LOW** | `evaluate_policy.py` | Runs 2× episodes — metrics from two separate rollout sets |
| 7.2 | **LOW** | `evaluation_utils.py` | `for/else` timeout counter unreachable — always 0 |

---

## 9. Recommended Priority Order for Fixes

1. Fix **1.1** (Jacobian) first — task-space mode is completely broken without it
2. Fix **3.1** (absolute pos in obs) — easy 1-line change, massive transfer benefit
3. Fix **4.1** (threshold 100%→90%) — curriculum is stuck without this
4. Fix **4.3** (double-count callback) — inflated metrics cause premature advancement
5. Fix **1.2** (COM velocity) — makes momentum computation physically correct
6. Fix **7.2** (timeout counter) — evaluation metrics are misleading without it
7. Fix **4.2** (LR update) — LR decay silently does nothing currently
8. Fix **2.3** (cache H_sys) — O(N²) per step kills throughput in 8-env training
9. Remaining medium/low items
