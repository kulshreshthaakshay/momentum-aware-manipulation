# Zero-G Assembly RL — Critical Audit Report

## Executive Summary

The project's core idea is sound: a free-floating 7-DOF manipulator in microgravity controlled by
PPO with curriculum learning and an angular-momentum-aware reward. The architecture — PyBullet,
gymnasium, SB3, VecNormalize, null-space projection — is well-chosen. However, several bugs would
**crash the training or evaluation loop**, and several algorithmic choices **directly conflict**
with the free-floating physics they claim to model. Every file has issues.

---

## 1. Critical Bugs (Will Crash at Runtime)

### 1.1 `evaluate_policy_metrics`: `info` is a list, not a dict — guaranteed AttributeError
**File:** `scripts/evaluation_utils.py`, ~line 160

```python
# CURRENT (BROKEN)
if len(step_res) == 5:
    obs, reward, terminated, truncated, info = step_res
    done = terminated or truncated
    info = info  # ← info is still a list[dict] from VecEnv!

# ... later:
if info.get("momentum_controlled_release", False):  # AttributeError: list has no .get()
```

With SB3 ≥ 2.0 + gymnasium, a `VecEnv.step()` returns 5 elements where `info` is
**`list[dict]`** (one per sub-env). The 5-element branch never does `info = info[0]`, so every
downstream `info.get(...)` call crashes. Since `evaluate_policy_metrics` is called in both
`curriculum_train.py` and `evaluate_policy.py`, **all evaluation is broken**.

**Fix:**
```python
if len(step_res) == 5:
    obs, reward, terminated, truncated, info = step_res
    # VecEnv returns list[dict]; raw env returns dict
    if isinstance(info, list):
        info = info[0]
        reward = float(reward[0]) if hasattr(reward, '__len__') else reward
        terminated = bool(terminated[0]) if hasattr(terminated, '__len__') else terminated
        truncated  = bool(truncated[0])  if hasattr(truncated,  '__len__') else truncated
    done = terminated or truncated
else:
    obs, reward, done, info = step_res
    info   = info[0]
    reward = float(reward[0])
    done   = bool(done[0])
```

---

### 1.2 `evaluate_policy_metrics`: `NameError: truncated` in the `else` branch
**File:** `scripts/evaluation_utils.py`, ~line 170

```python
else:  # 4-element VecEnv return (legacy API)
    obs, reward, done, info = step_res
    info   = info[0]
    reward = reward[0]
    done   = done[0]
    truncated = truncated[0]   # ← truncated is NEVER assigned in this branch!
```

If this branch is ever reached, it raises `NameError`. `truncated` carries over from a
previous *iteration* only if the 5-element branch ran first — a silent, fragile dependency.

**Fix:** Remove the `truncated = truncated[0]` line entirely; `done` already encodes termination.

---

### 1.3 `curriculum_train.py`: Double VecNormalize wrapping when using `--continue_from`
**File:** `scripts/curriculum_train.py`, ~line 110

```python
# train_env is already:  VecNormalize(SubprocVecEnv(...))
train_env = VecNormalize.load(stats_path, train_env)
# Now:  VecNormalize(VecNormalize(SubprocVecEnv(...)))  ← obs normalized TWICE
```

`VecNormalize.load(path, venv)` wraps whatever you pass as `venv` in a *new* VecNormalize.
Passing an already-normalized env means the policy sees doubly-normalized observations, while
training saw singly-normalized ones. The model's weights are useless in this configuration.

**Fix:**
```python
# Build the raw env first, THEN load stats into it
raw_env  = SubprocVecEnv([make_env(stage, config['max_steps']) for _ in range(n_envs)])
train_env = VecNormalize.load(stats_path, raw_env)    # wraps raw_env only
train_env.training = True
train_env.norm_reward = True
model = PPO.load(continue_from, env=train_env, device='cpu')
```

---

### 1.4 `curriculum_train.py`: Lambda closure captures mutable variable
**File:** `scripts/curriculum_train.py`, ~line 120

```python
eval_env_raw = make_env(stage, config['max_steps'], control_mode)()
eval_env = Monitor(eval_env_raw)
eval_env = DummyVecEnv([lambda: eval_env])   # ← captures name, not value
eval_env = VecNormalize(eval_env, ...)
```

`lambda: eval_env` is evaluated lazily. If `DummyVecEnv.__init__` defers the call (or if
reset is called after the outer assignment), `eval_env` refers to the `VecNormalize` object,
causing the lambda to return a VecNormalize when DummyVecEnv expects a plain env, leading to
triple-wrapping or infinite recursion on reset.

**Fix:**
```python
# Capture the value, not the name
_mon = Monitor(make_env(stage, config['max_steps'], control_mode)())
eval_env = DummyVecEnv([lambda e=_mon: e])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                        training=False, clip_obs=10.0)
```

---

## 2. Significant Algorithm Errors

### 2.1 Free-Floating Jacobian is Kinematically Wrong
**File:** `envs/truss_assembly_env.py`, `step()` task-space branch

```python
linear_jacobian, angular_jacobian = p.calculateJacobian(
    self.robot_id, self.ee_link_idx, [0, 0, 0],
    q_movable, dq_movable, zero_acc, ...
)
# Sliced to arm joints only:
J = np.vstack([linear_jacobian, angular_jacobian])[:, :len(self.arm_indices)]
```

`p.calculateJacobian` treats the base as **fixed**. In a free-floating system the base is
not fixed — every arm motion induces equal-and-opposite base motion. The kinematically correct
Jacobian for a free-floater is the *Generalized Jacobian Matrix* (GJM):

```
J_gen = J_arm - J_base * (I_base)^{-1} * J_base^T * J_arm_reac
```

Using the fixed-base Jacobian means:
- Null-space projections do **not** conserve momentum as intended.
- End-effector tracking errors increase because base reaction is unmodelled.
- The momentum penalty reward signal is incoherent with the control law.

**Fix (minimum viable):** Add the base reaction as a feedforward correction:
```python
# After computing target_joint_vels, apply base compensation:
# dq_base_reaction ≈ -I_base^{-1} * J_arm_angular.T @ target_joint_vels
# Then apply via the base torque channel (already separate in action space)
```
Or switch to Joint-space mode (already the default) and rely on learned base compensation.
The joint-space mode is physically simpler and more appropriate for this codebase.

---

### 2.2 Momentum Penalty Scale is Inconsistent with Task Rewards
**File:** `envs/truss_assembly_env.py`, `_compute_reward()`

```python
momentum_penalty = 0.1 * H_sys_norm
```

The base torque action can exert up to `max_torque = 8 N·m`, and the arm has non-trivial
inertia. A `H_sys_norm` of 2–5 kg·m²/s is easily reachable within a few seconds, giving
`momentum_penalty ≈ 0.2–0.5 /step`. Meanwhile progress rewards (`100 * delta_dist`) can
be `+10 to +30 /step` during an approach phase. The momentum penalty is **20–100× too weak**
to actually constrain momentum during task execution. The agent learns to ignore it.

**Fix:** Scale momentum penalty relative to action scales and reward magnitudes:
```python
# Normalise against system inertia characteristic scale
H_sys_norm_normalised = H_sys_norm / 5.0   # 5 kg·m²/s = "high" for this system
momentum_penalty = 2.0 * H_sys_norm_normalised   # Now comparable to progress rewards
```
Also consider a **quadratic** penalty `H_sys_norm ** 2` so small momenta are tolerated
but large ones are strongly penalised, which is physically motivated.

---

### 2.3 EE Link Index Causes Systematic Grasp Distance Error
**File:** `envs/truss_assembly_env.py`, `_create_robot()` and `_handle_gripper()`

```python
self.ee_link_idx = 6  # wrist_link_2
```

From the URDF, `finger_left` and `finger_right` are attached to `wrist_link_2` at
`xyz="0.08 ±0.02 0"`. The actual grasping surface is ~0.08 m ahead of `wrist_link_2`.
Yet all grasp-proximity calculations use the `wrist_link_2` origin:

```python
ee_state = p.getLinkState(self.robot_id, self.ee_link_idx, ...)
ee_pos = np.array(ee_state[0])            # wrist center, NOT finger tips
dist_to_part = np.linalg.norm(ee_pos - np.array(part_pos))
```

With `grasp_distance = 0.25`, the actual fingertip distance at grasp is `0.25 - 0.08 = 0.17 m`
— the fingers may never physically reach the part before the constraint is triggered. Conversely,
the constraint can be created when the gripper is still 8 cm away, causing a violent snap.

**Fix:** Define a synthetic EE frame at the gripper midpoint:
```python
# In _create_robot(), after setting ee_link_idx:
self.gripper_offset = np.array([0.085, 0.0, 0.0])  # local +X offset to finger midpoint

# In _handle_gripper() and _compute_reward():
ee_state  = p.getLinkState(self.robot_id, self.ee_link_idx, ...)
ee_pos_world = np.array(ee_state[0])
ee_orn_world = ee_state[1]
R = np.array(p.getMatrixFromQuaternion(ee_orn_world)).reshape(3,3)
ee_pos = ee_pos_world + R @ self.gripper_offset   # true gripper center
```

---

### 2.4 Null-Space Secondary Task Doesn't Minimise Momentum
**File:** `envs/truss_assembly_env.py`, task-space control branch

```python
null_projector = np.eye(7) - J_pinv @ J
target_joint_vels = J_pinv @ arm_vel + null_projector @ (-0.5 * dq_arm)
```

The null-space term (`-0.5 * dq_arm`) damps joint velocity toward zero. This is a generic
damping objective and has **no relation to angular momentum**. The project's stated goal is
*null-space control for disturbance minimization*, which should minimise `d(H_sys)/dt`
attributable to arm motion.

The correct secondary task for momentum management:
```python
# Gradient of angular momentum w.r.t. joint velocities is the angular Jacobian transposed
# Secondary objective: drive arm joints toward a posture that minimises H contribution
q_mid = np.array([(l + u) / 2.0 for l, u in self.joint_limits])   # neutral posture
q_arm_arr = np.array(q_arm)
posture_gradient = -0.5 * (q_arm_arr - q_mid)   # attract toward neutral, low-momentum posture

# Optionally: additionally damp angular momentum via H
# dq_null = -k_h * J_ang.T @ H_sys  (arm joints that reduce angular momentum)

target_joint_vels = J_pinv @ arm_vel + null_projector @ posture_gradient
```

---

### 2.5 Stage 5 Phase Logic: `was_grasped_before` Off-by-One Step
**File:** `envs/truss_assembly_env.py`, `_compute_reward()` Stage 5

```python
was_grasped_before = self.grasped_part         # snapshot BEFORE update
if self.gripper_closed and not self.grasped_part:
    self.grasped_part = True                   # update NOW

if not was_grasped_before:
    # Phase 1: approach + grasp
    ...
    if dist_to_part < self.grasp_distance:
        if self.gripper_closed:
            self.grasp_hold_steps += 1
            if not self.milestone_first_grasp_ever and self.grasp_hold_steps >= 3:
                reward += MILESTONE_BONUS
                ...
                self.prev_dist_to_goal = dist_part_to_goal   # ← resets goal distance
```

On the step where the gripper first closes AND `was_grasped_before == False`, we are
simultaneously in Phase 1 and also `self.grasped_part = True`. On the **next step**,
`was_grasped_before = True` and we enter Phase 2. This 1-step Phase 1 overlap is harmless,
**except** that `prev_dist_to_goal` is reset inside Phase 1, which then governs Phase 2
progress on the very next step — correct in intent, but the logic is fragile.

Also, `milestone_first_grasp_ever` gates the `prev_dist_to_goal` reset. If the milestone
threshold (3 hold-steps) is never reached, `prev_dist_to_goal` is never reset, and Phase 2
progress rewards use a stale initial distance. This causes the first Phase 2 step to give an
artificially huge (or negative) progress reward.

**Fix:** Decouple the `prev_dist_to_goal` reset from the milestone:
```python
# In reset():
self.prev_dist_to_goal = None

# At the transition from Phase 1 → Phase 2 (first step where was_grasped_before = True):
elif self.gripper_closed:  # Phase 2
    if self.prev_dist_to_goal is None:  # first Phase 2 step
        self.prev_dist_to_goal = dist_part_to_goal
    ...
```

---

### 2.6 Stage 4 Success Distance Inconsistency with Stage 5 Release Zone
**File:** `envs/truss_assembly_env.py`

```python
self.stage4_success_distance = 0.30   # Stage 4 success: part within 30cm of goal
self.at_goal_distance = 0.40          # Stage 5 release trigger: part within 40cm of goal
```

Stage 4 trains the agent to get within 30 cm. Stage 5 then rewards the *holding* penalty
only when within 40 cm. So the agent trained in Stage 4 already knows to get within 30 cm,
but Stage 5's Phase 2 progress reward stops being dense past 40 cm. This creates a 10 cm
"gap" where Stage 4 skill transfers correctly but Stage 5 reward is suboptimal. Worse,
Stage 5's "at release zone" condition:

```python
at_release_zone = (
    dist_part_to_goal < self.at_goal_distance
    or (self.milestone_reached_goal_area and dist_part_to_goal < 0.50)
)
```
can be satisfied at 0.50 m (50 cm) once the milestone is reached. This means the `-2.0`
holding penalty activates 50 cm away from goal, before the agent has finished transporting.

**Fix:** Unify these distances and make the release zone strictly smaller than the transport
arrival zone:
```python
self.transport_arrival_distance = 0.35  # Part must reach within 35cm (Stages 4 & 5)
self.release_distance = 0.35            # Part released within 35cm = success
self.at_goal_distance = 0.25           # Holding penalty activates only when very close
```

---

## 3. Training Pipeline Issues

### 3.1 VecNormalize Stats Drift During Training (eval_env desync)
**File:** `scripts/curriculum_train.py`, training loop

The `eval_callback` uses `eval_env` (a separate VecNormalize). Its `obs_rms` is synced once
before training:
```python
eval_env.obs_rms = train_env.obs_rms
```
But during `model.learn(...)`, `train_env.obs_rms` is continuously updated by new samples.
`eval_env` receives no further updates. By the time the eval callback fires, `eval_env` is
evaluating with stats that can be 25,000+ steps stale, producing systematically inaccurate
evaluation metrics.

**Fix:** Use SB3's built-in mechanism to sync stats in the callback:
```python
class SyncNormCallback(BaseCallback):
    def __init__(self, train_env, eval_env):
        super().__init__()
        self.train_env = train_env
        self.eval_env  = eval_env
    def _on_step(self):
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms
        return True

# Add to callback list alongside eval_callback
```

---

### 3.2 `evaluate_stage()` Creates a Redundant Env (Resource Leak)
**File:** `scripts/curriculum_train.py`, `evaluate_stage()` and training loop

The training loop creates `eval_env_raw` for the `EvalCallback`, then separately calls
`evaluate_stage()` which creates *another* `TrussAssemblyEnv`. Neither version properly
shares VecNormalize stats. The redundant env starts a PyBullet server, uses ~50 MB of RAM,
and never gets the training normalization stats — so its metrics are computed on raw
(unnormalized) observations passed to a policy that expects normalized inputs.

**Fix:** Rewrite `evaluate_stage` to accept the already-normalized VecEnv:
```python
def evaluate_stage(model, norm_eval_env, n_episodes=50):
    """Accepts an already-normalized VecEnv with current stats."""
    norm_eval_env.training = False
    # sync stats from model if possible
    if model.get_vec_normalize_env() is not None:
        sync_stats(model.get_vec_normalize_env(), norm_eval_env)
    return evaluate_policy_metrics(model, norm_eval_env, n_episodes=n_episodes)
```

---

### 3.3 PPO `n_epochs=10` with Large Buffer is Excessive
**File:** `scripts/curriculum_train.py`, `PPO_CONFIG`

```python
PPO_CONFIG = {
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    ...
}
```

With `n_envs=8`: buffer = `8 × 2048 = 16 384` samples. Mini-batches of 256 gives 64 batches/epoch.
At 10 epochs = **640 gradient steps per PPO update**. This is extremely high and causes
severe policy overfitting to each rollout, especially at lower curriculum stages where episode
variance is low. Typical well-tuned PPO uses 3–5 epochs.

**Fix:**
```python
PPO_CONFIG = {
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 4,           # Reduced from 10
    "clip_range": 0.2,       # Tighten slightly (was 0.3) to compensate
    ...
}
```

---

### 3.4 `gamma` Progression is Too Aggressive for Later Stages
**File:** `scripts/curriculum_train.py`, `STAGE_CONFIGS`

```python
# Stage 4: gamma=0.998, Stage 5: gamma=0.999
```

`gamma=0.999` over a 1000-step episode gives an effective horizon of `1/(1-0.999) = 1000`
steps — the full episode. This is mathematically correct for a long-horizon task but makes
the value function extremely difficult to fit, causing high variance and slow convergence.
For a task where most reward comes in the final 100 steps (transport + release), a horizon
of 200–300 steps is sufficient.

**Fix:**
```python
4: {"gamma": 0.995, ...},
5: {"gamma": 0.997, ...},
```

---

## 4. Observation & Reward Design Issues

### 4.1 `robot_pos` in Observation Only for Stage 1 — Breaks Transfer
**File:** `envs/truss_assembly_env.py`, `_get_obs()`

```python
if self.curriculum_stage == 1:
    robot_pos_obs = np.clip(np.array(robot_pos) / 5.0, -1.0, 1.0)
else:
    robot_pos_obs = np.zeros(3)   # Zeros for stages 2–5
```

This means the **observation space changes semantically between stages**. The VecNormalize
running mean/std for indices 40–42 will diverge: Stage 1 sees real position values there,
Stages 2–5 see zeros. When the policy trained on Stage 1 is transferred to Stage 2, those
units are repurposed as constant zeros, but the normalizer has built statistics around them
being position values. The policy's weights for those inputs produce undefined behaviour
after transfer.

**Fix:** Remove the stage-conditional behaviour entirely. Station keeping (Stage 1) can use
`ee_to_part` (which encodes distance to the part, effectively tracking origin position since
the part is fixed at [1,0,0]) and `robot_lin_vel`. Absolute robot position is unnecessary.

---

### 4.2 Contact Force Observation is Unnormalised and Unbounded
**File:** `envs/truss_assembly_env.py`, `_get_obs()`

```python
contact_force = list(constraint_state[:3])  # raw Newtons, can be 0–500+
```

Contact forces from PyBullet constraints are in raw Newtons and can be very large when the
arm is under load. The observation vector contains these unbounded values at indices 31–33.
VecNormalize will eventually learn the running statistics, but in early training the network
sees extreme inputs that cause gradient explosions.

**Fix:** Clip and scale:
```python
max_contact_force = 100.0  # N, typical max for this system
contact_force = np.clip(contact_force, -max_contact_force, max_contact_force)
contact_force = (np.array(contact_force) / max_contact_force).tolist()
```

---

### 4.3 `H_sys` Observation Not Normalised
**File:** `envs/truss_assembly_env.py`, `_get_obs()`

```python
self._cached_H_sys,           # raw kg·m²/s, can be 0–20+
[self._cached_H_sys_norm],    # raw scalar
```

The angular momentum vector and its norm are passed raw. Like the contact force, this leads
to large inputs in early training. VecNormalize helps, but the issue is more subtle: the
OBS_LAYOUT exposes `h_sys` (indices 36–38) and `h_sys_norm` (index 39) separately. Normalising
both redundantly means the network gets correlated features.

**Fix:** Only include the normalised scalar and drop the raw 3-vector, or normalise manually:
```python
H_scale = 5.0   # kg·m²/s characteristic scale for this system
h_obs = np.clip(self._cached_H_sys / H_scale, -3.0, 3.0)
h_norm_obs = np.clip(self._cached_H_sys_norm / H_scale, 0.0, 3.0)
```
(and update `OBS_LAYOUT` accordingly if dropping the vector).

---

### 4.4 Stage 2 Progress Reward Scale is Mismatched Relative to Stage 3
**File:** `envs/truss_assembly_env.py`, `_compute_reward()`

```python
# Stage 2:
reward += 100.0 * progress

# Stage 3:
reward += 50.0 * progress
```

Stage 2 (approach) rewards distance reduction at 2× the scale of Stage 3 (grasp), even
though Stage 3 requires the same approach behaviour PLUS the grasp. This makes Stage 3
feel worse than Stage 2 for the same approach motion. The agent may unlearn the approach
policy during Stage 3 warm-up.

**Fix:** Unify progress scales: `75.0` for both, or keep Stage 2 at 100 and Stage 3 at 100,
offset by the grasp bonus:
```python
# Stage 3:
reward += 100.0 * progress   # Same as Stage 2
# (grasp bonuses are already separate)
```

---

## 5. URDF Issues

### 5.1 `wrist_link_0` Has No Collision Geometry
**File:** `assets/zero_g_servicer.urdf`

`wrist_link_0` (connected via `wrist_yaw` joint) has only a visual shape and no `<collision>`
element. This means the wrist yaw link passes through objects without interaction. The
`elbow_link` has a collision element, and `finger_left`/`finger_right` have collision elements,
but the wrist chain has gaps.

**Fix:** Add collision shapes to `wrist_link_0` and `wrist_link_1` matching their visual geometry.

---

### 5.2 `shoulder_link_2` and `shoulder_link_3` Missing Collision Geometry
**File:** `assets/zero_g_servicer.urdf`

Both links have visual geometry but no `<collision>` block. For a free-floating manipulator
in contact-rich assembly, missing collision on upper arm links means the arm can interpenetrate
the payload or the servicer body silently.

---

### 5.3 Base Inertia Comment is Wrong
**File:** `assets/zero_g_servicer.urdf`

```xml
<!-- Inertia for 60kg box 0.5^3: I = m/12 * (w^2 + h^2) = 5 * (0.25+0.25) = 2.5 -->
```

The formula `m/12 * (w² + h²)` is correct for a solid box, giving:
`60/12 * (0.5² + 0.5²) = 5 * 0.5 = 2.5 kg·m²`. ✓

But the comment says `(0.25+0.25)` where it should say `(0.25+0.25) = 0.5²+0.5²`. Minor
documentation error, does not affect the value.

---

## 6. Evaluation Script Issues

### 6.1 `evaluate_policy.py`: GIF Capture Overrides Aggregate Metrics Environment
**File:** `scripts/evaluate_policy.py`

```python
metrics = evaluate_policy_metrics(model, env, n_episodes=n_episodes, deterministic=True)
# ...
if args.save_gif:
    for i in range(min(n_episodes, 5)):
        reset_res = env.reset()
```

After `evaluate_policy_metrics` runs, `env` (a VecNormalize wrapping `raw_env`) is used
for GIF capture. But `raw_env.render()` is called for frames. When `env` is a VecEnv, its
`reset()` returns batched observations; the `step_res` unpacking logic below it handles
both 4 and 5 element returns, but `done` is handled as a scalar without proper indexing:

```python
while not (done or truncated):  # truncated may be undefined (VecEnv case)
```

This is the same `truncated` undefined-variable bug from issue 1.2 reappearing.

**Fix:** Unify the GIF capture loop to always use the raw (non-VecEnv) environment directly
to avoid all VecEnv unpacking complexity:
```python
if args.save_gif:
    gif_env = TrussAssemblyEnv(render_mode="rgb_array",
                               curriculum_stage=args.stage,
                               control_mode=args.control_mode)
    # wrap gif_env with VecNormalize + loaded stats for correct observation normalization
    gif_venv = DummyVecEnv([lambda: gif_env])
    gif_venv = VecNormalize.load(stats_path, gif_venv)
    gif_venv.training = False
    # ... run episode using gif_venv, render via gif_env.render()
```

---

### 6.2 `continue_training.py`: `evaluate()` Creates New Env Without Normalization Stats
**File:** `scripts/continue_training.py`, `evaluate()` function

```python
def evaluate(model, stage, max_steps, n_episodes=50):
    env = TrussAssemblyEnv(curriculum_stage=stage, max_steps=max_steps)
    if hasattr(model, "get_vec_normalize_env") and model.get_vec_normalize_env() is not None:
        stats_env = model.get_vec_normalize_env()
        eval_env = DummyVecEnv([lambda: env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = stats_env.obs_rms
```

`PPO.load()` does **not** retain a reference to the VecNormalize env unless you pass
`env=train_env` *and* save the `_vecnorm.pkl` separately. After a fresh `PPO.load()`,
`model.get_vec_normalize_env()` returns `None` unless the model was saved with the env
reference. This means the `if` branch (with normalization) often won't run, and evaluation
proceeds on raw observations — the model will act randomly.

**Fix:** Always load the normalization stats explicitly:
```python
stats_path = model_path.replace(".zip", "_vecnorm.pkl")
eval_env = DummyVecEnv([lambda: TrussAssemblyEnv(curriculum_stage=stage, max_steps=max_steps)])
if os.path.exists(stats_path):
    eval_env = VecNormalize.load(stats_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
```

---

## 7. Minor / Code Quality Issues

| # | Location | Issue |
|---|----------|-------|
| 7.1 | `truss_assembly_env.py` | `steps_at_goal_holding` is initialised in `reset()` and guarded by `hasattr` in `_compute_reward`, but never incremented anywhere. Dead code. |
| 7.2 | `truss_assembly_env.py` | Comment `# NOTE: ... line 608` is stale. Line numbers in comments become wrong immediately after any edit. Remove or use function-relative references. |
| 7.3 | `curriculum_train.py` | `make_env()` already wraps in `Monitor`. The training loop adds a second `Monitor` around `eval_env_raw`. Double-Monitor causes double episode counting in the EvalCallback. |
| 7.4 | `curriculum_train.py` | `eval_deterministic = stage >= 4`. For Stage 3 (grasp), stochastic evaluation is fine, but for Stage 2 the agent is exploratory enough that deterministic evaluation gives a pessimistic success rate. Consider `stage >= 3`. |
| 7.5 | `evaluation_utils.py` | `GRIPPER_STATE`, `DIST_TO_PART`, `DIST_TO_GOAL` are defined as integer indices (tuple[0]) but used as array indices. This works but is fragile if OBS_LAYOUT is changed to non-consecutive slices. Use `slice(*OBS_LAYOUT[...])` consistently. |
| 7.6 | `truss_assembly_env.py` | `_compute_system_angular_momentum()` iterates `_movable_indices` which includes both arm joints (7) and gripper joints (2). Gripper mass is 0.05 kg each — their contribution is ~1% of total momentum. Worth skipping for performance (called every step). |
| 7.7 | `truss_assembly_env.py` | `p.disconnect()` → `p.connect()` on every `reset()` is expensive. For training with SubprocVecEnv (one env per process), this is fine. For DummyVecEnv or `if __name__ == "__main__"` test, this adds ~200ms overhead per episode. Consider lazy reconnection. |
| 7.8 | `curriculum_train.py` | `assert (n_envs * PPO_CONFIG["n_steps"]) % PPO_CONFIG["batch_size"] == 0` is inside the stage loop but only catches the error at runtime. Move to script startup. |