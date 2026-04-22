# Zero-G Assembly RL — Final Validation Report (Round 3)

## Score Card

| Category | Count |
|---|---|
| ✅ Confirmed Correct | 21 |
| 🔴 Bugs (will crash or corrupt training) | 7 |
| ⚠️ Warnings (degrade quality silently) | 2 |

---

## 🔴 Bug 1 — CRASH: `physicsClientId` Passed to Pure Math Functions

**File:** `envs/truss_assembly_env.py` — `_handle_gripper()` and `_compute_system_angular_momentum()`

When adding `physicsClientId` everywhere, it was also applied to PyBullet utility
functions that are pure math operations with **no physics client context**. These
will raise `TypeError` at runtime immediately:

```python
# _handle_gripper() — WILL CRASH:
ee_inv_pos, ee_inv_orn = p.invertTransform(ee_pos.tolist(), ee_orn,
                                            physicsClientId=self.physics_client)  # ← TypeError
local_pos, local_orn = p.multiplyTransforms(...,
                                            physicsClientId=self.physics_client)  # ← TypeError

# _compute_system_angular_momentum() — WILL CRASH every call:
R_link = np.array(p.getMatrixFromQuaternion(orn,
                   physicsClientId=self.physics_client)).reshape(3, 3)  # ← TypeError
R_inertial = np.array(p.getMatrixFromQuaternion(local_inertial_orn,
                       physicsClientId=self.physics_client)).reshape(3, 3)  # ← TypeError
```

`p.invertTransform`, `p.multiplyTransforms`, and `p.getMatrixFromQuaternion` are
stateless math utilities. They do not interact with any physics simulation.
PyBullet's C binding will immediately raise `TypeError: unexpected keyword argument
'physicsClientId'`.

**Fix — remove `physicsClientId` from exactly these calls:**
```python
# _handle_gripper():
ee_inv_pos, ee_inv_orn = p.invertTransform(ee_pos.tolist(), ee_orn)
local_pos, local_orn = p.multiplyTransforms(ee_inv_pos, ee_inv_orn,
                                             list(part_pos), list(part_orn))

# _compute_system_angular_momentum():
R_link = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
R_inertial = np.array(p.getMatrixFromQuaternion(local_inertial_orn)).reshape(3, 3)
```

---

## 🔴 Bug 2 — Two Remaining `p.*` Calls Missing `physicsClientId`

**File:** `envs/truss_assembly_env.py`

After the broad `physicsClientId` sweep, two calls were missed:

**In `_handle_gripper()`:**
```python
ee_state = p.getLinkState(self.robot_id, self.ee_link_idx)  # ← missing physicsClientId
```

**In `_compute_reward()`:**
```python
joint_states_for_limits = p.getJointStates(self.robot_id, self.arm_indices)  # ← missing
```

**Fix:**
```python
# _handle_gripper():
ee_state = p.getLinkState(self.robot_id, self.ee_link_idx,
                           physicsClientId=self.physics_client)

# _compute_reward():
joint_states_for_limits = p.getJointStates(self.robot_id, self.arm_indices,
                                            physicsClientId=self.physics_client)
```

---

## 🔴 Bug 3 — Stale `_get_obs()` Docstring (Shape Mismatch Documentation)

**File:** `envs/truss_assembly_env.py` — `_get_obs()` docstring

The docstring was never updated after removing `robot_pos (3)` and `base_to_ee (3)`:

```python
def _get_obs(self):
    """
    Robot: pos(3), orn(4), lin_vel(3), ang_vel(3) = 13   ← WRONG (pos removed → 10)
    Arm: joint_pos(7), joint_vel(7) = 14
    Relative: ee_to_part(3), part_to_goal(3), base_to_ee(3) = 9  ← WRONG (base_to_ee removed → 6)
    ...
    Total = 13 + 14 + 9 + 1 + 3 + 2 + 4 = 46             ← WRONG (actual = 40)
    """
```

The `observation_space` correctly declares `shape=(40,)`, so training won't crash, but the
docstring actively misleads anyone reading the code. It's especially dangerous because
`evaluation_utils.py` hardcodes obs indices — any contributor who trusts the docstring
to add a new field will get the indices wrong.

**Fix — update docstring to match reality:**
```python
"""
Robot (10): orn(4), lin_vel(3), ang_vel(3)
Arm   (14): joint_pos(7), joint_vel(7)
Relative(6): ee_to_part(3), part_to_goal(3)
Gripper (1): gripper_state(1)
Contact (3): contact_force(3)
Dist    (2): dist_to_part(1), dist_to_goal(1)
Momentum(4): H_sys(3), ||H_sys||(1)
Total = 10 + 14 + 6 + 1 + 3 + 2 + 4 = 40
"""
```

---

## 🔴 Bug 4 — Stage 4: `grasp_hold_steps` Never Reset on Gripper Open

**File:** `envs/truss_assembly_env.py` — `_compute_reward()`, Stage 4

When the agent is in Stage 4 transport phase and opens the gripper (dropping the part),
the code enters the `if not self.gripper_closed` approach branch. That branch never
resets `grasp_hold_steps`. If the agent then re-closes the gripper, the hold counter
continues from where it left off — so a non-consecutive grasp pattern triggers the
milestone:

```python
# Stage 4 approach branch — gripper OPEN:
if not self.gripper_closed:
    progress = self.prev_dist_to_part - dist_to_part
    reward += 50.0 * progress
    ...
    self.prev_dist_to_part = dist_to_part
    # grasp_hold_steps is NEVER reset here!

# Stage 4 transport branch — gripper CLOSED:
else:
    self.grasp_hold_steps += 1
    if not self.milestone_first_grasp and self.grasp_hold_steps >= 3:
        reward += 200.0   # ← fires on accumulated, non-consecutive count
```

**Fix:**
```python
if not self.gripper_closed:
    self.grasp_hold_steps = 0   # ← add this
    progress = self.prev_dist_to_part - dist_to_part
    ...
```

---

## 🔴 Bug 5 — Stage 5 Re-grasp: `prev_dist_to_goal` Not Reset

**File:** `envs/truss_assembly_env.py` — `_compute_reward()`, Stage 5

The attempt to fix this (Minor-4) used the wrong condition:

```python
# Current fix attempt:
if self.grasped_part and self.gripper_closed and not was_grasped_before:
    self.prev_dist_to_goal = dist_part_to_goal
```

On a **re-grasp** scenario (grasp → release → re-grasp), when re-grasping:
- `was_grasped_before = self.grasped_part = True` (latched from first grasp)
- `not was_grasped_before = False`
- Condition fails → `prev_dist_to_goal` is NOT reset

The correct approach is to track the gripper state transition explicitly:

```python
# In reset():
self._prev_gripper_closed = False

# In _compute_reward(), Stage 5, at the top of the stage block:
gripper_just_closed = self.gripper_closed and not self._prev_gripper_closed
if gripper_just_closed and self.grasped_part:
    self.prev_dist_to_goal = dist_part_to_goal   # reset on every grasp transition

# At the end of _compute_reward():
self._prev_gripper_closed = self.gripper_closed
```

---

## 🔴 Bug 6 — Dead Computation in `_get_obs()`

**File:** `envs/truss_assembly_env.py` — `_get_obs()`

`base_to_ee` and `robot_pos_arr` are computed but never used in the observation:

```python
robot_pos_arr = np.array(robot_pos)      # ← fetched but only used for base_to_ee
ee_to_part = part_pos - ee_pos           # used ✓
part_to_goal = goal_pos - part_pos       # used ✓
base_to_ee = ee_pos - robot_pos_arr      # ← computed but NOT in obs array
```

The `obs` array no longer includes `base_to_ee`. This is dead code that wastes
one array allocation per step. It also signals incomplete cleanup.

**Fix:** Remove these two lines entirely:
```python
# DELETE:
robot_pos_arr = np.array(robot_pos)
base_to_ee = ee_pos - robot_pos_arr
```

---

## 🔴 Bug 7 — Stage 1 Escalating Reward Spike Disrupts PPO Value Estimates

**File:** `envs/truss_assembly_env.py` — `_compute_reward()`, Stage 1

The escalating station-keeping reward creates an extreme per-step spike:

```python
if dist_from_origin < 0.1 and vel_magnitude < 0.1:
    self.station_keeping_steps += 1
    reward += 2.0 * self.station_keeping_steps   # step 20: +40
    if self.station_keeping_steps >= 20:
        reward += 200.0   # +200 at step 20 → total spike ~241 in ONE step
```

Normal per-step reward is ~0 to ~1. A single step with reward ~241 will cause the
GAE advantage estimate to spike catastrophically. PPO's clipping is on the policy
ratio, not on reward magnitude — PPO has no built-in protection against this.
The value network will be unable to predict this value, causing large gradient
updates and policy degradation ("reward hacking" or collapse).

**Fix — cap the escalating component and separate the terminal bonus:**
```python
if dist_from_origin < 0.1 and vel_magnitude < 0.1:
    self.station_keeping_steps += 1
    # Capped escalating reward: max +5/step at N=5+
    reward += min(1.0 * self.station_keeping_steps, 5.0)
    if self.station_keeping_steps >= 20:
        reward += 50.0   # Reduced but still significant success bonus
        info["success"] = True
```

Alternatively, normalize all stage rewards to the same scale (~0 to ~5/step) as a
general principle across all stages. Currently Stage 3 has a +200 milestone bonus
and Stage 5 has a +600 terminal bonus — these are very large relative to the
continuous shaping signals.

---

## ⚠️ Warning 1 — `base_to_ee` and `robot_pos_arr` Are Dead Variables

(Covered in Bug 6 above — included here as a warning because it doesn't crash,
it just wastes allocations and signals incomplete cleanup.)

---

## ⚠️ Warning 2 — `evaluate_policy.py` Does Not Wrap Env in `MomentumMonitor`

**File:** `scripts/evaluate_policy.py`

```python
env = TrussAssemblyEnv(render_mode="rgb_array", curriculum_stage=args.stage, ...)
metrics = evaluate_policy_metrics(model, env, ...)
```

`evaluate_policy_metrics` reads `max_h` directly from `obs[H_SYS_NORM]` (which
works correctly without the wrapper), but the `metrics.mean_max_momentum` field
computed this way correctly tracks the episode max. The `MomentumMonitor` wrapper
is only needed for TensorBoard logging during training. Evaluation is fine.

No code change needed — just document why the wrapper is absent here.

---

## ✅ Everything Confirmed Correct (21 Items)

| Item | Status |
|------|--------|
| obs shape = 40, matches `observation_space` | ✅ |
| All 11 obs index constants in `evaluation_utils.py` | ✅ All correct |
| `_compute_system_angular_momentum` COM velocity fix (frame→COM) | ✅ |
| Base link COM velocity correction in same function | ✅ |
| `_cached_H_sys` shared between `_get_obs()` and `_compute_reward()` | ✅ |
| Jacobian out of substep loop, computed once per policy step | ✅ |
| Jacobian column slice `[:, :7]` (not `[:, 6:13]`) | ✅ |
| `_movable_indices` cached in `_create_robot()` | ✅ |
| `_handle_gripper()` inside substep loop | ✅ |
| Right finger `targetPosition=target_pos` (sign fix) | ✅ |
| `ee_link_idx = 7` (finger_left) | ✅ |
| All `_create_robot()` calls pass `physicsClientId` | ✅ |
| All `_create_part()` / `_create_goal()` calls pass `physicsClientId` | ✅ |
| `continue_training.py` LR update patches `param_groups` | ✅ |
| `SuccessRateCallback` no double-counting | ✅ |
| `success_threshold` = 0.95→0.75 (not 1.00) | ✅ |
| Consecutive evaluation requirement (2×) before advancing | ✅ |
| `MomentumMonitor` wrapper tracks episode max correctly | ✅ |
| `MomentumMonitor → Monitor` wrapping order correct | ✅ |
| `evaluate_stage` uses unwrapped env, H_SYS_NORM tracked from obs | ✅ |
| `timeouts` counter in reachable branch of `evaluation_utils` | ✅ |

---

## Final Project Assessment

### What Is Now Fully Sound

The architecture is clean and the implementation has converged on a good state.
The core contributions of the project — momentum-aware observation, physically
correct H_sys computation, null-space redundancy resolution, and curriculum
progression — are all correctly implemented after the previous two rounds of fixes.

The observation space is now properly translation-invariant (40D, no absolute
position). The `physicsClientId` propagation is ~95% complete. The `MomentumMonitor`
wrapper solves the TensorBoard logging problem correctly. The curriculum advancement
logic (2 consecutive evals, realistic thresholds) is solid.

### What Must Be Fixed Before Running

**Priority 1 — Will crash immediately:**
- Bug 1: Remove `physicsClientId` from `invertTransform`, `multiplyTransforms`, `getMatrixFromQuaternion`

**Priority 2 — Silent corruption:**
- Bug 2: Add `physicsClientId` to the two remaining `p.*` calls
- Bug 4: Reset `grasp_hold_steps` when gripper opens in Stage 4
- Bug 7: Cap the Station Keeping escalating reward spike

**Priority 3 — Minor cleanup:**
- Bug 3: Update `_get_obs()` docstring
- Bug 5: Fix Stage 5 re-grasp `prev_dist_to_goal` reset logic
- Bug 6: Remove dead `base_to_ee` / `robot_pos_arr` variables

### One Remaining Architectural Note

The contact force observation (`contact_force[0:3]`) is still asymmetric — it only
reports the total normal force as the X component with Y=Z=0. For a 7-DOF arm
manipulating in 3D space, contact forces along all three axes carry useful
information for the policy (especially for the release phase). If you want to
improve Phase 3 performance in Stage 5, make this a real 3-axis measurement:

```python
if contact_points:
    total_force = np.zeros(3)
    for cp in contact_points:
        normal = np.array(cp[7])           # contact normal direction (world frame)
        force_magnitude = cp[9]            # normal force magnitude
        total_force += force_magnitude * normal
    contact_force = total_force.tolist()
```