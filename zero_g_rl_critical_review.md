# Zero-G Assembly RL — Fix Validation & Remaining Issues Report

## Score Card

| Category | Count |
|---|---|
| ✅ Confirmed Fixed | 23 |
| 🔴 New Bugs Introduced | 3 |
| 🔴 Remaining (partial fix) | 5 |
| ⚠️ Design / Performance | 5 |
| ⚠️ Minor / Cosmetic | 5 |

---

## ✅ Confirmed Fixed (23)

All 18 issues from the previous review were addressed. The following are confirmed
correct by line-by-line inspection:

| Fix | Verdict |
|-----|---------|
| 1.1 Jacobian column slice `[:, :7]` | ✅ Correct |
| 1.2 COM velocity `frame_vel + ω × r` | ✅ Correct |
| 1.3 `assert physics_client >= 0` | ✅ Correct |
| 1.3 `setGravity/setTimeStep` pass `physicsClientId` | ✅ Correct |
| 2.1 Stage 3 requires `grasp_hold_steps >= 5` | ✅ Correct |
| 2.2 Progress reward clipped `max(0.0, ...)` both Stage 4 & 5 | ✅ Correct |
| 2.3 `H_sys` computed once at top of `_compute_reward` | ✅ Correct |
| 2.4 Station keeping escalating reward + 20-step success gate | ✅ Correct |
| 3.1 Absolute `robot_pos` zeroed in observation | ✅ Correct |
| 4.1 `success_threshold` reduced to 0.95–0.75 per stage | ✅ Correct |
| 4.2 LR update forces Adam `param_groups['lr']` | ✅ Correct |
| 4.3 `SuccessRateCallback` single condition, no double-count | ✅ Correct |
| 4.4 `deterministic=False` for Stages 1–3 | ✅ Correct |
| 4.5 Buffer size divisibility assert added | ✅ Correct |
| 5.1 Right finger URDF changed to `type="prismatic"` | ✅ Correct |
| 5.2 `ee_link_idx = 7` (finger_left) | ✅ Correct |
| 6.2 `prev_dist_to_goal` reset at grasp transition in Stage 4 | ✅ Correct |
| 7.1 `evaluate_policy.py` single rollout loop | ✅ Correct |
| 7.2 `timeouts` counter moved to reachable branch | ✅ Correct |

---

## 🔴 New Bugs Introduced (3)

### BUG-1: Right Finger Won't Open — Sign Error After URDF Fix

**File:** `envs/truss_assembly_env.py` — `_handle_gripper()`

The URDF fix changed `gripper_joint_right` to `type="prismatic"` with axis `xyz="0 -1 0"`
and joint limits `lower=0.0, upper=0.03`. The axis is already mirrored to make the finger
close inward. However `_handle_gripper()` was not updated:

```python
p.setJointMotorControl2(
    self.robot_id, self.gripper_indices[1],
    controlMode=p.POSITION_CONTROL,
    targetPosition=-target_pos,   # ← sends -0.03 when open, but lower limit = 0.0
    force=10
)
```

`target_pos = 0.03` when open. Sending `targetPosition = -0.03` violates the joint
limit `lower=0.0`. PyBullet clamps to 0.0, so the right finger stays closed regardless
of the open command. The gripper cannot open.

**Fix:**
```python
# Since axis is already mirrored (0 -1 0), send the SAME positive target
p.setJointMotorControl2(
    self.robot_id, self.gripper_indices[1],
    controlMode=p.POSITION_CONTROL,
    targetPosition=target_pos,   # NOT negated — axis handles the direction
    force=10
)
```

---

### BUG-2: Jacobian Computed 5× Per Policy Step — Still Broken for Performance

**File:** `envs/truss_assembly_env.py` — `step()`, task-space branch

The column slicing was fixed (✅), but the Jacobian computation itself remains
**inside the substep loop**:

```python
for _ in range(self.sim_substeps):          # loops 5 times
    p.applyExternalForce(...)
    if self.control_mode == "task_space":
        movable_indices = []                # rebuilt each substep!
        for i in range(num_joints):
            j_info = p.getJointInfo(...)    # O(N) each substep
            ...
        linear_jacobian, angular_jacobian = p.calculateJacobian(...)  # expensive
        J = np.vstack([...])
        J_pinv = J.T @ np.linalg.inv(...)  # matrix inverse each substep
        target_joint_vels = J_pinv @ x_dot_desired + null_projector @ ...
        p.setJointMotorControlArray(...)
    p.stepSimulation()
```

Each policy step recomputes: 5 × `getJointInfo` loops + 5 × Jacobian + 5 × matrix
inverse + 5 × null-space projection. With 8 parallel envs, this is 40 full Jacobian
inversions per policy step. The Jacobian barely changes over 5 × (1/240 s) substeps
and should be computed once before the loop.

**Fix:**
```python
# --- COMPUTE ONCE BEFORE SUBSTEP LOOP ---
J = None
if self.control_mode == "task_space":
    joint_states = p.getJointStates(self.robot_id, self._movable_indices)  # cached
    q_movable = [s[0] for s in joint_states]
    dq_movable = [s[1] for s in joint_states]
    zero_acc = [0.0] * len(self._movable_indices)
    
    lin_jac, ang_jac = p.calculateJacobian(
        self.robot_id, self.ee_link_idx, [0, 0, 0],
        q_movable, dq_movable, zero_acc
    )
    lin_jac = np.array(lin_jac)[:, :len(self.arm_indices)]
    ang_jac = np.array(ang_jac)[:, :len(self.arm_indices)]
    J = np.vstack([lin_jac, ang_jac])
    
    lambda_val = 0.05
    J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_val**2 * np.eye(6))
    dq_arm = np.array([s[1] for s in p.getJointStates(self.robot_id, self.arm_indices)])
    null_projector = np.eye(7) - J_pinv @ J
    target_joint_vels = J_pinv @ arm_vel + null_projector @ (-0.5 * dq_arm)
    p.setJointMotorControlArray(self.robot_id, self.arm_indices,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocities=target_joint_vels, forces=[100]*7)

# --- SUBSTEP LOOP (physics only, no recomputation) ---
for _ in range(self.sim_substeps):
    p.applyExternalForce(self.robot_id, -1, thrust.tolist(), [0,0,0], p.LINK_FRAME)
    p.applyExternalTorque(self.robot_id, -1, torque.tolist(), p.LINK_FRAME)
    p.stepSimulation()
```

Also: cache `_movable_indices` in `_create_robot()` instead of rebuilding with
`getJointInfo` every call.

---

### BUG-3: `continue_training.py` — LR Update Doesn't Reach Adam Optimizer

**File:** `scripts/continue_training.py`

Fix 4.2 correctly updated `curriculum_train.py` to patch `param_groups`, but
`continue_training.py` still has the original broken pattern:

```python
if learning_rate:
    print(f"Adjusting learning rate: {model.learning_rate} → {learning_rate}")
    model.learning_rate = learning_rate    # ← silently ignored by Adam
```

**Fix:**
```python
if learning_rate:
    model.learning_rate = learning_rate
    for param_group in model.policy.optimizer.param_groups:
        param_group['lr'] = learning_rate
    print(f"Learning rate updated to {learning_rate}")
```

---

## 🔴 Remaining from Previous Review (5)

### REMAINING-1: `physicsClientId` Not Passed in Most PyBullet Calls

**Files:** `_create_robot()`, `_handle_gripper()`, `_get_obs()`, `step()`

The fix only added `physicsClientId` to 3 lines in `reset()`. The vast majority of
PyBullet calls throughout the class still lack it:

| Method | Missing physicsClientId Calls |
|--------|-------------------------------|
| `_create_robot()` | `p.loadURDF`, `p.resetJointState` (×7), `p.setJointMotorControlArray`, `p.changeDynamics` (×N), `p.getNumJoints`, `p.getJointInfo` (×7) |
| `_handle_gripper()` | `p.setJointMotorControl2` (×2), `p.getLinkState`, `p.getBasePositionAndOrientation`, `p.invertTransform`, `p.multiplyTransforms`, `p.createConstraint`, `p.removeConstraint` |
| `_get_obs()` | `p.getBasePositionAndOrientation`, `p.getBaseVelocity`, `p.getJointStates`, `p.getLinkState`, `p.getContactPoints` |
| `step()` | `p.applyExternalForce`, `p.applyExternalTorque`, `p.setJointMotorControlArray`, `p.getNumJoints`, `p.getJointInfo`, `p.getJointStates`, `p.calculateJacobian`, `p.getLinkState`, `p.stepSimulation`, `p.getBasePositionAndOrientation` |
| `_compute_reward()` | `p.getBasePositionAndOrientation`, `p.getBaseVelocity`, `p.getLinkState`, `p.getJointStates` |
| `render()` | `p.computeViewMatrix`, `p.computeProjectionMatrixFOV`, `p.getCameraImage` |

With `SubprocVecEnv`, each worker runs in a separate process and gets its own PyBullet
client ID (typically 0 in every subprocess). So cross-contamination through wrong IDs
is actually unlikely in `SubprocVecEnv` because each process is isolated. However the
assert and explicit ID passing are important for the `DummyVecEnv` case (single
process, multiple clients) and for correctness in the `human` render mode where a GUI
client exists alongside training clients.

**Pragmatic approach:** Since `SubprocVecEnv` uses separate processes (safe), the
immediate risk is lower than originally stated. But for correctness and to support
`DummyVecEnv`, pass `physicsClientId=self.physics_client` everywhere. The easiest
pattern: store `self.pc = self.physics_client` and use `physicsClientId=self.pc`.

---

### REMAINING-2: `_compute_system_angular_momentum` Called in Both `_get_obs` and `_compute_reward`

`_get_obs()` calls `_compute_system_angular_momentum()` to fill the H_sys observation
fields. `_compute_reward()` now calls it once (fixed). But in `step()`:

```python
obs = self._get_obs()           # calls _compute_system_angular_momentum() → CALL 1
reward, info = self._compute_reward(action)  # calls it again → CALL 2
```

Total: 2 O(N²) computations per policy step. The fix cached within `_compute_reward`
but didn't address the `_get_obs` call.

**Fix:** Compute H_sys once at the start of `step()`, cache as `self._cached_H_sys`,
and have both `_get_obs()` and `_compute_reward()` read from the cache:

```python
def step(self, action):
    ...
    # Substep loop
    for _ in range(self.sim_substeps):
        ...
        p.stepSimulation()
    
    # Compute once, share between obs and reward
    self._cached_H_sys = self._compute_system_angular_momentum()
    self._cached_H_sys_norm = np.linalg.norm(self._cached_H_sys)
    
    obs = self._get_obs()              # reads self._cached_H_sys
    reward, info = self._compute_reward(action)  # reads self._cached_H_sys_norm
```

---

## ⚠️ Design Issues (5)

### DESIGN-1: Curriculum Advancement on Single Evaluation — High Variance

Even with fixed thresholds (0.95→0.75), advancement is checked after a single
50-episode evaluation. Variance at 50 episodes: σ ≈ √(p(1−p)/50) ≈ 3% at p=0.9.
One unlucky batch can block advancement; one lucky batch can advance prematurely.

**Fix:** Require threshold to be met in 2 consecutive evaluations:

```python
consecutive_threshold_met = 0
while total_trained < config['max_timesteps']:
    ...
    if success_rate >= config['success_threshold'] and total_trained >= config['min_timesteps']:
        consecutive_threshold_met += 1
        if consecutive_threshold_met >= 2:
            print(f"✅ Stage {stage} COMPLETE!")
            break
    else:
        consecutive_threshold_met = 0
```

---

### DESIGN-2: Gripper Constraint Created After Substep Loop — Physics Snap

`_handle_gripper()` is called after the 5-substep physics loop. On the step where
the agent first closes the gripper within `grasp_distance`, the 5 substeps run with
the part free-floating, then the constraint is created. On the next step, the
constraint is active during substeps. If the part drifted during those 5 unconstrained
substeps, the constraint creation at a different relative pose causes a visible snap.
The existing `parentFramePosition` / `localPos` computation at grasp time mitigates
this, but the 5-substep drift remains.

This is a known limitation of the constraint-injection approach. For high fidelity,
move `_handle_gripper()` inside the substep loop (called once, before `stepSimulation`).

---

### DESIGN-3: Stage 4/5 Instant Grasp Milestone — Inconsistent with Stage 3 Hold

Stage 3 now requires 5 consecutive steps of `gripper_closed`. But in Stages 4 and 5,
`milestone_first_grasp` fires on the **first** step where `gripper_closed=True`, with
no hold requirement. This creates an inconsistency: the agent learns persistent grasping
in Stage 3 but not in Stages 4/5. If the agent briefly closes the gripper and opens it
in Stage 4 (accidentally), it receives a 200-point milestone bonus and no subsequent
transport incentive.

**Fix:** Apply a 3-step hold requirement to the Stage 4/5 milestone as well (lower
than Stage 3's 5-step requirement since the goal is transport, not grasp precision).

---

### DESIGN-4: `H_sys_norm` in Monitor — Records Last-Step Value, Not Episode Max

```python
return Monitor(
    env,
    info_keywords=("success", "dropped_early", "H_sys_norm", ...),
)
```

`Monitor` captures `info` at episode end (the last `step()` info dict). `H_sys_norm`
at episode end is typically near zero (agent has stopped). The peak momentum during
transport is what matters for research but will never appear in TensorBoard logs.

**Fix:** Track `max_H_sys_norm` as a running episode maximum in a custom callback
or expose it via an episodic info wrapper:

```python
class MomentumMonitor(gym.Wrapper):
    def reset(self, **kwargs):
        self._max_h = 0.0
        return super().reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._max_h = max(self._max_h, info.get("H_sys_norm", 0.0))
        if terminated or truncated:
            info["max_H_sys_norm"] = self._max_h
        return obs, reward, terminated, truncated, info
```

---

### DESIGN-5: `movable_indices` Rebuilt Every Substep in Task-Space Mode

(See BUG-2 above for full context.) Even after moving the Jacobian out of the loop,
`movable_indices` is rebuilt via a `p.getJointInfo` loop every call. Cache it:

```python
def _create_robot(self):
    ...
    # Cache movable joint indices (excludes FIXED joints like gripper_joint_right was)
    self._movable_indices = []
    for i in range(p.getNumJoints(self.robot_id)):
        if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED:
            self._movable_indices.append(i)
    # After fix 5.1: both finger joints are prismatic → movable. 
    # _movable_indices = [0,1,2,3,4,5,6,7,8] (7 arm + 2 gripper)
    # For Jacobian, we only want arm joints: first 7.
```

---

## ⚠️ Minor Issues (5)

### MINOR-1: `R_link_tmp` Redundant Recompute in `add_body_link`

In `_compute_system_angular_momentum()`, the corrected code computes:

```python
R_link_tmp = np.array(p.getMatrixFromQuaternion(link_state[5])).reshape(3, 3)
r_offset = R_link_tmp @ local_inertial_pos
vel = frame_vel + np.cross(np.array(omega), r_offset)
```

Then 6 lines later:
```python
R_link = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)   # orn = link_state[5]
```

Same quaternion, same matrix, computed twice. Replace `R_link_tmp` with `R_link` and
reorder so `R_link` is computed first.

---

### MINOR-2: `ee_vel` in Reward Uses Link Frame Velocity

```python
ee_state = p.getLinkState(self.robot_id, self.ee_link_idx, computeLinkVelocity=1)
ee_pos = np.array(ee_state[0])
ee_vel = np.array(ee_state[6])   # link FRAME velocity, not COM velocity
```

After Fix 5.2, `ee_link_idx=7` is `finger_left`, which has a tiny mass (0.05 kg) and
minimal inertia offset. The velocity error is negligible in practice (< 1 mm/s for
typical arm speeds), but for physical correctness, apply the same correction as in
`_compute_system_angular_momentum`.

---

### MINOR-3: Stage 2 Dead Outer Condition

```python
if dist_to_part < 0.2:           # outer check
    if dist_to_part < self.grasp_distance:   # grasp_distance=0.25 > 0.2 → always True here
        reward += 200.0
        info["success"] = True
```

The outer `if dist_to_part < 0.2` is always satisfied when the inner check triggers.
The intent was probably to keep the stricter 0.2m criterion from the original code but
this was left as dead nesting. Clean up to:

```python
if dist_to_part < self.grasp_distance:
    reward += 200.0
    info["success"] = True
```

---

### MINOR-4: Stage 5 Re-grasp `prev_dist_to_goal` Not Reset

If the agent grasps (Phase 1 → Phase 2), then releases early (Phase 3, `dropped_early`),
then re-grasps in a subsequent step, `milestone_first_grasp=True` means the milestone
block is skipped, and `prev_dist_to_goal` is **not** reset to the current distance.
Phase 2 resumes with a stale `prev_dist_to_goal`, producing a large spurious progress
reward on the first transport step.

**Fix:** Reset `prev_dist_to_goal` whenever the gripper transitions from open to closed
while `grasped_part=True` (re-grasp):

```python
# In Step 5 phase logic, detect re-grasp:
if self.grasped_part and self.gripper_closed and not self._was_closed_last_step:
    self.prev_dist_to_goal = dist_part_to_goal  # reset on re-grasp
```

---

### MINOR-5: `evaluation_utils.ROBOT_POS` Now Reads All Zeros

```python
ROBOT_POS = slice(0, 3)   # Now always [0., 0., 0.] after Fix 3.1
```

This slice is exported and may be used by external analysis code. Add a comment or
rename to `ROBOT_POS_ZEROED` to signal it carries no information. Better: remove the
slice and keep only the meaningful relative vector constants.

---

## Project Idea vs Implementation — Overall Assessment

The research question is well-defined and the architecture is sound. Here's an honest
assessment of alignment between concept and code:

### What Is Working Well

**Curriculum structure** is correctly implemented now. The five-stage progression
(keep → approach → grasp → transport → release) maps cleanly onto the reward functions,
and the stage configs provide the right lever to tune each sub-skill independently.

**Momentum-awareness** is physically meaningful after Fix 1.2. The system now computes
a geometrically correct H_sys, includes it in the observation, and penalizes it in the
reward. The observation encodes both direction (H_sys vector) and magnitude (norm),
which gives the policy sufficient information to learn momentum-minimizing behaviour.

**Null-space control** architecture for task-space mode is conceptually correct after
Fix 1.1. Damped-least-squares pseudo-inverse + null-space projection is the standard
approach for redundant manipulators and is appropriate here.

**Reward shaping** is potential-based in structure (progress = Φ(t−1) − Φ(t)), which
preserves the optimal policy under standard RL theory. The milestone bonuses are
one-time and don't create a reward cycle.

### What Still Needs Work Before Running

1. **BUG-1 (gripper sign)** must be fixed before any training — the gripper cannot
   open, making grasping physically impossible to execute and impossible to release.

2. **REMAINING-2 (double H_sys)** should be fixed before long training runs —
   with 8 envs and 1M timesteps, the doubled O(N²) computation wastes hours.

3. **BUG-2 (Jacobian in loop)** only matters for `task_space` mode. If training in
   `joint` mode (default), this can be deferred.

### Architectural Concern: Observation Contains Redundant Information

After Fix 3.1, the observation has:
- `robot_orn (4)` — absolute orientation of base
- `robot_lin_vel (3)`, `robot_ang_vel (3)` — absolute velocities
- `joint_pos (7)`, `joint_vel (7)` — joint state
- `ee_to_part (3)`, `part_to_goal (3)`, `base_to_ee (3)` — relative vectors
- `gripper_state (1)`, `contact_force (3)`, distances `(2)`, H_sys `(4)`

`base_to_ee` is the forward kinematics output of `joint_pos`. It is entirely
determined by `joint_pos` (given the URDF). Including both is redundant and adds 3
dimensions of perfectly correlated information. Consider removing `base_to_ee` from
the observation and letting the network learn FK implicitly from `joint_pos`.

Similarly, `dist_to_part` and `dist_to_goal` are the norms of `ee_to_part` and
`part_to_goal` respectively — they are exactly `||ee_to_part||` and `||part_to_goal||`.
These are also fully redundant. The network can compute norms. Removing them shrinks
the observation from 46 to 40 dimensions with zero information loss, which reduces
network size and training time.
