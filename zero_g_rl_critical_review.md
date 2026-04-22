Zero-G Assembly — Critical Audit

1. CRITICAL: gamma Is Wrong for the Task Horizon
File: scripts/curriculum_train.py → PPO_CONFIG
python"gamma": 0.995,  # BUG: effective horizon = 1/(1−γ) ≈ 200 steps
Stage 5 has max_steps=1000. At γ=0.995, a reward 1000 steps away is discounted to 0.995^1000 ≈ 0.0067 — effectively invisible to the agent. The release event (the final success signal) is the last thing that happens, meaning PPO never correctly propagates its value backward through the approach→grasp→transport chain.
Fix:
python# Per-stage gamma in STAGE_CONFIGS:
1: {"gamma": 0.99,  ...},
2: {"gamma": 0.99,  ...},
3: {"gamma": 0.995, ...},
4: {"gamma": 0.998, ...},
5: {"gamma": 0.999, ...},

# And pass gamma to PPO.set_env() or re-init per stage:
model = PPO(..., gamma=STAGE_CONFIGS[stage]["gamma"], ...)

2. CRITICAL: No Observation Normalization (Missing VecNormalize)
File: scripts/curriculum_train.py — make_env, run_curriculum
The raw observation vector has wildly heterogeneous scales:

Quaternions: [−1, 1]
Linear/angular velocities: unbounded, easily ±5 or more
Distances: 0 → 3m
H_sys: 0 → ∞ (can spike during collisions)

PPO's neural network receives all of these unnormalized. The value function and policy gradient variance will be dominated by whichever feature has the largest magnitude. This is one of the most common reasons deep RL on continuous control fails silently.
Fix — wrap the training env:
pythonfrom stable_baselines3.common.vec_env import VecNormalize

train_env = SubprocVecEnv([make_env(stage, config['max_steps']) for _ in range(n_envs)])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# When loading a model for a new stage, transfer the normalization stats:
if model is not None:
    old_norm_stats = (train_env.obs_rms, train_env.ret_rms)
    train_env = SubprocVecEnv(...)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    train_env.obs_rms = old_norm_stats[0]  # carry forward running stats
    train_env.ret_rms = old_norm_stats[1]

# Save the normalizer alongside the model:
train_env.save(f"{save_dir}/stage{stage}_vecnorm.pkl")
Also update evaluate_stage and evaluate_policy.py to load and apply the saved normalizer.

3. CRITICAL: Contact Force Observation Is Always Zero
File: envs/truss_assembly_env.py → _get_obs()
python# This block only executes if gripper_closed AND constraint exists
contact_points = p.getContactPoints(self.robot_id, self.part_id,
                                     self.ee_link_idx, -1, ...)
Grasping is done via p.createConstraint(JOINT_FIXED), not via physics-based contact. After the constraint fires, getContactPoints between EE and part returns empty — the constraint bypasses the collision pipeline. So contact_force is [0, 0, 0] for the entire duration the part is held. Three observation dimensions are wasted and the agent has no haptic feedback.
Fix: Replace with constraint reaction forces:
pythonif self.gripper_closed and self.gripper_constraint is not None:
    # getConstraintState returns (appliedForce, appliedTorque) in world frame
    constraint_state = p.getConstraintState(
        self.gripper_constraint, physicsClientId=self.physics_client
    )
    contact_force = list(constraint_state[0])  # [Fx, Fy, Fz]
else:
    contact_force = [0.0, 0.0, 0.0]

4. CRITICAL: Stage 5 Has No Recovery Path After Early Drop
File: envs/truss_assembly_env.py → _compute_reward() Stage 5
pythonwas_grasped_before = self.grasped_part
if self.gripper_closed and not self.grasped_part:
    self.grasped_part = True  # ← set permanently True, never reset
Once grasped_part=True, the agent is routed to Phase 2 (transport) or Phase 3 (release). Phase 3 activates whenever grasped_part and not gripper_closed. If the agent drops the part before reaching the goal — common early in training — it enters Phase 3 which gives -100 every step and there is no way back to Phase 1 to re-grasp. The remaining episode is a punishment loop providing no gradient signal for improvement.
Fix: Track drop vs. intentional release:
python# In reset():
self.grasped_part = False
self.part_dropped_early = False

# In _compute_reward() Phase 3:
elif self.grasped_part and not self.gripper_closed:
    if dist_part_to_goal < self.release_distance:
        reward += release_bonus
        info["success"] = True
    else:
        # Allow re-grasp instead of terminal punishment loop
        reward -= 20.0  # Smaller, recoverable penalty
        self.grasped_part = False   # ← allow re-entry into Phase 1
        self.milestone_first_grasp = False
        info["dropped_early"] = True

5. CRITICAL: Reward Scale Inconsistency Destroys Value Function Transfer
File: envs/truss_assembly_env.py → _compute_reward()
Comparing success bonuses across stages:
StageSuccess BonusMilestone BonusTotal Max1+50—~552+200—~2053+500+200~7004+500+200~7005+500+300~810
When the PPO model transfers from Stage 1 → Stage 2, its value function was calibrated for returns of ~50. Stage 2 immediately introduces returns of ~200, causing large Bellman errors and unstable updates for the first chunk of Stage 2 training. This also partially defeats the purpose of curriculum transfer.
Fix: Normalize all success bonuses to a single scale, e.g., +200 for all stages. The relative difficulty is handled by the threshold and timestep budget, not by inflating bonuses.
pythonSUCCESS_BONUS = 200.0
MILESTONE_BONUS = 50.0
# Replace all stage-specific success/milestone rewards with these constants

6. SIGNIFICANT: gamma and n_epochs / vf_coef Hyperparameter Issues
File: scripts/curriculum_train.py
python"n_epochs": 15,   # Too high — over-fits each rollout batch
"vf_coef": 0.25,  # Half the standard; slow value function learning
With n_envs=8, n_steps=2048, the rollout buffer is 16 384 samples. Reusing them 15 times with clip_range=0.3 means the policy moves far from the collected data's behavior, violating PPO's trust-region assumption. The standard is 10 epochs.
vf_coef=0.25 halves the speed at which the value function converges. For long-horizon tasks where credit assignment is already hard (partially addressed by the gamma fix above), slow value learning makes the advantage estimates noisier for longer.
Fix:
python"n_epochs": 10,
"vf_coef": 0.5,

7. SIGNIFICANT: ent_coef Is Not Staged
File: scripts/curriculum_train.py
python"ent_coef": 0.03,  # Fixed for all stages
Stage 1 (station-keeping) benefits from entropy since the task is spatially broad. Stage 5 (precision insertion) actively suffers from it — high entropy prevents the fine-grained repeated motor patterns needed for release at the goal. A fixed 0.03 is a large penalty for Stage 5 precision.
Fix: Add per-stage entropy override:
pythonSTAGE_CONFIGS = {
    1: {..., "ent_coef": 0.05},
    2: {..., "ent_coef": 0.03},
    3: {..., "ent_coef": 0.02},
    4: {..., "ent_coef": 0.01},
    5: {..., "ent_coef": 0.005},
}
# When advancing stage: model.ent_coef = STAGE_CONFIGS[stage]["ent_coef"]

8. SIGNIFICANT: Task-Space EE Velocity Mixes Units Without Separate Scaling
File: envs/truss_assembly_env.py → step(), task_space branch
pythonarm_vel = action[6:12] * self.max_arm_vel  # max_arm_vel = 1.0 m/s AND rad/s
action[6:9] is EE linear velocity (m/s), action[9:12] is EE angular velocity (rad/s). Both are scaled by 1.0. A wrist rotation of 1.0 rad/s is dimensionally different from a translation of 1.0 m/s. The network has no way to know this, and the policy gradient treats them identically.
Fix:
pythonself.max_ee_linear_vel  = 0.5   # m/s
self.max_ee_angular_vel = 1.0   # rad/s

# In step():
arm_vel = np.concatenate([
    action[6:9]  * self.max_ee_linear_vel,
    action[9:12] * self.max_ee_angular_vel
])

9. SIGNIFICANT: EE Reference Point Is Asymmetric (Finger Left Only)
File: envs/truss_assembly_env.py → _create_robot()
pythonself.ee_link_idx = 7  # finger_left only
The grasp proximity check, Jacobian computation, and observation vector all reference finger_left. The geometric center of the parallel-jaw gripper is midway between the two fingers. Using only finger_left introduces a ~2cm lateral bias in all approach and grasp computations. When the agent learns to position the EE, it will consistently approach from slightly off-center.
Fix: Reference wrist_link_2 (index 6), which is the wrist before the fingers split, or compute the midpoint:
pythonself.ee_link_idx = 6  # wrist_link_2: symmetric gripper base

# OR: for grasping only, compute midpoint dynamically:
left_state  = p.getLinkState(self.robot_id, 7, ...)
right_state = p.getLinkState(self.robot_id, 8, ...)
ee_pos = (np.array(left_state[0]) + np.array(right_state[0])) / 2.0

10. SIGNIFICANT: Movable-Index Jacobian Includes Gripper Joints
File: envs/truss_assembly_env.py → step(), task_space branch
pythonjoint_states = p.getJointStates(self.robot_id, self._movable_indices, ...)
# _movable_indices = [0,1,2,3,4,5,6, 7, 8]  ← includes gripper joints 7 & 8

linear_jacobian, angular_jacobian = p.calculateJacobian(
    self.robot_id, self.ee_link_idx, [0, 0, 0],
    q_movable, ...   # q_movable is length 9
)
linear_jacobian = np.array(linear_jacobian)[:, :len(self.arm_indices)]  # slice to 7
calculateJacobian needs the full q vector for FK (so passing 9 joints is technically correct for computing the EE Jacobian). However, dq_movable also includes gripper velocities for the Coriolis/centrifugal term of the Jacobian formula (dq is used internally), adding a small but spurious velocity contribution from the prismatic gripper joint. Additionally, arm_joint_states is fetched separately for dq_arm:
pythonarm_joint_states = p.getJointStates(self.robot_id, self.arm_indices, ...)
dq_arm = np.array([s[1] for s in arm_joint_states])  # length 7
This means dq_movable[0:7] and dq_arm are redundant fetches. Consolidate:
pythonarm_joint_states = p.getJointStates(self.robot_id, self.arm_indices, ...)
q_arm  = [s[0] for s in arm_joint_states]
dq_arm = np.array([s[1] for s in arm_joint_states])
# Add gripper states for FK completeness:
grip_states = p.getJointStates(self.robot_id, self.gripper_indices, ...)
q_movable  = q_arm + [s[0] for s in grip_states]
dq_movable = list(dq_arm) + [s[1] for s in grip_states]
zero_acc   = [0.0] * len(q_movable)

11. SIGNIFICANT: Observation Slice Definitions Must Be Kept in Sync
File: scripts/evaluation_utils.py
The slice constants are hardcoded:
pythonEE_TO_PART  = slice(24, 27)
GRIPPER_STATE = 30
H_SYS_NORM  = 39
These are derived from the observation layout in TrussAssemblyEnv._get_obs(). If any obs dimension is added/removed from the environment, the evaluation slices silently become wrong — there is no assertion or shared source of truth.
Fix: Define the layout in one place and derive slices programmatically:
python# In truss_assembly_env.py:
OBS_LAYOUT = {
    "robot_orn":    (0,  4),
    "robot_lin_vel":(4,  7),
    "robot_ang_vel":(7,  10),
    "joint_pos":    (10, 17),
    "joint_vel":    (17, 24),
    "ee_to_part":   (24, 27),
    "part_to_goal": (27, 30),
    "gripper_state":(30, 31),
    "contact_force":(31, 34),
    "dist_to_part": (34, 35),
    "dist_to_goal": (35, 36),
    "h_sys":        (36, 39),
    "h_sys_norm":   (39, 40),
}

# In evaluation_utils.py:
from envs.truss_assembly_env import OBS_LAYOUT
H_SYS_NORM  = OBS_LAYOUT["h_sys_norm"][0]
GRIPPER_STATE = OBS_LAYOUT["gripper_state"][0]
DIST_TO_GOAL  = OBS_LAYOUT["dist_to_goal"][0]

12. MINOR: H_SYS_NORM Imported but Unused in evaluate_policy.py
pythonfrom scripts.evaluation_utils import H_SYS_NORM, print_metrics, evaluate_policy_metrics
H_SYS_NORM is never referenced again in that file. Remove the import.

13. MINOR: Stage 5 Part Velocity Initialization vs. Stage 3
File: envs/truss_assembly_env.py → _create_part()
pythonif self.curriculum_stage > 3:
    rand_vel = self.np_random.uniform(-0.1, 0.1, size=3).tolist()
    p.resetBaseVelocity(self.part_id, rand_vel, ...)
Stage 3 is the first time the agent must close the gripper. Adding part motion at Stage 4 is the correct progression. ✓ However, the comment above this block says "Adjusted for new arm workspace" — this comment belongs to the Stage 5 part position, not the velocity initialization. Move the comment.

14. MINOR: Stage-Specific Positions Are Not Randomized
File: envs/truss_assembly_env.py → _create_part(), _create_robot()
For stages 1–5, positions are fixed ([1.0, 0, 0], [0.8, 0, 0], etc.). The agent will overfit to this exact layout and the trained policy will not generalize. Randomization only kicks in at Stage 6 which never exists.
Fix: Add bounded randomization from Stage 3 onward:
pythonelif self.curriculum_stage == 3:
    jitter = self.np_random.uniform(-0.1, 0.1, size=3)
    part_pos = np.array([0.8, 0.0, 0.0]) + jitter
elif self.curriculum_stage == 4:
    jitter = self.np_random.uniform(-0.2, 0.2, size=3)
    part_pos = np.array([0.8, 0.0, 0.0]) + jitter

15. MINOR: vf_coef Scheduling Could Help PPO During Transfer
When the model advances from Stage 1 to Stage 2, the value function is miscalibrated (trained on returns of ~50, now receiving ~200). Temporarily increasing vf_coef at the start of each new stage would speed up recalibration:
python# At start of each new stage:
if stage > start_stage:
    model.vf_coef = 0.75   # Recalibrate value function fast
    # After min_timesteps // 2, restore to 0.5

Summary Table
#SeverityFileIssueFix1🔴 Criticalcurriculum_train.pygamma=0.995 too low for 1000-step horizonPer-stage gamma up to 0.9992🔴 Criticalcurriculum_train.pyNo VecNormalizeWrap train_env and carry stats across stages3🔴 Criticaltruss_assembly_env.pyContact force always zeroUse getConstraintState instead4🔴 Criticaltruss_assembly_env.pyStage 5 no-recovery loop after dropReset grasped_part on non-goal release5🔴 Criticaltruss_assembly_env.pyReward scale 10× jump between stagesNormalize all success bonuses to single constant6🟠 Significantcurriculum_train.pyn_epochs=15, vf_coef=0.25n_epochs=10, vf_coef=0.57🟠 Significantcurriculum_train.pyent_coef fixed at 0.03 for all stagesDecay from 0.05 → 0.005 across stages8🟠 Significanttruss_assembly_env.pyTask-space linear/angular velocity share same scaleSeparate max_ee_linear_vel / max_ee_angular_vel9🟠 Significanttruss_assembly_env.pyEE reference at finger_left (asymmetric)Use wrist_link_2 or dynamic midpoint10🟠 Significanttruss_assembly_env.pyJacobian q-vector includes gripper joints redundantlyConsolidate to arm_indices + gripper supplement11🟠 Significantevaluation_utils.pyObs slice constants duplicated, not derived from envShared OBS_LAYOUT dict in env, imported in eval12🟡 Minorevaluate_policy.pyH_SYS_NORM unused importRemove13🟡 Minortruss_assembly_env.pyMisplaced comment near part velocity initMove comment14🟡 Minortruss_assembly_env.pyFixed positions for stages 1–5 (no generalization)Add bounded jitter from Stage 3 onward15🟡 Minorcurriculum_train.pyvf_coef not boosted temporarily on stage transitionTemporarily increase at stage start
The top two priorities that will cause training to fail silently even if everything else is correct are #1 (gamma) and #2 (VecNormalize). Start there.