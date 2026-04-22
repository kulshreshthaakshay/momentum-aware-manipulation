"""
TrussAssemblyEnv - Environment for truss structure assembly in microgravity.
Extends FreeFlyerEnv with manipulator arm and assembly objectives.
"""

import numpy as np
import pybullet as p
import pybullet_data
import os
import gymnasium as gym
from gymnasium import spaces


# Observation Layout for consistent slicing across scripts
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
    "robot_pos":    (40, 43), # For Stage 1 station keeping
}

# Reward Constants
SUCCESS_BONUS = 200.0
MILESTONE_BONUS = 50.0


class TrussAssemblyEnv(gym.Env):
    """
    A free-flying robot with a manipulator arm assembling truss structures.
    Task: Grasp floating parts and insert them into target locations.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None, max_steps=2000, curriculum_stage=1, control_mode="joint"):
        super().__init__()
        
        self.render_mode = render_mode
        self.control_mode = control_mode # "joint" or "task_space"
        self.max_steps = max_steps
        self.step_count = 0
        self.curriculum_stage = curriculum_stage
        
        # Physics parameters
        self.dt = 1.0 / 240.0
        self.sim_substeps = 5
        
        # Zero-G Servicer Config (60kg Class)
        self.max_thrust = 80.0  # Scaled for 60kg (was 20N for 15kg) -> ~1.33 m/s^2 accel
        self.max_torque = 8.0   # Scaled for larger inertia (was 2N)
        self.max_arm_vel = 1.0
        
        # Significant-8: Separate scaling for Task-Space velocities
        self.max_ee_linear_vel = 0.5   # m/s
        self.max_ee_angular_vel = 1.0  # rad/s
        
        
        # Fix 2: Gripper threshold (use 0.3 instead of 0 for clearer signal)
        self.gripper_threshold = 0.3
        
        # Fix 3: Increased grasp distance for easier discovery
        self.grasp_distance = 0.25  # Was 0.15
        self.stage4_success_distance = 0.30
        self.release_distance = 0.40
        self.at_goal_distance = 0.40
        
        # Track previous distance for shaping
        self.prev_dist_to_part = None

        
        # Observation space (relative vectors for translation-invariant transfer)
        # Robot (10): orn(4), lin_vel(3), ang_vel(3)
        # Arm (14): joint_pos(7), joint_vel(7)
        # Relative (6): ee_to_part(3), part_to_goal(3)
        # Gripper (1): gripper_state(1)
        # Contact (3): contact_force(3)
        # Distances (2): dist_part(1), dist_goal(1)
        # Momentum (4): H_sys(3), ||H_sys||(1)
        # Absolute (3): robot_pos(3) - Stage 1 only
        # Total = 10 + 14 + 6 + 1 + 3 + 2 + 4 + 3 = 43
        
        self.num_arm_joints = 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(43,), dtype=np.float32
        )
        
        # Action space
        if self.control_mode == "task_space":
             # Thrust(3) + Torque(3) + EE_Lin_Vel(3) + EE_Ang_Vel(3) + Gripper(1) = 13
             self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(13,), dtype=np.float32
            )
        else:
            # Thrust(3) + Torque(3) + Arm_vel(7) + Gripper(1) = 14
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(14,), dtype=np.float32
            )
        
        self.physics_client = None
        self.robot_id = None
        self.part_id = None
        self.goal_id = None
        self.gripper_constraint = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.physics_client is not None:
            p.disconnect(physicsClientId=self.physics_client)
        
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0,
                                       physicsClientId=self.physics_client)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Fix 1.3: Assert connection succeeded before proceeding
        assert self.physics_client >= 0, "PyBullet connection failed"
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.physics_client)
        p.setGravity(0, 0, 0, physicsClientId=self.physics_client)  # MICROGRAVITY!
        p.setTimeStep(self.dt, physicsClientId=self.physics_client)
        
        # Create robot from URDF
        self._create_robot()
        
        # Create floating part to assemble
        self._create_part()
        
        # Create goal location
        self._create_goal()
        
        self.step_count = 0
        self.gripper_constraint = None
        self.gripper_closed = False
        self.prev_dist_to_part = None  # Will be set on first step
        self.prev_dist_to_goal = None  # For Stage 4 transport tracking
        self.reached_goal = False  # For Stage 5 release tracking
        self.grasped_part = False  # Track if we ever grasped the part
        self.steps_at_goal_holding = 0  # Track time at goal while holding (Stage 5)
        
        # Fix 4: Milestone tracking for one-time bonuses
        self.milestone_first_grasp = False
        self.milestone_reached_goal_area = False
        
        # Significant-14: Persistent flags to prevent reward hacking via re-grasp loops (Stage 5)
        self.milestone_first_grasp_ever = False
        self.milestone_reached_goal_ever = False
        
        # Fix 2.1: Grasp hold counter for Stage 3
        self.grasp_hold_steps = 0
        
        # Fix 2.4: Station keeping sustained success counter
        self.station_keeping_steps = 0
        self._prev_gripper_closed = False  # Track for transition logic
        self.part_dropped_early = False    # Track for Stage 5 recovery path
        
        
        # Initial cached momentum for observation
        self._cached_H_sys = self._compute_system_angular_momentum()
        self._cached_H_sys_norm = np.linalg.norm(self._cached_H_sys)
        
        obs = self._get_obs()
        info = {"stage": self.curriculum_stage}
        
        return obs, info
    
    def _create_robot(self):
        """Create the free-flyer robot by loading the 7-DOF URDF."""
        # Random initial position based on curriculum
        if self.curriculum_stage <= 4:
            init_pos = [0.0, 0.0, 0.0]
        elif self.curriculum_stage == 5:
            init_pos = [1.0, 0.0, 0.0]
        else:
            init_pos = self.np_random.uniform(-0.3, 0.3, size=3).tolist()
            
        # Minor-14: Randomize initial robot position for generalization
        if self.curriculum_stage >= 3:
            jitter = self.np_random.uniform(-0.2, 0.2, size=3).tolist()
            init_pos = [init_pos[i] + jitter[i] for i in range(3)]

        # Load URDF
        urdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "zero_g_servicer.urdf")
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=init_pos,
            baseOrientation=[0, 0, 0, 1],
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.physics_client
        )
        
        # Identify Joint Indices
        # 0-6: Arm Joints. 7: Gripper left. 8: Gripper right.
        self.arm_indices = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_indices = [7, 8]
        
        # Reset Joint States (Neutral pose)
        # S-Y, S-P, S-R, E-P, W-Y, W-P, W-R
        neutral_pose = [0, 0, 0, 1.57, 0, -1.57, 0] # "Ready" pose
        for i, idx in enumerate(self.arm_indices):
            p.resetJointState(self.robot_id, idx, neutral_pose[i], physicsClientId=self.physics_client)
        
        # Enable Velocity Control Mode (disable default position control)
        p.setJointMotorControlArray(
            self.robot_id, self.arm_indices,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0]*7, # Allow free movement if not controlled
            physicsClientId=self.physics_client
        )
        
        # CRITICAL: Disable Damping for Base AND Joints
        p.changeDynamics(self.robot_id, -1, linearDamping=0.0, angularDamping=0.0, physicsClientId=self.physics_client)
        
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)):
            p.changeDynamics(self.robot_id, i, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0, physicsClientId=self.physics_client)
        
        # Significant-9: Reference wrist_link_2 (idx 6) for symmetric gripper base
        # This eliminates the 2cm lateral bias from finger_left (idx 7).
        self.ee_link_idx = 6
        
        # Cache joint limits from URDF for proximity penalty
        self.joint_limits = []
        for idx in self.arm_indices:
            joint_info = p.getJointInfo(self.robot_id, idx, physicsClientId=self.physics_client)
            lower = joint_info[8]  # lowerLimit
            upper = joint_info[9]  # upperLimit
            self.joint_limits.append((lower, upper))
            
        # Cache movable joint indices (excludes FIXED joints)
        self._movable_indices = []
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)):
            if p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)[2] != p.JOINT_FIXED:
                self._movable_indices.append(i)
    
    def _create_part(self):
        """Create the floating truss part to be assembled."""
        # Peg/rod shape
        part_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.04, height=0.3, physicsClientId=self.physics_client)
        part_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.04, length=0.3,
                                        rgbaColor=[0.9, 0.7, 0.2, 1.0], physicsClientId=self.physics_client)
        
        # Part position based on curriculum
        if self.curriculum_stage == 1:
            part_pos = [1.0, 0.0, 0.0]
        elif self.curriculum_stage <= 4:
            part_pos = [0.8, 0.0, 0.0]
        elif self.curriculum_stage == 5:
            # Minor-13 Fix: Corrected workspace target for full assembly
            part_pos = [1.6, 0.0, 0.0] 
        else:
            part_pos = self.np_random.uniform(1.0, 1.5, size=3)
            part_pos[1:] = self.np_random.uniform(-0.3, 0.3, size=2)
        
        # Minor-14: Bounded randomization for part position generalization
        if self.curriculum_stage >= 3:
            jitter = self.np_random.uniform(-0.1, 0.1, size=3)
            part_pos = np.array(part_pos) + jitter

        self.part_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=part_col,
            baseVisualShapeIndex=part_vis,
            basePosition=part_pos.tolist() if isinstance(part_pos, np.ndarray) else part_pos,
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.physics_client
        )
        
        p.changeDynamics(self.part_id, -1, linearDamping=0.0, angularDamping=0.0, physicsClientId=self.physics_client)
        
        # Minor-13: Move part velocity initialization to correct progression (Stage 4+)
        if self.curriculum_stage >= 4:
            rand_vel = self.np_random.uniform(-0.1, 0.1, size=3).tolist()
            p.resetBaseVelocity(self.part_id, rand_vel, [0, 0, 0], physicsClientId=self.physics_client)
    
    def _create_goal(self):
        """Create the goal location for assembly."""
        goal_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.06, length=0.1,
                                        rgbaColor=[0.2, 0.9, 0.2, 0.5], physicsClientId=self.physics_client)
        
        if self.curriculum_stage == 5:
            # Keep the full-assembly target far enough that release requires transport.
            goal_pos = [2.4, 0.0, 0.0]
        elif self.curriculum_stage < 5:
            goal_pos = [2.0, 0.0, 0.0]
        else:
            goal_pos = self.np_random.uniform(2.0, 3.0, size=3)
        
        self.goal_pos = np.array(goal_pos)
        self.goal_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=goal_vis,
            basePosition=goal_pos.tolist() if isinstance(goal_pos, np.ndarray) else goal_pos,
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.physics_client
        )
    
    def step(self, action):
        self.step_count += 1
        
        if self.control_mode == "task_space":
            # 13 dims: 6 base + 6 EE + 1 Gripper
            thrust = action[:3] * self.max_thrust
            torque = action[3:6] * self.max_torque
            
            # Significant-8: Separate scaling for Task-Space velocities (m/s vs rad/s)
            arm_vel = np.concatenate([
                action[6:9]  * self.max_ee_linear_vel,
                action[9:12] * self.max_ee_angular_vel
            ])
            gripper_action = action[12]
        else:
            # 14 dims: 6 base + 7 joint + 1 Gripper
            thrust = action[:3] * self.max_thrust
            torque = action[3:6] * self.max_torque
            arm_vel = action[6:13] * self.max_arm_vel # 7 joints
            gripper_action = action[13] # -1 to 1
            
            # Apply Arm Controls Directly (Joint Space)
            p.setJointMotorControlArray(
                self.robot_id, self.arm_indices,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=arm_vel,
                forces=[100]*7, # Max force
                physicsClientId=self.physics_client
            )
            
        if self.control_mode == "task_space":
            # --- TASK SPACE CONTROL (NULL SPACE PROJECTION) ---
            # Significant-10: Consolidate joint state fetching and FK completeness
            arm_joint_states = p.getJointStates(self.robot_id, self.arm_indices, physicsClientId=self.physics_client)
            q_arm  = [s[0] for s in arm_joint_states]
            dq_arm = np.array([s[1] for s in arm_joint_states])
            
            grip_states = p.getJointStates(self.robot_id, self.gripper_indices, physicsClientId=self.physics_client)
            q_movable  = q_arm + [s[0] for s in grip_states]
            dq_movable = list(dq_arm) + [s[1] for s in grip_states]
            zero_acc   = [0.0] * len(q_movable)
            
            linear_jacobian, angular_jacobian = p.calculateJacobian(
                self.robot_id, self.ee_link_idx, [0, 0, 0],
                q_movable, dq_movable, zero_acc,
                physicsClientId=self.physics_client
            )
            
            # Slice to only the arm joints (redundant joints removed from calculation)
            linear_jacobian = np.array(linear_jacobian)[:, :len(self.arm_indices)]
            angular_jacobian = np.array(angular_jacobian)[:, :len(self.arm_indices)]
            J = np.vstack([linear_jacobian, angular_jacobian])
            
            lambda_val = 0.05
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_val**2 * np.eye(6))
            
            arm_joint_states = p.getJointStates(self.robot_id, self.arm_indices,
                                                physicsClientId=self.physics_client)
            dq_arm = np.array([s[1] for s in arm_joint_states])
            
            null_projector = np.eye(7) - J_pinv @ J
            target_joint_vels = J_pinv @ arm_vel + null_projector @ (-0.5 * dq_arm)
            
            p.setJointMotorControlArray(
                self.robot_id, self.arm_indices,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=target_joint_vels,
                forces=[100]*7,
                physicsClientId=self.physics_client
            )
            
        # Action Repetition Loop
        for _ in range(self.sim_substeps):
            # Apply Base Thrust/Torque in BODY frame (thrusters are body-fixed)
            p.applyExternalForce(self.robot_id, -1, thrust.tolist(), [0, 0, 0], p.LINK_FRAME, physicsClientId=self.physics_client)
            p.applyExternalTorque(self.robot_id, -1, torque.tolist(), p.LINK_FRAME, physicsClientId=self.physics_client)
            
            # Handle Gripper (Prismatic/Constraint) inside the substep loop to avoid physics snap
            self._handle_gripper(gripper_action)
            
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # Compute momentum ONCE and share between obs and reward
        self._cached_H_sys = self._compute_system_angular_momentum()
        self._cached_H_sys_norm = np.linalg.norm(self._cached_H_sys)
        
        # Get observation and compute reward
        obs = self._get_obs()
        reward, info = self._compute_reward(action)
        
        terminated = bool(info.get("success", False))
        truncated = bool(self.step_count >= self.max_steps)
        
        # Out of bounds check
        robot_pos = np.array(p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)[0])
        if np.linalg.norm(robot_pos) > 15.0:
            truncated = True
            reward -= 50.0
        
        return obs, reward, terminated, truncated, info
    
    def _handle_gripper(self, gripper_action):
        """Handle gripper open/close via prismatic joint and constraint."""
        should_close = gripper_action > self.gripper_threshold
        
        # 1. Visual/Physics Gripper Movement (Prismatic Joint)
        target_pos = 0.0 if should_close else 0.03 # Close=0, Open=3cm
        p.setJointMotorControl2(
            self.robot_id, self.gripper_indices[0], # finger_left
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_pos,
            force=10,
            physicsClientId=self.physics_client
        )
        # Mirror for right finger (assuming symmetric gripper)
        p.setJointMotorControl2(
            self.robot_id, self.gripper_indices[1], # finger_right
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_pos, # Axis handles the direction
            force=10,
            physicsClientId=self.physics_client
        )
        
        # 2. Logical Grasping (Constraint Injection)
        # Get EE State
        ee_state = p.getLinkState(self.robot_id, self.ee_link_idx, physicsClientId=self.physics_client)
        ee_pos = np.array(ee_state[0])
        ee_orn = ee_state[1]
        
        part_pos, part_orn = p.getBasePositionAndOrientation(self.part_id, physicsClientId=self.physics_client)
        dist_to_part = np.linalg.norm(ee_pos - np.array(part_pos))
        
        if should_close and not self.gripper_closed and dist_to_part < self.grasp_distance:
            # Compute the actual relative transform between EE and part at grasp time.
            # This avoids the spring oscillation caused by a hardcoded offset mismatch.
            ee_inv_pos, ee_inv_orn = p.invertTransform(ee_pos.tolist(), ee_orn)
            local_pos, local_orn = p.multiplyTransforms(
                ee_inv_pos, ee_inv_orn,
                list(part_pos), list(part_orn)
            )
            
            self.gripper_constraint = p.createConstraint(
                self.robot_id, self.ee_link_idx, self.part_id, -1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=list(local_pos),
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=list(local_orn),
                physicsClientId=self.physics_client
            )
            self.gripper_closed = True
            
        elif not should_close and self.gripper_closed:
            # Remove constraint
            if self.gripper_constraint is not None:
                p.removeConstraint(self.gripper_constraint, physicsClientId=self.physics_client)
                self.gripper_constraint = None
            self.gripper_closed = False

    def _compute_system_angular_momentum(self):
        """
        Compute the total angular momentum of the multi-body system about the
        system center of mass. This is the conserved quantity in microgravity.
        
        H_sys = sum_i [ I_i^world * omega_i + m_i * (r_i x v_i) ]
        
        where r_i is the vector from system COM to body i's COM, and v_i is
        body i's COM velocity in world frame.
        """
        # --- Pass 1: Compute system COM ---
        total_mass = 0.0
        com_numerator = np.zeros(3)
        body_data = []  # Store (mass, com_world, vel_world, omega_world, I_world_com)

        def add_body_link(body_id, link_idx):
            nonlocal total_mass, com_numerator

            dyn_info = p.getDynamicsInfo(body_id, link_idx, physicsClientId=self.physics_client)
            mass = dyn_info[0]
            if mass == 0.0:
                return
            
            local_inertia_diag = np.array(dyn_info[2])
            local_inertial_pos = np.array(dyn_info[3])
            local_inertial_orn = dyn_info[4]
            
            if link_idx == -1:
                pos, orn = p.getBasePositionAndOrientation(body_id,
                                                           physicsClientId=self.physics_client)
                vel, omega = p.getBaseVelocity(body_id,
                                               physicsClientId=self.physics_client)
            else:
                # Fix 1.2: Use computeForwardKinematics=1 and correct COM velocity
                link_state = p.getLinkState(body_id, link_idx,
                                            computeLinkVelocity=1,
                                            computeForwardKinematics=1,
                                            physicsClientId=self.physics_client)
                pos = link_state[4]     # World position of link frame
                orn = link_state[5]     # World orientation of link frame
                frame_vel = np.array(link_state[6])
                omega = link_state[7]

            # Common rotation matrix
            R_link = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

            if link_idx != -1:
                # Fix 1.2: Correct via rigid-body kinematics: v_com = v_frame + omega × r_offset
                r_offset = R_link @ local_inertial_pos
                vel = frame_vel + np.cross(np.array(omega), r_offset)
            else:
                # For base, inertial offset is handled by getBasePositionAndOrientation if URDF is centered?
                # Actually, PyBullet base position IS the frame origin.
                # If we want COM velocity of base: v_com_base = v_base + omega_base x r_offset_base
                r_offset = R_link @ local_inertial_pos
                vel = np.array(vel) + np.cross(np.array(omega), r_offset)
            
            # Rotation matrices
            R_inertial = np.array(p.getMatrixFromQuaternion(local_inertial_orn)).reshape(3, 3)
            R_world = R_link @ R_inertial
            
            # COM position in world frame
            com_world = np.array(pos) + R_link @ local_inertial_pos
            
            # Inertia tensor in world frame (about body COM)
            I_diag = np.diag(local_inertia_diag)
            I_world = R_world @ I_diag @ R_world.T
            
            vel_world = np.array(vel)
            omega_world = np.array(omega)
            
            total_mass += mass
            com_numerator += mass * com_world
            body_data.append((mass, com_world, vel_world, omega_world, I_world))

        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        for idx in range(-1, num_joints):
            add_body_link(self.robot_id, idx)

        if self.part_id is not None:
            add_body_link(self.part_id, -1)
        
        if total_mass == 0.0:
            return np.zeros(3)
        
        system_com = com_numerator / total_mass
        
        # --- Pass 2: Compute total angular momentum about system COM ---
        H_sys = np.zeros(3)
        for mass, com_world, vel_world, omega_world, I_world in body_data:
            r = com_world - system_com
            # Spin angular momentum: I * omega
            H_spin = I_world @ omega_world
            # Orbital angular momentum: m * (r x v)
            H_orbital = mass * np.cross(r, vel_world)
            H_sys += H_spin + H_orbital
        
        return H_sys

    def _get_obs(self):
        """
        Constructs the observation space using RELATIVE vectors for
        translation-invariant policy transfer between curriculum stages.

        Robot (10): orn(4), lin_vel(3), ang_vel(3)
        Arm   (14): joint_pos(7), joint_vel(7)
        Relative(6): ee_to_part(3), part_to_goal(3)
        Gripper (1): gripper_state(1)
        Contact (3): contact_force(3)
        Dist    (2): dist_to_part(1), dist_to_goal(1)
        Momentum(4): H_sys(3), ||H_sys||(1)
        Total = 10 + 14 + 6 + 1 + 3 + 2 + 4 = 40
        """
        _, robot_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        robot_lin_vel, robot_ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)

        # Arm joint states
        joint_states = p.getJointStates(self.robot_id, self.arm_indices, physicsClientId=self.physics_client)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]

        # End effector position (using link state)
        ee_state = p.getLinkState(self.robot_id, self.ee_link_idx, physicsClientId=self.physics_client)
        ee_pos = np.array(ee_state[0])

        # Part state
        part_pos, _ = p.getBasePositionAndOrientation(self.part_id, physicsClientId=self.physics_client)
        part_pos = np.array(part_pos)

        # Goal position (static)
        goal_pos = self.goal_pos

        # Relative vectors (translation-invariant: same regardless of world position)
        ee_to_part = part_pos - ee_pos          # Direction from EE to part
        part_to_goal = goal_pos - part_pos       # Direction from part to goal

        # Gripper state (0 for open, 1 for closed)
        gripper_state = float(self.gripper_closed)

        # Contact force (3-axis measurement)
        # Critical-3: Constraint reaction forces provide haptic feedback when held
        contact_force = [0.0, 0.0, 0.0]
        if self.gripper_closed and self.gripper_constraint is not None:
            # getConstraintState returns [Fx, Fy, Fz, Tx, Ty, Tz] in world frame
            constraint_state = p.getConstraintState(self.gripper_constraint, physicsClientId=self.physics_client)
            contact_force = list(constraint_state[:3])  # [Fx, Fy, Fz]
        elif self.gripper_closed:
            # Fallback if constraint not yet formed but gripper is closing
            contact_points = p.getContactPoints(self.robot_id, self.part_id, self.ee_link_idx, -1, physicsClientId=self.physics_client)
            if contact_points:
                total_force = np.zeros(3)
                for cp in contact_points:
                    normal = np.array(cp[7])
                    force_magnitude = cp[9]
                    total_force += force_magnitude * normal
                contact_force = total_force.tolist()

        # Distances
        dist_to_part = np.linalg.norm(ee_to_part)
        dist_part_to_goal = np.linalg.norm(part_to_goal)

        # Stage 1 visibility: Include absolute position for station keeping
        # BUG-6 Fix: Gate absolute pos by stage (normed by ~workspace radius)
        if self.curriculum_stage == 1:
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
            robot_pos_obs = np.clip(np.array(robot_pos) / 5.0, -1.0, 1.0)
        else:
            robot_pos_obs = np.zeros(3)

        obs = np.concatenate([
            robot_orn,
            robot_lin_vel,
            robot_ang_vel,
            joint_pos,
            joint_vel,
            ee_to_part,
            part_to_goal,
            [gripper_state],
            contact_force,
            [dist_to_part, dist_part_to_goal],
            self._cached_H_sys,
            [self._cached_H_sys_norm],
            robot_pos_obs
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, action):
        """Compute reward based on curriculum stage with potential-based shaping."""
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        robot_pos = np.array(robot_pos)
        robot_vel, robot_ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)
        
        part_pos, _ = p.getBasePositionAndOrientation(self.part_id, physicsClientId=self.physics_client)
        part_pos = np.array(part_pos)
        
        # Calculate EE position (Dynamic from 7-DOF arm)
        ee_state = p.getLinkState(self.robot_id, self.ee_link_idx, computeLinkVelocity=1, physicsClientId=self.physics_client)
        ee_pos = np.array(ee_state[0])
        ee_vel = np.array(ee_state[6])
        part_vel, _ = p.getBaseVelocity(self.part_id, physicsClientId=self.physics_client)
        part_vel = np.array(part_vel)
        
        # Distances
        dist_to_part = np.linalg.norm(ee_pos - part_pos)
        dist_part_to_goal = np.linalg.norm(part_pos - self.goal_pos)
        
        info = {
            "dist_to_part": dist_to_part,
            "dist_to_goal": dist_part_to_goal,
            "gripper_closed": self.gripper_closed,
            "stage": self.curriculum_stage
        }
        gripper_cmd = float(action[-1])
        
        # Differentiated fuel penalties: RCS thrusters (expendable) vs electric joints (renewable)
        rcs_penalty = 0.05 * np.sum(np.abs(action[:6]))     # Base thrust + torque (expendable fuel)
        joint_penalty = 0.005 * np.sum(np.abs(action[6:]))  # Arm joints + gripper (electric)
        fuel_penalty = rcs_penalty + joint_penalty
        
        # Fix 2.3: Compute H_sys ONCE and cache — avoid O(N²) duplicate calls
        H_sys = self._cached_H_sys
        H_sys_norm = self._cached_H_sys_norm
        info["H_sys_norm"] = H_sys_norm
        # Scale coherence fix: Boost momentum penalty to 0.1
        momentum_penalty = 0.1 * H_sys_norm
        # H_sys_norm is now available for all stage blocks below (no re-computation)
        
        # Joint limit proximity penalty (smooth activation near URDF limits)
        joint_states_for_limits = p.getJointStates(self.robot_id, self.arm_indices, physicsClientId=self.physics_client)
        joint_limit_penalty = 0.0
        for i, idx in enumerate(self.arm_indices):
            pos = joint_states_for_limits[i][0]
            lower, upper = self.joint_limits[i]
            range_size = upper - lower
            if range_size > 0:
                normalized = 2.0 * abs(pos - (lower + upper) / 2.0) / range_size
                if normalized > 0.85:
                    joint_limit_penalty += (normalized - 0.85) ** 2
        
        reward = -fuel_penalty - momentum_penalty - 0.5 * joint_limit_penalty
        
        # Stage-specific rewards with potential-based shaping
        if self.curriculum_stage == 1:
            # Station keeping - stay near origin
            dist_from_origin = np.linalg.norm(robot_pos)
            vel_magnitude = np.linalg.norm(robot_vel)
            
            # Dense reward: closer is better
            reward += 1.0 - dist_from_origin  # Max +1 at origin
            reward -= 0.1 * vel_magnitude
            
            # Fix 2.4: Require sustained station keeping for success
            if dist_from_origin < 0.1 and vel_magnitude < 0.1:
                self.station_keeping_steps += 1
                # Bug 7: Cap escalating reward to prevent PPO value spike
                reward += min(1.0 * self.station_keeping_steps, 5.0)
                if self.station_keeping_steps >= 20:  # ~0.4s of success
                    reward += SUCCESS_BONUS / 4.0  # Normalized bonus
                    info["success"] = True
            else:
                self.station_keeping_steps = 0
        
        elif self.curriculum_stage == 2:
            # Approach part - PROGRESS-BASED shaping
            
            # Initialize prev distance on first step
            if self.prev_dist_to_part is None:
                self.prev_dist_to_part = dist_to_part
            
            # Progress reward: reward for getting CLOSER (delta-based)
            progress = self.prev_dist_to_part - dist_to_part
            reward += 100.0 * progress  # Boosted progress reward
            
            # Velocity toward target bonus
            direction_to_part = (part_pos - ee_pos) / (dist_to_part + 1e-6)
            velocity_toward = np.dot(ee_vel, direction_to_part)
            reward += 3.0 * max(0, velocity_toward)
            
            # Penalty for stalling when close
            if dist_to_part < self.grasp_distance and progress < 0.001:
                reward -= 1.0  # Get there faster!
            
            # Update for next step
            self.prev_dist_to_part = dist_to_part
            
            # Success condition
            if dist_to_part < self.grasp_distance:
                reward += SUCCESS_BONUS
                info["success"] = True
        
        elif self.curriculum_stage == 3:
            # Grasp part - approach + grasp
            # Key insight: Penalize hovering, reward ONLY progress and grasping
            
            # Initialize prev distance on first step
            if self.prev_dist_to_part is None:
                self.prev_dist_to_part = dist_to_part
            
            if not self.gripper_closed:
                # Progress reward for approaching
                progress = self.prev_dist_to_part - dist_to_part
                reward += 50.0 * progress  # Strong progress incentive
                
                # Velocity toward target (only when far)
                if dist_to_part > self.grasp_distance:
                    direction_to_part = (part_pos - ee_pos) / (dist_to_part + 1e-6)
                    velocity_toward = np.dot(ee_vel, direction_to_part)
                    reward += 2.0 * max(0, velocity_toward)
                
                # Encourage grasping when close
                if dist_to_part < self.grasp_distance:
                    # Proximity bonus: reward getting deeper into the grasp zone
                    proximity_bonus = 5.0 * (1.0 - dist_to_part / self.grasp_distance)
                    reward += proximity_bonus
                    # Explicitly shape gripper command near grasp zone.
                    reward += 2.0 * max(0.0, gripper_cmd)
                    reward -= 1.0 * max(0.0, -gripper_cmd)
                else:
                    # Discourage premature gripper closure far from the part.
                    reward -= 0.2 * max(0.0, gripper_cmd)
            
            # Update for next step
            self.prev_dist_to_part = dist_to_part
            
            # Success: Grasped! MUCH bigger reward
            # Success: Grasped! MUCH bigger reward
            # Fix 2.1: Require N consecutive steps of closed gripper for Stage 3 success
            if self.gripper_closed:
                self.grasp_hold_steps += 1
                # Fix: Milestone bonus for first grasp (crucial for learning)
                if not self.milestone_first_grasp:
                    reward += MILESTONE_BONUS
                    self.milestone_first_grasp = True
                
                if self.grasp_hold_steps >= 5:  # Hold for 5 consecutive steps
                    reward += SUCCESS_BONUS
                    info["success"] = True
            else:
                self.grasp_hold_steps = 0
        
        elif self.curriculum_stage == 4:
            # Transport part to goal - two phases: grasp then transport
            
            # Initialize tracking
            if self.prev_dist_to_part is None:
                self.prev_dist_to_part = dist_to_part
            if self.prev_dist_to_goal is None:
                self.prev_dist_to_goal = dist_part_to_goal
            
            if not self.gripper_closed:
                # PHASE 1: Approach and grasp (reuse Stage 3 logic)
                self.grasp_hold_steps = 0  # Bug 4: Reset hold counter
                progress = self.prev_dist_to_part - dist_to_part
                reward += 50.0 * progress
                
                # Velocity toward part
                if dist_to_part > self.grasp_distance:
                    direction_to_part = (part_pos - ee_pos) / (dist_to_part + 1e-6)
                    velocity_toward = np.dot(ee_vel, direction_to_part)
                    reward += 2.0 * max(0, velocity_toward)
                
                # Encourage grasping when close
                if dist_to_part < self.grasp_distance:
                    reward += 2.0 * max(0.0, gripper_cmd)
                    reward -= 1.0 * max(0.0, -gripper_cmd)
                
                # Grasp bonus
                self.prev_dist_to_part = dist_to_part
            else:
                # PHASE 2: Transport to goal
                
                self.grasp_hold_steps += 1
                
                # Fix: Milestone bonus for first grasp (require 3 holds)
                if not self.milestone_first_grasp and self.grasp_hold_steps >= 3:
                    reward += MILESTONE_BONUS
                    self.milestone_first_grasp = True
                    # Fix 6.2: Reset prev_dist_to_goal at grasp transition
                    self.prev_dist_to_goal = dist_part_to_goal
                
                if not self.milestone_first_grasp:
                    pass # Waiting to stabilize grasp
                else:
                    # Fix 2.2: Unified progress multiplier, clipped non-negative
                    goal_progress = self.prev_dist_to_goal - dist_part_to_goal
                    reward += 100.0 * max(0.0, goal_progress)  # Same scale as Stage 5
                    
                    # Velocity toward goal
                    direction_to_goal = (self.goal_pos - part_pos) / (dist_part_to_goal + 1e-6)
                    velocity_toward_goal = np.dot(part_vel, direction_to_goal)
                    reward += 3.0 * max(0, velocity_toward_goal)
                    
                    # Velocity damping near goal: penalize excess speed to prevent overshoot
                    if dist_part_to_goal < 1.0:
                        approach_speed = np.linalg.norm(part_vel)
                        desired_max_speed = dist_part_to_goal * 0.5
                        excess_speed = max(0, approach_speed - desired_max_speed)
                        reward -= 5.0 * excess_speed
                    
                    # Small holding bonus (but not too big to encourage hovering)
                    reward += 0.1
                    
                    self.prev_dist_to_goal = dist_part_to_goal
                    
                    # Success: reached goal with part
                    if dist_part_to_goal < self.stage4_success_distance:
                        reward += SUCCESS_BONUS
                        info["success"] = True
        
        else:  # Stage 5: Full assembly sequence (approach → grasp → transport → release)
            # Initialize tracking
            if self.prev_dist_to_part is None:
                self.prev_dist_to_part = dist_to_part
            if self.prev_dist_to_goal is None:
                self.prev_dist_to_goal = dist_part_to_goal
            
            # Track time spent at goal while holding (for escalating penalty)
            if not hasattr(self, 'steps_at_goal_holding'):
                self.steps_at_goal_holding = 0
            
            was_grasped_before = self.grasped_part
            if self.gripper_closed and not self.grasped_part:
                self.grasped_part = True
            
            # Bug 5: Reset prev_dist_to_goal on re-grasp transition
            gripper_just_closed = self.gripper_closed and not getattr(self, "_prev_gripper_closed", False)
            if gripper_just_closed and self.grasped_part:
                self.prev_dist_to_goal = dist_part_to_goal

            if not self.gripper_closed:
                self.grasp_hold_steps = 0
            
            # === PHASE 1: APPROACH AND GRASP ===
            if not was_grasped_before:
                # Progress toward part
                progress = self.prev_dist_to_part - dist_to_part
                reward += 50.0 * progress
                
                # Velocity toward part bonus
                direction_to_part = (part_pos - ee_pos) / (dist_to_part + 1e-6)
                velocity_toward = np.dot(ee_vel, direction_to_part)
                reward += 2.0 * max(0, velocity_toward)
                
                if dist_to_part < self.grasp_distance:
                    if self.gripper_closed:
                        self.grasp_hold_steps += 1
                        # Fix 4: First grasp milestone bonus (require 3 holds)
                        if not self.milestone_first_grasp_ever and self.grasp_hold_steps >= 3:
                            reward += MILESTONE_BONUS
                            self.milestone_first_grasp_ever = True
                            self.milestone_first_grasp = True
                            self.prev_dist_to_goal = dist_part_to_goal
                        elif self.milestone_first_grasp:
                            reward += 20.0  # Regular grasp bonus
                    else:
                        reward -= 2.0  # Penalty for being close but not grasping
                        reward += 2.0 * max(0.0, gripper_cmd)
                else:
                    reward -= 0.2 * max(0.0, gripper_cmd)
                
                self.prev_dist_to_part = dist_to_part
            
            # === PHASE 2: TRANSPORT TO GOAL ===
            elif self.gripper_closed:
                # Progress toward goal
                # Fix 2.2: Unified progress multiplier, clipped non-negative
                goal_progress = self.prev_dist_to_goal - dist_part_to_goal
                reward += 100.0 * max(0.0, goal_progress)
                
                self.prev_dist_to_goal = dist_part_to_goal
                direction_to_goal = (self.goal_pos - part_pos) / (dist_part_to_goal + 1e-6)
                reward += 4.0 * max(0.0, np.dot(part_vel, direction_to_goal))
                
                # Velocity damping near goal: penalize excess speed to prevent overshoot
                if dist_part_to_goal < 1.0:
                    approach_speed = np.linalg.norm(part_vel)
                    desired_max_speed = dist_part_to_goal * 0.5
                    excess_speed = max(0, approach_speed - desired_max_speed)
                    reward -= 5.0 * excess_speed
                
                # === AT GOAL: MUST RELEASE ===
                at_release_zone = (
                    dist_part_to_goal < self.at_goal_distance
                    or (self.milestone_reached_goal_area and dist_part_to_goal < 0.50)
                )
                if at_release_zone:
                    # Fix 4: First time reaching goal area milestone
                    if not self.milestone_reached_goal_ever:
                        reward += 100.0  # ONE-TIME bonus for reaching goal!
                        self.milestone_reached_goal_ever = True
                        self.milestone_reached_goal_area = True
                    
                    # Significant-15: Fixed penalty (Markovian) instead of escalating
                    # This ensures identical states have identical rewards
                    reward -= 2.0
                    
                    # Explicit release incentive: signal that opening is good here
                    # gripper_action is index 12 in task_space, 13 in joint mode (both are last)
                    if action[-1] < -0.3: # Threshold slightly below gripper_threshold
                        reward += 5.0
                    
                    info["at_goal_holding"] = True
                else:
                    # Clear flag if drifted out
                    self.milestone_reached_goal_area = False
            
            # === PHASE 3: RELEASE AT GOAL ===
            elif self.grasped_part and not self.gripper_closed:
                # Part has been released! Check if at goal
                if dist_part_to_goal < self.release_distance:
                    # SUCCESS! Give massive reward
                    H_release = H_sys_norm
                    release_bonus = SUCCESS_BONUS
                    if H_release < 0.2:
                        reward += MILESTONE_BONUS # Momentum bonus
                        info["momentum_controlled_release"] = True
                    elif H_release > 0.5:
                        reward -= MILESTONE_BONUS # High momentum penalty
                        info["high_momentum_release"] = True
                    reward += release_bonus
                    info["success"] = True
                else:
                    # Critical-4: Allow re-grasp instead of terminal punishment loop
                    reward -= 20.0 # Recoverable penalty
                    self.grasped_part = False
                    self.milestone_first_grasp = False
                    self.milestone_reached_goal_area = False # Allow re-entering Phase 2
                    info["dropped_early"] = True
            
            # NOTE: Global momentum penalty (0.01 * |H_sys|) is already applied at line 608.
            # A duplicate 0.02 penalty was removed here to avoid 3x stacking in Stage 5.

        info.setdefault("success", False)
        info.setdefault("dropped_early", False)
        info.setdefault("at_goal_holding", False)
        info.setdefault("momentum_controlled_release", False)
        info.setdefault("high_momentum_release", False)
        
        self._prev_gripper_closed = self.gripper_closed
        return reward, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[8, 8, 5],
                cameraTargetPosition=[2, 0, 0],
                cameraUpVector=[0, 0, 1],
                physicsClientId=self.physics_client
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.33, nearVal=0.1, farVal=100,
                physicsClientId=self.physics_client
            )
            _, _, rgb, _, _ = p.getCameraImage(
                width=640, height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                physicsClientId=self.physics_client
            )
            return np.array(rgb)[:, :, :3]
        return None
    
    def close(self):
        if self.physics_client is not None:
            p.disconnect(physicsClientId=self.physics_client)
            self.physics_client = None


if __name__ == "__main__":
    # Test the environment
    print("Testing TrussAssemblyEnv...")
    
    for stage in [1, 2, 3, 4]:
        print(f"\n--- Curriculum Stage {stage} ---")
        env = TrussAssemblyEnv(render_mode=None, curriculum_stage=stage)
        obs, info = env.reset()
        
        print(f"Observation shape: {obs.shape}")
        print(f"Action shape: {env.action_space.shape}")
        
        total_reward = 0
        for i in range(300):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if i % 100 == 0:
                print(f"  Step {i}: reward={reward:.2f}, dist_part={info.get('dist_to_part', 0):.2f}")
            
            if terminated:
                print(f"  SUCCESS at step {i}!")
                break
            if truncated:
                break
        
        print(f"  Total reward: {total_reward:.2f}")
        env.close()
    
    print("\nTest complete!")
