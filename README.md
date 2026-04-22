# Zero-G Assembly

## Core Idea
This project aims to achieve **autonomous in-space assembly** using Deep Reinforcement Learning. The system controls a free-flying robot in a microgravity environment to perform precision manipulation tasks. 

**Current Scope**: The focus is on **single-part assembly**, where the agent must identify, grasp, transport, and insert a truss element into a target structure.

## Research Context
This project addresses the core question: **"How can RL agents control free-flying robotic arms to perform dexterous, long-horizon tasks in microgravity?"**

It specifically targets the challenge of **managing reaction forces and momentum** in a frictionless environment by combining:
- **7-DOF Redundant Manipulation**
- **Null Space Control** for disturbance minimization
- **Curriculum Learning** for long-horizon sequential tasks

## Environment & Agent
-   **Simulation**: Built on PyBullet with a custom gymnasium interface (`TrussAssemblyEnv`).
-   **Physics**: True zero-gravity ($g=0$) with realistic collision dynamics and momentum conservation.
-   **Robot**: A **60kg 6-DOF Servicer Bus** equipped with a **7-DOF Manipulator Arm** (Shoulder Y/P/R, Elbow P, Wrist Y/P/R).
-   **Dexterity**: The agent uses a simple **parallel-jaw gripper** with a constraint-based locking mechanism. There is no multi-fingered in-hand manipulation; the complexity lies in managing the coupled 13-DOF kinematics (Base + Arm) and drift inherent in space.
