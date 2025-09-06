# Inverse Reinforcement Learning Experiments (AIRL & MAIRL)

This project explores **inverse reinforcement learning (IRL)**, specifically **Adversarial Inverse Reinforcement Learning (AIRL)** and attempts at **Multi-Agent IRL (MAIRL)**.  
We tested IRL in multiple environments and documented the successes and challenges, particularly around integrating **Gym**, **PettingZoo**, and the **imitation** library.

---

## Overview

Inverse reinforcement learning (IRL) is the process of recovering a reward function by observing expert behavior rather than relying on predefined reward signals.  

In this project, we:

- Implemented **PPO** (Proximal Policy Optimization) as a baseline.
- Ran experiments on:
  - **CartPole** (AIRL successfully trained).
  - **Multiwalker** (multi-agent PPO training, with interesting emergent behaviors).
  - **PettingZoo environments** (attempted AIRL/MAIRL but faced library incompatibilities).
- Investigated environment conversion challenges between **Gym** and **PettingZoo**.

---

## Project Structure

