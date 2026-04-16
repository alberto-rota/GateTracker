"""
GateTracker - Self-supervised feature matching and point tracking for endoscopy.

Two-task framework:
  Task 1 (Pretrain): Descriptor learning via DINOv3 + gated hierarchical fusion + InfoNCE
  Task 2 (Tracking): Long-term dense point tracking via temporal refinement network
"""

__version__ = "0.1.0"
