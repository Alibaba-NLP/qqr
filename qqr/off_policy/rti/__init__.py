"""RTI (Rollout-Time Intervention): trajectory calibration at rollout time.

When every rollout of a prompt group fails, RTI re-generates trajectories with
the inspector enabled (groundtruth-guided calibration) and replaces failed
samples with successful calibrated ones.
"""
