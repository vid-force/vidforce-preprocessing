"""Tests of the record containers.
"""
from __future__ import annotations

import numpy as np
import pytest

from vidforce.data.hand_records import HandRecord


def _hand(depth_map, depth_K=None, depth_maps=None, depth_Ks=None, K=4):
    T = 16
    return HandRecord(
        rgb=np.zeros((K, 48, 64, 3), np.uint8), frame_times_ns=np.arange(K, dtype=np.float64),
        depth_map=depth_map, traj_m=np.zeros((T, 3), np.float32) + [[0, 0, 1.0]],
        waypoint_times_s=np.linspace(0, 1, T), force_valid=False,
        force_vec_n=np.zeros((T, 3), np.float32), peak_force_n=0.0,
        contact=np.zeros(T, bool), T_world_iphone=np.eye(4), prompt="open",
        provenance={}, depth_K=depth_K, depth_maps=depth_maps, depth_Ks=depth_Ks)


def test_sensor_depth_must_match_rgb_res():
    _hand(np.ones((48, 64), np.float32))
    with pytest.raises(ValueError):
        _hand(np.ones((12, 16), np.float32))


def test_da3_depth_at_own_res_with_K():
    K = np.array([[100.0, 0, 8], [0, 100.0, 6], [0, 0, 1]])
    r = _hand(np.ones((12, 16), np.float16), depth_K=K)
    assert r.depth_K.shape == (3, 3)
    with pytest.raises(ValueError):
        _hand(np.ones((12, 16), np.float16), depth_K=np.eye(2))


def test_depth_video_shapes():
    K = np.tile(np.eye(3), (4, 1, 1))
    maps = np.ones((4, 12, 16), np.float16)
    r = _hand(maps[0], depth_K=K[0], depth_maps=maps, depth_Ks=K)
    assert r.depth_maps.shape == (4, 12, 16)
    with pytest.raises(ValueError):
        _hand(maps[0], depth_K=K[0], depth_maps=maps[:3], depth_Ks=K[:3])
    with pytest.raises(ValueError):
        _hand(maps[0], depth_K=K[0], depth_maps=maps, depth_Ks=None)


def test_exo_record_accepts_depth_video():
    from vidforce.data.corpus_build import ExoRecord
    K, T = 4, 16
    r = ExoRecord(
        rgb=np.zeros((K, 48, 64, 3), np.uint8), frame_times_ns=np.arange(K, dtype=np.float64),
        prompt="open", traj_m=np.linspace([0, 0, 1.0], [0.3, 0, 1.0], T).astype(np.float32),
        force_vec_n=np.zeros((T, 3), np.float32),
        waypoint_times_s=np.linspace(0, 1, T), contact=np.zeros(T, bool), peak_force_n=1.0,
        T_world_iphone=np.eye(4), depth_map=np.ones((12, 16), np.float16), depth_K=np.eye(3),
        depth_maps=np.ones((K, 12, 16), np.float16), depth_Ks=np.tile(np.eye(3), (K, 1, 1)))
    assert r.depth_maps.shape[0] == K and r.depth_Ks.shape == (K, 3, 3)
