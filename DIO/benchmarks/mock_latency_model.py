"""
Calibrated mock inference latency for heterogeneity experiments.

Models latency as: TTFT = intercept + prefill_slope * prompt_tokens
                   decode = decode_slope * output_tokens
with multiplicative jitter and optional thermal-throttle ramp.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional


PROFILES_PATH = os.path.join(os.path.dirname(__file__), "heterogeneity_profiles.json")


@dataclass
class ThermalThrottle:
    enabled: bool = False
    start_request: int = 40
    ramp_per_request: float = 0.02
    max_factor: float = 1.4

    def factor(self, request_idx: int) -> float:
        if not self.enabled or request_idx < self.start_request:
            return 1.0
        extra = (request_idx - self.start_request) * self.ramp_per_request
        return min(self.max_factor, 1.0 + extra)


@dataclass
class LatencyProfile:
    name: str
    gpu: str = "emulated"
    ttft_intercept_ms: float = 180.0
    prefill_slope_ms_per_token: float = 0.5
    decode_slope_ms_per_token: float = 12.0
    jitter_pct: float = 0.10
    thermal: ThermalThrottle = field(default_factory=ThermalThrottle)
    latency_mult: float = 1.0

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any], latency_mult: float = 1.0) -> "LatencyProfile":
        th = d.get("thermal_throttle") or {}
        return cls(
            name=name,
            gpu=d.get("gpu", "emulated"),
            ttft_intercept_ms=float(d.get("ttft_intercept_ms", 180)),
            prefill_slope_ms_per_token=float(d.get("prefill_slope_ms_per_token", 0.5)),
            decode_slope_ms_per_token=float(d.get("decode_slope_ms_per_token", 12)),
            jitter_pct=float(d.get("jitter_pct", 0.10)),
            thermal=ThermalThrottle(
                enabled=bool(th.get("enabled", False)),
                start_request=int(th.get("start_request", 40)),
                ramp_per_request=float(th.get("ramp_per_request", 0.02)),
                max_factor=float(th.get("max_factor", 1.4)),
            ),
            latency_mult=latency_mult,
        )


def load_profiles(path: str = PROFILES_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_profile(
    profile_name: Optional[str] = None,
    profile_role: Optional[str] = None,
    pairing_name: Optional[str] = None,
    latency_mult: float = 1.0,
    profiles_path: str = PROFILES_PATH,
) -> LatencyProfile:
    """Resolve a LatencyProfile from --latency-profile / --profile-role / pairing."""
    data = load_profiles(profiles_path)
    profiles = data.get("profiles", {})

    if pairing_name and pairing_name in data.get("pairings", {}):
        pairing = data["pairings"][pairing_name]
        role = (profile_role or "slow").lower()
        key = pairing["fast"] if role == "fast" else pairing["slow"]
        if key not in profiles:
            raise KeyError(f"Profile '{key}' missing for pairing '{pairing_name}'")
        return LatencyProfile.from_dict(key, profiles[key], latency_mult)

    if profile_name:
        if profile_name in data.get("pairings", {}):
            return resolve_profile(
                pairing_name=profile_name,
                profile_role=profile_role or "slow",
                latency_mult=latency_mult,
                profiles_path=profiles_path,
            )
        if profile_name in profiles:
            return LatencyProfile.from_dict(profile_name, profiles[profile_name], latency_mult)

    # Legacy uniform mock (backward compatible)
    base_decode = 10.0 * max(1.0, latency_mult)
    return LatencyProfile(
        name="legacy_uniform",
        ttft_intercept_ms=50.0 * max(1.0, latency_mult),
        decode_slope_ms_per_token=base_decode,
        jitter_pct=0.05,
    )


def estimate_prompt_tokens(prompt: str) -> int:
    return max(1, len(prompt) // 4)


@dataclass
class MockInferenceResult:
    ttft_ms: float
    total_latency_ms: float
    tokens_generated: int
    profile_name: str
    thermal_factor: float


class MockLatencySimulator:
    def __init__(self, profile: LatencyProfile, seed: Optional[int] = None):
        self.profile = profile
        self.request_count = 0
        if seed is not None:
            random.seed(seed)

    def predict(self, prompt: str, output_len: int = 128) -> MockInferenceResult:
        self.request_count += 1
        p = self.profile
        prompt_tokens = estimate_prompt_tokens(prompt)

        ttft = p.ttft_intercept_ms + p.prefill_slope_ms_per_token * prompt_tokens
        decode_ms = p.decode_slope_ms_per_token * output_len
        base_ms = ttft + decode_ms

        noise = random.gauss(1.0, p.jitter_pct)
        noise = max(0.65, min(1.45, noise))
        base_ms *= noise
        ttft *= noise

        thermal = p.thermal.factor(self.request_count)
        base_ms *= thermal
        ttft *= thermal
        base_ms *= p.latency_mult
        ttft *= p.latency_mult

        return MockInferenceResult(
            ttft_ms=ttft,
            total_latency_ms=base_ms,
            tokens_generated=output_len,
            profile_name=p.name,
            thermal_factor=thermal,
        )

    def execute(self, prompt: str, output_len: int = 128) -> MockInferenceResult:
        """Sleep for simulated latency (used by mock workers)."""
        result = self.predict(prompt, output_len)
        time.sleep(result.total_latency_ms / 1000.0)
        return result