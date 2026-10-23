"""
memory_trp
==========
Modeling memory in time-respecting paths on temporal networks.

Reference implementation of:
  Guerrini, Cattuto, Dall'Amico,
  'Modeling memory in time-respecting paths on temporal networks',
  EPJ Data Science (2026).
"""

from __future__ import annotations

__version__ = "0.1.0"

from memory_trp.aggregation import TemporalGraph, build_temporal_graph, split_by_day
from memory_trp.bootstrap import bootstrap_p
from memory_trp.loading import load_contacts
from memory_trp.models import (
    FitResult,
    SBMFitResult,
    fit_mem_fixed_point,
    fit_mem_numeric,
    fit_mem_sbm,
    fit_p_per_group,
)
from memory_trp.records import Records, build_records
from memory_trp.sir import SIRResult, calibrate_rate, simulate_sir
from memory_trp.timevarying import fit_p_by_time_bin
from memory_trp.trp import generate_many_trps, generate_trp

__all__ = [
    "TemporalGraph",
    "build_temporal_graph",
    "split_by_day",
    "bootstrap_p",
    "load_contacts",
    "FitResult",
    "SBMFitResult",
    "fit_mem_fixed_point",
    "fit_mem_numeric",
    "fit_mem_sbm",
    "fit_p_per_group",
    "Records",
    "build_records",
    "SIRResult",
    "calibrate_rate",
    "simulate_sir",
    "fit_p_by_time_bin",
    "generate_many_trps",
    "generate_trp",
]
