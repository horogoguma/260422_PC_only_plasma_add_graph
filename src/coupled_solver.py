"""Coupled plasma and circuit iteration helpers."""

from dataclasses import dataclass, replace
from math import sqrt
from typing import Literal

from .plasma import ChamberConditions, PlasmaCalculator, PlasmaComputationResult, PlasmaConditions
from .spice import PlasmaCircuitParameters, PlasmaCircuitResult, SpiceSimulator

MIN_BULK_HEIGHT_M = 0.5e-3
RFDriveMode = Literal["power", "current"]


class CoupledSolverConvergenceError(ValueError):
    """Raised when the coupled plasma-circuit iteration fails to converge."""


@dataclass(frozen=True)
class SelfConsistentPlasmaCircuitResult:
    """Final outputs from the coupled plasma-circuit iteration."""

    plasma_result: PlasmaComputationResult
    circuit_result: PlasmaCircuitResult
    current_density_a_per_m2: float
    current_density_rms_a_per_m2: float
    sheath_length_electrode_m: float
    sheath_length_grounded_m: float
    iterations: int
    converged: bool
    sheath_length_relative_change: float
    sheath_voltage_relative_change: float
    bulk_power_relative_change: float
    absorbed_bulk_power_w: float


def _validate_positive_bulk_height(
    electrode_sheath_m: float,
    grounded_sheath_m: float,
    chamber_height_m: float,
) -> None:
    """Reject sheath pairs that consume the whole chamber."""
    bulk_height_m = chamber_height_m - electrode_sheath_m - grounded_sheath_m
    if bulk_height_m <= MIN_BULK_HEIGHT_M:
        raise ValueError(
            "Coupled solver predicted sheath lengths that leave too little bulk plasma. "
            f"chamber_height={chamber_height_m * 1e3:g} mm, "
            f"electrode_sheath={electrode_sheath_m * 1e3:g} mm, "
            f"grounded_sheath={grounded_sheath_m * 1e3:g} mm, "
            f"bulk_height={bulk_height_m * 1e3:g} mm."
        )


def _has_positive_bulk_height(
    electrode_sheath_m: float,
    grounded_sheath_m: float,
    chamber_height_m: float,
) -> bool:
    """Return whether sheath lengths leave enough bulk height."""
    return chamber_height_m - electrode_sheath_m - grounded_sheath_m > MIN_BULK_HEIGHT_M


def solve_self_consistent_plasma_circuit(
    plasma: PlasmaCalculator,
    simulator: SpiceSimulator,
    chamber: ChamberConditions,
    plasma_conditions: PlasmaConditions,
    rf_drive_mode: RFDriveMode = "power",
    target_current_rms_a: float | None = None,
    max_iterations: int = 1000,
    min_iterations: int = 30,
    min_sheath_hold_iterations: int = 20,
    max_sheath_hold_iterations: int = 60,
    pre_sheath_relative_tolerance: float = 1e-2,
    pre_sheath_stable_iterations: int = 5,
    relative_tolerance: float = 1e-2,
    damping: float = 0.5,
    sheath_damping: float = 0.05,
) -> SelfConsistentPlasmaCircuitResult:
    """Iterate until plasma sheath lengths and sheath voltage are self-consistent."""
    if rf_drive_mode not in ("power", "current"):
        raise ValueError("rf_drive_mode must be either 'power' or 'current'.")
    if rf_drive_mode == "current":
        if target_current_rms_a is None or target_current_rms_a <= 0:
            raise ValueError("target_current_rms_a must be positive in current mode.")
    if plasma.compute_electrode_area_m2(chamber) <= 0:
        raise ValueError("Electrode area must be positive to compute current density.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if min_iterations <= 0:
        raise ValueError("min_iterations must be positive.")
    if min_iterations > max_iterations:
        raise ValueError("min_iterations cannot exceed max_iterations.")
    if min_sheath_hold_iterations < 0:
        raise ValueError("min_sheath_hold_iterations must be non-negative.")
    if min_sheath_hold_iterations > max_iterations:
        raise ValueError("min_sheath_hold_iterations cannot exceed max_iterations.")
    if max_sheath_hold_iterations < min_sheath_hold_iterations:
        raise ValueError(
            "max_sheath_hold_iterations cannot be less than min_sheath_hold_iterations."
        )
    if max_sheath_hold_iterations > max_iterations:
        raise ValueError("max_sheath_hold_iterations cannot exceed max_iterations.")
    if pre_sheath_relative_tolerance <= 0:
        raise ValueError("pre_sheath_relative_tolerance must be positive.")
    if pre_sheath_stable_iterations <= 0:
        raise ValueError("pre_sheath_stable_iterations must be positive.")
    if not (0 < damping <= 1):
        raise ValueError("damping must be in the interval (0, 1].")
    if not (0 < sheath_damping <= 1):
        raise ValueError("sheath_damping must be in the interval (0, 1].")

    _validate_positive_bulk_height(
        plasma_conditions.sheath_length_electrode_m,
        plasma_conditions.sheath_length_grounded_m,
        chamber.chamber_height_m,
    )
    working_conditions = replace(
        plasma_conditions,
    )
    initial_sheath_length_electrode_m = plasma_conditions.sheath_length_electrode_m
    initial_sheath_length_grounded_m = plasma_conditions.sheath_length_grounded_m
    last_relative_change = float("inf")
    last_sheath_voltage_relative_change = float("inf")
    last_bulk_power_relative_change = float("inf")
    converged = False
    pre_sheath_stable_count = 0
    sheath_updates_enabled = False

    if working_conditions.absorbed_bulk_power_w is None:
        working_conditions = replace(
            working_conditions,
            absorbed_bulk_power_w=working_conditions.RF_power,
        )

    for iteration in range(1, max_iterations + 1):
        plasma_result = plasma.compute_plasma_properties(
            chamber=chamber,
            plasma_conditions=working_conditions,
        )
        plasma_circuit = PlasmaCircuitParameters(
            plasma_resistance=plasma_result.plasma_resistance,
            plasma_coil_henry=plasma_result.plasma_coil_henry,
            plasma_cap_farad=plasma_result.plasma_cap_farad,
            plasma_sheath_capacitance_electrode=plasma_result.plasma_sheath_capacitance_electrode,
            plasma_sheath_capacitance_grounded=plasma_result.plasma_sheath_capacitance_grounded,
            plasma_sheath_resistance_electrode=plasma_result.plasma_sheath_resistance_electrode,
            plasma_sheath_resistance_grounded=plasma_result.plasma_sheath_resistance_grounded,
            rf_frequency_hz=working_conditions.RF_frequency,
        )
        if rf_drive_mode == "power":
            simulator.build_plasma_equivalent_circuit(
                plasma_circuit,
                target_power_w=working_conditions.RF_power,
            )
        else:
            simulator.build_plasma_equivalent_circuit_for_current(
                plasma_circuit,
                target_current_rms_a=target_current_rms_a,
            )
        circuit_result = simulator.compute_plasma_circuit_response()
        updated_rf_power = (
            working_conditions.RF_power
            if rf_drive_mode == "power"
            else (1 - damping) * working_conditions.RF_power
            + damping * circuit_result.total_resistor_power_w
        )
        updated_absorbed_bulk_power_w = (
            (1 - damping) * working_conditions.absorbed_bulk_power_w
            + damping * circuit_result.plasma_resistance_power_w
        )

        current_density_electrode_rms, current_density_grounded_rms = (
            plasma.compute_sheath_current_densities(
                current_a=circuit_result.src_node_current_rms,
                chamber=chamber,
            )
        )
        current_density_electrode = current_density_electrode_rms * sqrt(2.0)
        current_density_grounded = current_density_grounded_rms * sqrt(2.0)
        current_density_a_per_m2 = current_density_electrode
        total_sheath_voltage = plasma.compute_voltage_sheath_total_sum(
            current_density_a_per_m2=current_density_electrode,
            rf_frequency_hz=working_conditions.RF_frequency,
            pressure_torr=chamber.pressure_torr,
            electron_temperature_ev=plasma_result.electron_temperature_ev,
            rf_power=working_conditions.RF_power,
            sheath_length_m=working_conditions.sheath_length_electrode_m,
            electrode_radius_m=plasma.compute_effective_electrode_radius_m(chamber),
            chamber_radius_m=chamber.chamber_radius_m,
            chamber_height_m=chamber.chamber_height_m,
            rf_voltage=circuit_result.source_voltage_peak,
        )
        updated_sheath_voltage = (
            (1 - sheath_damping) * working_conditions.sheath_voltage
            + sheath_damping * total_sheath_voltage
        )
        last_sheath_voltage_relative_change = abs(
            total_sheath_voltage - working_conditions.sheath_voltage
        ) / max(abs(working_conditions.sheath_voltage), 1e-30)
        last_bulk_power_relative_change = abs(
            updated_absorbed_bulk_power_w - working_conditions.absorbed_bulk_power_w
        ) / max(abs(working_conditions.absorbed_bulk_power_w), 1e-30)
        if (
            last_sheath_voltage_relative_change < pre_sheath_relative_tolerance
            and last_bulk_power_relative_change < pre_sheath_relative_tolerance
        ):
            pre_sheath_stable_count += 1
        else:
            pre_sheath_stable_count = 0
        if (
            not sheath_updates_enabled
            and iteration >= min_sheath_hold_iterations
            and (
                pre_sheath_stable_count >= pre_sheath_stable_iterations
                or iteration >= max_sheath_hold_iterations
            )
        ):
            sheath_updates_enabled = True

        geometry_limited = False
        if not sheath_updates_enabled:
            updated_sheath_length_electrode_m = initial_sheath_length_electrode_m
            updated_sheath_length_grounded_m = initial_sheath_length_grounded_m
        else:
            raw_sheath_length_electrode_m = plasma.compute_plasma_sheath_length_electrode(
                current_density_a_per_m2=current_density_electrode,
                rf_frequency_hz=working_conditions.RF_frequency,
                pressure_torr=chamber.pressure_torr,
                electron_temperature_ev=plasma_result.electron_temperature_ev,
                rf_power=working_conditions.RF_power,
                sheath_voltage=updated_sheath_voltage,
                chamber_radius_m=chamber.chamber_radius_m,
                chamber_height_m=chamber.chamber_height_m,
            )
            raw_sheath_length_grounded_m = plasma.compute_plasma_sheath_length_grounded(
                current_density_a_per_m2=current_density_grounded,
                rf_frequency_hz=working_conditions.RF_frequency,
                pressure_torr=chamber.pressure_torr,
                electron_temperature_ev=plasma_result.electron_temperature_ev,
                rf_power=working_conditions.RF_power,
                sheath_voltage=updated_sheath_voltage,
                chamber_radius_m=chamber.chamber_radius_m,
                chamber_height_m=chamber.chamber_height_m,
            )
            current_electrode_sheath_m = working_conditions.sheath_length_electrode_m
            current_grounded_sheath_m = working_conditions.sheath_length_grounded_m
            sheath_step = sheath_damping
            updated_sheath_length_electrode_m = (
                (1 - sheath_step) * current_electrode_sheath_m
                + sheath_step * raw_sheath_length_electrode_m
            )
            updated_sheath_length_grounded_m = (
                (1 - sheath_step) * current_grounded_sheath_m
                + sheath_step * raw_sheath_length_grounded_m
            )
            while not _has_positive_bulk_height(
                updated_sheath_length_electrode_m,
                updated_sheath_length_grounded_m,
                chamber.chamber_height_m,
            ):
                geometry_limited = True
                sheath_step *= 0.5
                if sheath_step < 1e-9:
                    updated_sheath_length_electrode_m = current_electrode_sheath_m
                    updated_sheath_length_grounded_m = current_grounded_sheath_m
                    break
                updated_sheath_length_electrode_m = (
                    (1 - sheath_step) * current_electrode_sheath_m
                    + sheath_step * raw_sheath_length_electrode_m
                )
                updated_sheath_length_grounded_m = (
                    (1 - sheath_step) * current_grounded_sheath_m
                    + sheath_step * raw_sheath_length_grounded_m
                )
        last_relative_change = abs(
            updated_sheath_length_electrode_m - working_conditions.sheath_length_electrode_m
        ) / max(abs(working_conditions.sheath_length_electrode_m), 1e-30)
        working_conditions = replace(
            working_conditions,
            sheath_length_electrode_m=updated_sheath_length_electrode_m,
            sheath_length_grounded_m=updated_sheath_length_grounded_m,
            sheath_voltage=updated_sheath_voltage,
            Current_density=current_density_a_per_m2,
            electron_temperature_ev=plasma_result.electron_temperature_ev,
            rf_voltage=circuit_result.source_voltage_peak,
            RF_power=updated_rf_power,
            absorbed_bulk_power_w=updated_absorbed_bulk_power_w,
        )

        if (
            iteration >= min_iterations
            and sheath_updates_enabled
            and not geometry_limited
            and last_relative_change < relative_tolerance
            and last_sheath_voltage_relative_change < relative_tolerance
            and last_bulk_power_relative_change < relative_tolerance
        ):
            converged = True
            break

    if not converged:
        raise CoupledSolverConvergenceError(
            "Coupled solver did not converge after "
            f"{max_iterations} iterations. "
            f"relative sheath change={last_relative_change:g}, "
            "relative sheath voltage change="
            f"{last_sheath_voltage_relative_change:g}, "
            f"relative bulk power change={last_bulk_power_relative_change:g}."
        )

    final_plasma_result = plasma.compute_plasma_properties(
        chamber=chamber,
        plasma_conditions=working_conditions,
    )
    final_plasma_circuit = PlasmaCircuitParameters(
        plasma_resistance=final_plasma_result.plasma_resistance,
        plasma_coil_henry=final_plasma_result.plasma_coil_henry,
        plasma_cap_farad=final_plasma_result.plasma_cap_farad,
        plasma_sheath_capacitance_electrode=final_plasma_result.plasma_sheath_capacitance_electrode,
        plasma_sheath_capacitance_grounded=final_plasma_result.plasma_sheath_capacitance_grounded,
        plasma_sheath_resistance_electrode=final_plasma_result.plasma_sheath_resistance_electrode,
        plasma_sheath_resistance_grounded=final_plasma_result.plasma_sheath_resistance_grounded,
        rf_frequency_hz=working_conditions.RF_frequency,
    )
    if rf_drive_mode == "power":
        simulator.build_plasma_equivalent_circuit(
            final_plasma_circuit,
            target_power_w=working_conditions.RF_power,
        )
    else:
        simulator.build_plasma_equivalent_circuit_for_current(
            final_plasma_circuit,
            target_current_rms_a=target_current_rms_a,
        )
    final_circuit_result = simulator.compute_plasma_circuit_response()
    final_current_density_electrode_rms, final_current_density_grounded_rms = (
        plasma.compute_sheath_current_densities(
            current_a=final_circuit_result.src_node_current_rms,
            chamber=chamber,
        )
    )
    final_current_density_electrode = final_current_density_electrode_rms * sqrt(2.0)
    final_current_density_grounded = final_current_density_grounded_rms * sqrt(2.0)
    final_total_sheath_voltage = plasma.compute_voltage_sheath_total_sum(
        current_density_a_per_m2=final_current_density_electrode,
        rf_frequency_hz=working_conditions.RF_frequency,
        pressure_torr=chamber.pressure_torr,
        electron_temperature_ev=final_plasma_result.electron_temperature_ev,
        rf_power=working_conditions.RF_power,
        sheath_length_m=working_conditions.sheath_length_electrode_m,
        electrode_radius_m=plasma.compute_effective_electrode_radius_m(chamber),
        chamber_radius_m=chamber.chamber_radius_m,
        chamber_height_m=chamber.chamber_height_m,
        rf_voltage=final_circuit_result.source_voltage_peak,
    )
    final_updated_sheath_voltage = (
        (1 - damping) * working_conditions.sheath_voltage
        + damping * final_total_sheath_voltage
    )
    working_conditions = replace(
        working_conditions,
        Current_density=final_current_density_electrode,
        sheath_voltage=final_updated_sheath_voltage,
        rf_voltage=final_circuit_result.source_voltage_peak,
        RF_power=(
            working_conditions.RF_power
            if rf_drive_mode == "power"
            else final_circuit_result.total_resistor_power_w
        ),
        absorbed_bulk_power_w=final_circuit_result.plasma_resistance_power_w,
    )
    final_plasma_result = plasma.compute_plasma_properties(
        chamber=chamber,
        plasma_conditions=working_conditions,
    )
    final_plasma_circuit = PlasmaCircuitParameters(
        plasma_resistance=final_plasma_result.plasma_resistance,
        plasma_coil_henry=final_plasma_result.plasma_coil_henry,
        plasma_cap_farad=final_plasma_result.plasma_cap_farad,
        plasma_sheath_capacitance_electrode=final_plasma_result.plasma_sheath_capacitance_electrode,
        plasma_sheath_capacitance_grounded=final_plasma_result.plasma_sheath_capacitance_grounded,
        plasma_sheath_resistance_electrode=final_plasma_result.plasma_sheath_resistance_electrode,
        plasma_sheath_resistance_grounded=final_plasma_result.plasma_sheath_resistance_grounded,
        rf_frequency_hz=working_conditions.RF_frequency,
    )
    if rf_drive_mode == "power":
        simulator.build_plasma_equivalent_circuit(
            final_plasma_circuit,
            target_power_w=working_conditions.RF_power,
        )
    else:
        simulator.build_plasma_equivalent_circuit_for_current(
            final_plasma_circuit,
            target_current_rms_a=target_current_rms_a,
        )
    final_circuit_result = simulator.compute_plasma_circuit_response()
    final_current_density_electrode_rms, final_current_density_grounded_rms = (
        plasma.compute_sheath_current_densities(
            current_a=final_circuit_result.src_node_current_rms,
            chamber=chamber,
        )
    )
    final_current_density_electrode = final_current_density_electrode_rms * sqrt(2.0)
    final_current_density_grounded = final_current_density_grounded_rms * sqrt(2.0)

    return SelfConsistentPlasmaCircuitResult(
        plasma_result=final_plasma_result,
        circuit_result=final_circuit_result,
        current_density_a_per_m2=final_current_density_electrode,
        current_density_rms_a_per_m2=final_current_density_electrode_rms,
        sheath_length_electrode_m=working_conditions.sheath_length_electrode_m,
        sheath_length_grounded_m=working_conditions.sheath_length_grounded_m,
        iterations=iteration,
        converged=converged,
        sheath_length_relative_change=last_relative_change,
        sheath_voltage_relative_change=last_sheath_voltage_relative_change,
        bulk_power_relative_change=last_bulk_power_relative_change,
        absorbed_bulk_power_w=working_conditions.absorbed_bulk_power_w,
    )
