"""Application services for simulation runs and sweep execution."""

from __future__ import annotations

from collections.abc import Callable

from src.coupled_solver import solve_self_consistent_plasma_circuit
from src.plasma import PlasmaCalculator
from src.plasma.constants import MM_TO_M
from src.spice import SpiceSimulator

from .models import FixedInputs, SimulationResult, SweepSpec


def run_single_simulation(inputs: FixedInputs) -> SimulationResult:
    """Run a single self-consistent plasma/circuit operating point."""
    chamber = inputs.to_chamber_conditions()
    plasma_conditions = inputs.to_plasma_conditions()
    plasma = PlasmaCalculator(
        chamber=chamber,
        plasma_conditions=plasma_conditions,
    )
    simulator = SpiceSimulator()
    coupled_result = solve_self_consistent_plasma_circuit(
        plasma=plasma,
        simulator=simulator,
        chamber=chamber,
        plasma_conditions=plasma_conditions,
    )
    plasma_result = coupled_result.plasma_result
    circuit_result = coupled_result.circuit_result

    electrode_radius_m = plasma.compute_effective_electrode_radius_m(chamber)
    bulk_plasma_height_m = (
        chamber.chamber_height_m
        - coupled_result.sheath_length_electrode_m
        - coupled_result.sheath_length_grounded_m
    )
    plasma_voltage_bias = plasma.compute_plasma_voltage_bias(
        current_density_a_per_m2=coupled_result.current_density_a_per_m2,
        rf_frequency_hz=plasma_conditions.RF_frequency,
        pressure_torr=chamber.pressure_torr,
        electron_temperature_ev=plasma_result.electron_temperature_ev,
        rf_power=plasma_conditions.RF_power,
        sheath_length_m=coupled_result.sheath_length_electrode_m,
        electrode_radius_m=electrode_radius_m,
        chamber_radius_m=chamber.chamber_radius_m,
        chamber_height_m=chamber.chamber_height_m,
        rf_voltage=circuit_result.source_voltage_peak,
    )
    plasma_bias_v_theta = plasma.compute_bias_V_theta(
        current_density_a_per_m2=coupled_result.current_density_a_per_m2,
        rf_frequency_hz=plasma_conditions.RF_frequency,
        pressure_torr=chamber.pressure_torr,
        electron_temperature_ev=plasma_result.electron_temperature_ev,
        rf_power=plasma_conditions.RF_power,
        sheath_length_m=coupled_result.sheath_length_electrode_m,
        electrode_radius_m=electrode_radius_m,
        chamber_radius_m=chamber.chamber_radius_m,
        chamber_height_m=chamber.chamber_height_m,
        rf_voltage=circuit_result.source_voltage_peak,
    )
    plasma_voltage_sheath_grounded = plasma.compute_voltage_sheath_grounded(
        current_density_a_per_m2=coupled_result.current_density_a_per_m2,
        rf_frequency_hz=plasma_conditions.RF_frequency,
        pressure_torr=chamber.pressure_torr,
        electron_temperature_ev=plasma_result.electron_temperature_ev,
        rf_power=plasma_conditions.RF_power,
        sheath_length_m=coupled_result.sheath_length_grounded_m,
        electrode_radius_m=electrode_radius_m,
        chamber_radius_m=chamber.chamber_radius_m,
        chamber_height_m=chamber.chamber_height_m,
        rf_voltage=circuit_result.source_voltage_peak,
    )
    plasma_voltage_sheath_electrode = plasma.compute_voltage_sheath_electrode(
        current_density_a_per_m2=coupled_result.current_density_a_per_m2,
        rf_frequency_hz=plasma_conditions.RF_frequency,
        pressure_torr=chamber.pressure_torr,
        electron_temperature_ev=plasma_result.electron_temperature_ev,
        rf_power=plasma_conditions.RF_power,
        sheath_length_m=coupled_result.sheath_length_electrode_m,
        electrode_radius_m=electrode_radius_m,
        chamber_radius_m=chamber.chamber_radius_m,
        chamber_height_m=chamber.chamber_height_m,
        rf_voltage=circuit_result.source_voltage_peak,
    )

    output_values: dict[str, float | bool | complex] = {
        "target_value_close_to_1": plasma_result.electron_temperature_target_value,
        "electron_temperature_iterations": plasma_result.electron_temperature_iterations,
        "coupled_converged": coupled_result.converged,
        "coupled_iterations": coupled_result.iterations,
        "sheath_length_relative_change": coupled_result.sheath_length_relative_change,
        "sheath_voltage_relative_change": coupled_result.sheath_voltage_relative_change,
        "bulk_power_relative_change": coupled_result.bulk_power_relative_change,
        "absorbed_bulk_power_w": coupled_result.absorbed_bulk_power_w,
        "self_consistent_electrode_sheath_length_m": coupled_result.sheath_length_electrode_m,
        "self_consistent_electrode_sheath_length_mm": coupled_result.sheath_length_electrode_m / MM_TO_M,
        "self_consistent_grounded_sheath_length_m": coupled_result.sheath_length_grounded_m,
        "self_consistent_grounded_sheath_length_mm": coupled_result.sheath_length_grounded_m / MM_TO_M,
        "electrode_radius_m": electrode_radius_m,
        "electrode_radius_mm": electrode_radius_m / MM_TO_M,
        "electrode_area_m2": plasma.compute_electrode_area_m2(chamber),
        "grounded_area_m2": plasma.compute_grounded_area_m2(chamber),
        "bulk_plasma_height_m": bulk_plasma_height_m,
        "bulk_plasma_height_mm": bulk_plasma_height_m / MM_TO_M,
        "current_density_a_per_m2": coupled_result.current_density_a_per_m2,
        "current_density_rms_a_per_m2": coupled_result.current_density_rms_a_per_m2,
        "electron_temperature_ev": plasma_result.electron_temperature_ev,
        "number_need_to_be_one": plasma_result.number_need_to_be_one,
        "elastic_collision_constant": plasma_result.elastic_collision_constant,
        "excitation_constant": plasma_result.excitation_constant,
        "debye_length_m": plasma_result.debye_length_m,
        "ionization_constant": plasma_result.ionization_constant,
        "bohm_velocity": plasma_result.bohm_velocity,
        "gas_number_density": plasma_result.gas_number_density,
        "effective_length": plasma_result.effective_length,
        "collision_energy_loss": plasma_result.collision_energy_loss,
        "electron_ion_energy_loss": plasma_result.electron_ion_energy_loss,
        "total_energy_loss": plasma_result.total_energy_loss,
        "plasma_total_sheath_voltage": plasma_result.sheath_voltage,
        "plasma_density": plasma_result.plasma_density,
        "ion_mean_free_path_m": plasma_result.ion_mean_free_path_m,
        "collisional_frequency": plasma_result.collisional_frequency,
        "plasma_angular_frequency": plasma_result.plasma_angular_frequency,
        "plasma_conductivity": plasma_result.plasma_conductivity,
        "plasma_relative_permittivity": plasma_result.plasma_relative_permittivity,
        "plasma_resistance": plasma_result.plasma_resistance,
        "plasma_coil_reactance": plasma_result.plasma_coil_reactance,
        "plasma_capacitive_reactance": plasma_result.plasma_cap_reactance,
        "plasma_coil_inductance_h": plasma_result.plasma_coil_henry,
        "plasma_capacitance_f": plasma_result.plasma_cap_farad,
        "plasma_sheath_capacitance_f_per_m2": plasma_result.plasma_sheath_capacitance,
        "plasma_sheath_capacitance_electrode_f": plasma_result.plasma_sheath_capacitance_electrode,
        "plasma_sheath_capacitance_grounded_f": plasma_result.plasma_sheath_capacitance_grounded,
        "electron_velocity": plasma_result.electron_velocity,
        "plasma_sheath_conductance_s_per_m2": plasma_result.plasma_sheath_conductance,
        "plasma_sheath_resistance_electrode_ohm": plasma_result.plasma_sheath_resistance_electrode,
        "plasma_sheath_resistance_grounded_ohm": plasma_result.plasma_sheath_resistance_grounded,
        "plasma_wall_potential_v": plasma_result.plasma_wall_potential,
        "plasma_target_power_w": circuit_result.target_power_w,
        "plasma_source_voltage_peak_v": circuit_result.source_voltage_peak,
        "plasma_source_voltage_rms_v": circuit_result.source_voltage_rms,
        "plasma_voltage_bias_v": plasma_voltage_bias,
        "plasma_bias_v_theta_rad": plasma_bias_v_theta,
        "plasma_voltage_sheath_grounded_v": plasma_voltage_sheath_grounded,
        "plasma_voltage_sheath_electrode_v": plasma_voltage_sheath_electrode,
        "plasma_bulk_impedance_ohm": circuit_result.bulk_plasma_impedance,
        "plasma_grounded_sheath_impedance_ohm": circuit_result.grounded_sheath_impedance,
        "plasma_total_impedance_ohm": circuit_result.total_impedance,
        "plasma_source_current_rms_a": circuit_result.source_current_rms,
        "plasma_src_node_current_rms_a": circuit_result.src_node_current_rms,
        "plasma_src_node_resistor_current_rms_a": circuit_result.src_node_resistor_current_rms,
        "plasma_src_node_capacitor_current_rms_a": circuit_result.src_node_capacitor_current_rms,
        "electrode_sheath_resistor_power_w": circuit_result.electrode_sheath_resistor_power_w,
        "plasma_resistance_power_w": circuit_result.plasma_resistance_power_w,
        "grounded_sheath_resistor_power_w": circuit_result.grounded_sheath_resistor_power_w,
        "total_resistor_power_w": circuit_result.total_resistor_power_w,
        "plasma_average_power_w": circuit_result.average_power_w,
    }

    return SimulationResult(
        inputs=inputs,
        chamber=chamber,
        plasma_conditions=plasma_conditions,
        plasma_result=plasma_result,
        circuit_result=circuit_result,
        coupled_result=coupled_result,
        output_values=output_values,
    )


def run_parameter_sweep(
    fixed_inputs: FixedInputs,
    sweep_spec: SweepSpec,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> list[SimulationResult]:
    """Run a one-dimensional sweep while keeping all other inputs fixed."""
    results: list[SimulationResult] = []
    sweep_values = sweep_spec.values()
    total_points = len(sweep_values)
    for index, sweep_value in enumerate(sweep_values, start=1):
        run_inputs = fixed_inputs.with_value(sweep_spec.variable_name, sweep_value)
        results.append(run_single_simulation(run_inputs))
        if progress_callback is not None:
            progress_callback(index, total_points, sweep_value)
    return results
