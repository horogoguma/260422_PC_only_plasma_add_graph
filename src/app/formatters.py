"""Formatting helpers for CLI-style simulation summaries."""

from __future__ import annotations

from .models import SimulationResult


def format_simulation_result(result: SimulationResult) -> str:
    """Render the full human-readable simulation summary."""
    values = result.output_values
    lines = [
        (
            "Target value close to 1: "
            f"{values['target_value_close_to_1']} "
            f"(iterations: {values['electron_temperature_iterations']})"
        ),
        (
            "Coupled iteration converged: "
            f"{values['coupled_converged']} "
            f"(iterations: {values['coupled_iterations']}, "
            f"relative sheath change: {values['sheath_length_relative_change']}, "
            f"relative sheath voltage change: {values['sheath_voltage_relative_change']})"
        ),
        (
            "RF drive mode: "
            f"{'current' if values['rf_drive_mode_is_current'] else 'power'} "
            f"(current setpoint: {values['rf_current_rms_setpoint_a']} A rms)"
        ),
        (
            "Absorbed bulk power: "
            f"{values['absorbed_bulk_power_w']} W "
            f"(relative change: {values['bulk_power_relative_change']})"
        ),
        (
            "Self-consistent electrode sheath length: "
            f"{values['self_consistent_electrode_sheath_length_m']} m"
        ),
        (
            "Self-consistent grounded sheath length: "
            f"{values['self_consistent_grounded_sheath_length_m']} m"
        ),
        f"Electrode radius: {values['electrode_radius_m']} m",
        f"Electrode area: {values['electrode_area_m2']} m^2",
        f"Grounded area: {values['grounded_area_m2']} m^2",
        f"Bulk plasma height: {values['bulk_plasma_height_m']} m",
        (
            "Self-consistent sheath current density (peak): "
            f"{values['current_density_a_per_m2']} A/m^2"
        ),
        (
            "Self-consistent sheath current density (rms): "
            f"{values['current_density_rms_a_per_m2']} A/m^2"
        ),
        f"Computed electron temperature: {values['electron_temperature_ev']} eV",
        f"Number that should be 1: {values['number_need_to_be_one']}",
        f"Elastic collision constant: {values['elastic_collision_constant']}",
        f"Excitation constant: {values['excitation_constant']}",
        f"Debye length: {values['debye_length_m']} m",
        f"Ionization constant: {values['ionization_constant']}",
        f"Bohm velocity: {values['bohm_velocity']} m/s",
        f"Gas number density: {values['gas_number_density']} m^-3",
        f"Effective length: {values['effective_length']} m",
        f"Collision energy loss: {values['collision_energy_loss']} eV",
        f"Electron-ion energy loss: {values['electron_ion_energy_loss']} eV",
        f"Total energy loss: {values['total_energy_loss']} eV",
        f"Plasma total sheath voltage: {values['plasma_total_sheath_voltage']} V",
        f"Plasma density: {values['plasma_density']} m^-3",
        f"Ion mean free path: {values['ion_mean_free_path_m']} m",
        f"Collisional frequency: {values['collisional_frequency']} /s",
        f"Plasma angular frequency: {values['plasma_angular_frequency']} rad/s",
        f"Plasma conductivity: {values['plasma_conductivity']} S/m",
        f"Plasma relative permittivity: {values['plasma_relative_permittivity']}",
        f"Plasma resistance: {values['plasma_resistance']} ohm",
        (
            "Bulk plasma electron-inertia inductive reactance (XL at RF): "
            f"{values['plasma_coil_reactance']} ohm"
        ),
        (
            "Bulk plasma space capacitive reactance (XC at RF): "
            f"{values['plasma_capacitive_reactance']} ohm"
        ),
        (
            "Bulk plasma electron-inertia equivalent inductance: "
            f"{values['plasma_coil_inductance_h']} H"
        ),
        (
            "Bulk plasma space equivalent capacitance: "
            f"{values['plasma_capacitance_f']} F"
        ),
        f"Plasma sheath capacitance: {values['plasma_sheath_capacitance_f_per_m2']} F/m^2",
        (
            "Plasma sheath capacitance (electrode): "
            f"{values['plasma_sheath_capacitance_electrode_f']} F"
        ),
        (
            "Plasma sheath capacitance (grounded): "
            f"{values['plasma_sheath_capacitance_grounded_f']} F"
        ),
        f"Electron velocity: {values['electron_velocity']} m/s",
        (
            "Plasma sheath conductance: "
            f"{values['plasma_sheath_conductance_s_per_m2']} S/m^2"
        ),
        (
            "Plasma sheath resistance (electrode): "
            f"{values['plasma_sheath_resistance_electrode_ohm']} ohm"
        ),
        (
            "Plasma sheath resistance (grounded): "
            f"{values['plasma_sheath_resistance_grounded_ohm']} ohm"
        ),
        f"Plasma wall potential: {values['plasma_wall_potential_v']} V",
        f"Plasma target power: {values['plasma_target_power_w']} W",
        f"Plasma target current rms: {values['plasma_target_current_rms_a']} A",
        f"Plasma source voltage peak: {values['plasma_source_voltage_peak_v']} V",
        f"Plasma source voltage rms: {values['plasma_source_voltage_rms_v']} V",
        f"Plasma voltage bias: {values['plasma_voltage_bias_v']} V",
        f"Plasma bias V_theta: {values['plasma_bias_v_theta_rad']} rad",
        f"Plasma voltage sheath grounded: {values['plasma_voltage_sheath_grounded_v']} V",
        f"Plasma voltage sheath electrode: {values['plasma_voltage_sheath_electrode_v']} V",
        f"Bulk plasma equivalent impedance: {values['plasma_bulk_impedance_ohm']} ohm",
        (
            "Plasma grounded sheath impedance: "
            f"{values['plasma_grounded_sheath_impedance_ohm']} ohm"
        ),
        f"Plasma total impedance: {values['plasma_total_impedance_ohm']} ohm",
        f"Plasma source current rms: {values['plasma_source_current_rms_a']} A",
        f"Plasma src node current rms: {values['plasma_src_node_current_rms_a']} A",
        (
            "Electrode sheath series resistor current rms: "
            f"{values['plasma_src_node_resistor_current_rms_a']} A"
        ),
        (
            "Electrode sheath series capacitor current rms: "
            f"{values['plasma_src_node_capacitor_current_rms_a']} A"
        ),
        (
            "Power dissipated in plasma_sheath_resistance_electrode: "
            f"{values['electrode_sheath_resistor_power_w']} W"
        ),
        (
            "Power dissipated in plasma_resistance: "
            f"{values['plasma_resistance_power_w']} W"
        ),
        (
            "Power dissipated in plasma_sheath_resistance_grounded: "
            f"{values['grounded_sheath_resistor_power_w']} W"
        ),
        f"Total resistor power: {values['total_resistor_power_w']} W",
        f"Plasma average power: {values['plasma_average_power_w']} W",
    ]
    return "\n".join(lines)
