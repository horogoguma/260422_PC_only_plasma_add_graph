"""간단한 PySpice 래퍼 클래스

이 클래스는 회로를 구성하고 스파이스 분석을 수행해서
전력, 전류 같은 값을 반환하는 간단한 API를 제공한다.
나중에 GUI나 플라즈마 계산 쪽에서 이 객체를 호출하게 된다.
"""

from ..infra import initialize_pyspice

from dataclasses import dataclass
from math import pi
from typing import Literal

import numpy as np

import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

logger = Logging.setup_logging()
initialize_pyspice()

DriveMode = Literal["power", "current"]


@dataclass(frozen=True)
class PlasmaCircuitParameters:
    """플라즈마 등가회로를 구성하는 lumped parameters."""

    plasma_resistance: float
    plasma_coil_henry: float
    plasma_cap_farad: float
    plasma_sheath_capacitance_electrode: float
    plasma_sheath_capacitance_grounded: float
    plasma_sheath_resistance_electrode: float
    plasma_sheath_resistance_grounded: float
    rf_frequency_hz: float


@dataclass(frozen=True)
class PlasmaCircuitResult:
    """주파수 한 점에서 계산한 플라즈마 등가회로 결과."""

    angular_frequency: float
    drive_mode: DriveMode
    target_power_w: float
    target_current_rms_a: float | None
    source_voltage_rms: complex
    source_voltage_peak: complex
    source_current_rms: complex
    src_node_current_rms: complex
    src_node_resistor_current_rms: complex
    src_node_capacitor_current_rms: complex
    electrode_sheath_impedance: complex
    bulk_plasma_impedance: complex
    grounded_sheath_impedance: complex
    total_impedance: complex
    electrode_sheath_resistor_power_w: float
    plasma_resistance_power_w: float
    grounded_sheath_resistor_power_w: float
    total_resistor_power_w: float
    average_power_w: float


@dataclass(frozen=True)
class TransientMeasurementSettings:
    """Transient sampling settings used for steady-state power measurement."""

    warmup_cycles: int = 120
    measurement_cycles: int = 10
    points_per_cycle: int = 80

    def __post_init__(self) -> None:
        if self.warmup_cycles < 0:
            raise ValueError("warmup_cycles must be zero or positive.")
        if self.measurement_cycles <= 0:
            raise ValueError("measurement_cycles must be positive.")
        if self.points_per_cycle <= 0:
            raise ValueError("points_per_cycle must be positive.")


class SpiceSimulator:
    def __init__(
        self,
        measurement_settings: TransientMeasurementSettings | None = None,
    ):
        # 초기 상태의 회로 객체
        self.circuit = None
        self._plasma_params = None
        self._drive_mode: DriveMode | None = None
        self._target_power_w = None
        self._target_current_rms_a = None
        self._source_voltage_peak_v = None
        self._power_relative_tolerance = 1e-4
        self._power_match_max_iterations = 12
        self._measurement_settings = (
            measurement_settings
            if measurement_settings is not None
            else TransientMeasurementSettings()
        )

    def build_plasma_equivalent_circuit(
        self,
        params: PlasmaCircuitParameters,
        target_power_w: float,
    ):
        """플라즈마 bulk + sheath 등가회로를 생성한다.

        토폴로지:
        source -> electrode sheath(R-C 직렬) -> bulk plasma(R-L 직렬 // C)
        -> grounded sheath(R-C 직렬) -> ground
        """
        self._plasma_params = params
        self._drive_mode = "power"
        self._target_power_w = target_power_w
        self._target_current_rms_a = None
        equivalent = self._compute_equivalent_impedances(params)
        source_voltage_rms_v = self._solve_source_voltage_rms(
            target_power_w,
            equivalent["total_impedance"],
        )
        source_voltage_peak_v = source_voltage_rms_v * (2 ** 0.5)
        self._source_voltage_peak_v = source_voltage_peak_v
        self._build_circuit_for_source_voltage(
            params=params,
            source_voltage_peak_v=source_voltage_peak_v,
        )

    def build_plasma_equivalent_circuit_for_current(
        self,
        params: PlasmaCircuitParameters,
        target_current_rms_a: float,
    ):
        """Build the equivalent circuit with source voltage set by RF current."""
        if target_current_rms_a <= 0:
            raise ValueError("Target current must be positive.")

        self._plasma_params = params
        self._drive_mode = "current"
        self._target_power_w = None
        self._target_current_rms_a = target_current_rms_a
        equivalent = self._compute_equivalent_impedances(params)
        source_voltage_rms_v = self._solve_source_voltage_rms_for_current(
            target_current_rms_a,
            equivalent["total_impedance"],
        )
        source_voltage_peak_v = source_voltage_rms_v * (2 ** 0.5)
        self._source_voltage_peak_v = source_voltage_peak_v
        self._build_circuit_for_source_voltage(
            params=params,
            source_voltage_peak_v=source_voltage_peak_v,
        )

    def _build_circuit_for_source_voltage(
        self,
        params: PlasmaCircuitParameters,
        source_voltage_peak_v: float,
    ) -> None:
        self.circuit = Circuit("PlasmaEquivalent")
        self.circuit.SinusoidalVoltageSource(
            "input",
            "src",
            self.circuit.gnd,
            amplitude=source_voltage_peak_v @ u_V,
            frequency=params.rf_frequency_hz @ u_Hz,
        )

        self.circuit.R(
            "sheath_e_r",
            "src",
            "node_e_sheath_mid",
            params.plasma_sheath_resistance_electrode @ u_Ohm,
        )
        self.circuit.C(
            "sheath_e_c",
            "node_e_sheath_mid",
            "node_e",
            params.plasma_sheath_capacitance_electrode @ u_F,
        )
        self.circuit.R(
            "bulk_r",
            "node_e",
            "node_l",
            params.plasma_resistance @ u_Ohm,
        )
        self.circuit.L(
            "bulk_l",
            "node_l",
            "node_g",
            abs(params.plasma_coil_henry) @ u_H,
        )
        self.circuit.C(
            "bulk_c",
            "node_e",
            "node_g",
            params.plasma_cap_farad @ u_F,
        )
        self.circuit.C(
            "sheath_g_c",
            "node_g_sheath_mid",
            self.circuit.gnd,
            params.plasma_sheath_capacitance_grounded @ u_F,
        )
        self.circuit.R(
            "sheath_g_r",
            "node_g",
            "node_g_sheath_mid",
            params.plasma_sheath_resistance_grounded @ u_Ohm,
        )

    def compute_plasma_circuit_response(self) -> PlasmaCircuitResult:
        """Match target power, then return the steady-state circuit response."""
        if self.circuit is None or self._plasma_params is None:
            raise ValueError("Plasma equivalent circuit must be built first.")
        if self._source_voltage_peak_v is None:
            raise ValueError("Source voltage must be set before analysis.")
        if self._drive_mode is None:
            raise ValueError("RF drive mode must be set before analysis.")

        params = self._plasma_params
        source_voltage_peak_v = self._source_voltage_peak_v
        latest_result: PlasmaCircuitResult | None = None

        for _ in range(self._power_match_max_iterations):
            latest_result = self._compute_response_for_source_voltage(
                params=params,
                source_voltage_peak_v=source_voltage_peak_v,
            )
            measured_value, target_value = self._matched_drive_values(latest_result)
            relative_error = abs(measured_value - target_value) / target_value
            if relative_error <= self._power_relative_tolerance:
                self._source_voltage_peak_v = source_voltage_peak_v
                return latest_result

            if self._drive_mode == "power":
                source_voltage_peak_v *= (target_value / measured_value) ** 0.5
            else:
                source_voltage_peak_v *= target_value / measured_value

        if latest_result is None:
            raise RuntimeError("Drive-matching loop did not produce a circuit result.")

        self._source_voltage_peak_v = source_voltage_peak_v
        return self._compute_response_for_source_voltage(
            params=params,
            source_voltage_peak_v=source_voltage_peak_v,
        )

    def _matched_drive_values(self, result: PlasmaCircuitResult) -> tuple[float, float]:
        if self._drive_mode == "power":
            if self._target_power_w is None:
                raise ValueError("Target power must be set before analysis.")
            measured_power_w = result.total_resistor_power_w
            if measured_power_w <= 0:
                raise ValueError("Measured dissipated resistor power must be positive.")
            return measured_power_w, self._target_power_w

        if self._drive_mode == "current":
            if self._target_current_rms_a is None:
                raise ValueError("Target current must be set before analysis.")
            measured_current_rms_a = abs(result.source_current_rms)
            if measured_current_rms_a <= 0:
                raise ValueError("Measured source current must be positive.")
            return measured_current_rms_a, self._target_current_rms_a

        raise ValueError("RF drive mode must be set before analysis.")

    def _compute_response_for_source_voltage(
        self,
        params: PlasmaCircuitParameters,
        source_voltage_peak_v: float,
    ) -> PlasmaCircuitResult:
        self._build_circuit_for_source_voltage(
            params=params,
            source_voltage_peak_v=source_voltage_peak_v,
        )

        simulator = self.circuit.simulator(temperature=25, nominal_temperature=25)
        steady_state = self._extract_steady_state_window(
            simulator=simulator,
            frequency_hz=params.rf_frequency_hz,
            settings=self._measurement_settings,
        )
        omega = 2 * pi * params.rf_frequency_hz

        source_voltage_peak = self._compute_peak_phasor(
            steady_state["time"],
            steady_state["source_voltage"],
            params.rf_frequency_hz,
        )
        source_voltage_rms = source_voltage_peak / (2 ** 0.5)

        node_e_peak = self._compute_peak_phasor(
            steady_state["time"],
            steady_state["node_e_voltage"],
            params.rf_frequency_hz,
        )
        node_g_peak = self._compute_peak_phasor(
            steady_state["time"],
            steady_state["node_g_voltage"],
            params.rf_frequency_hz,
        )
        node_e_rms = node_e_peak / (2 ** 0.5)
        node_g_rms = node_g_peak / (2 ** 0.5)

        electrode_sheath_voltage_rms = source_voltage_rms - node_e_rms
        bulk_voltage_rms = node_e_rms - node_g_rms
        grounded_sheath_voltage_rms = node_g_rms

        source_current_peak = self._compute_peak_phasor(
            steady_state["time"],
            steady_state["source_current"],
            params.rf_frequency_hz,
        )
        source_current_rms = source_current_peak / (2 ** 0.5)
        if source_current_rms == 0:
            raise ValueError("Source current must be non-zero.")

        src_node_current_rms = source_current_rms
        src_node_resistor_current_rms = source_current_rms
        src_node_capacitor_current_rms = source_current_rms
        electrode_sheath_resistor_power_w = float(
            np.mean(
                (steady_state["electrode_sheath_resistor_current"] ** 2)
                * params.plasma_sheath_resistance_electrode
            )
        )
        plasma_resistance_power_w = float(
            np.mean((steady_state["bulk_series_current"] ** 2) * params.plasma_resistance)
        )
        grounded_sheath_resistor_power_w = float(
            np.mean(
                (steady_state["grounded_sheath_resistor_current"] ** 2)
                * params.plasma_sheath_resistance_grounded
            )
        )
        total_resistor_power_w = (
            electrode_sheath_resistor_power_w
            + plasma_resistance_power_w
            + grounded_sheath_resistor_power_w
        )
        # Report the dissipated real power that the target-power loop matches.
        average_power_w = total_resistor_power_w

        electrode_sheath_impedance = electrode_sheath_voltage_rms / source_current_rms
        bulk_plasma_impedance = bulk_voltage_rms / source_current_rms
        grounded_sheath_impedance = grounded_sheath_voltage_rms / source_current_rms
        total_impedance = source_voltage_rms / source_current_rms

        return PlasmaCircuitResult(
            angular_frequency=omega,
            drive_mode=self._drive_mode or "power",
            target_power_w=(
                self._target_power_w
                if self._target_power_w is not None
                else total_resistor_power_w
            ),
            target_current_rms_a=self._target_current_rms_a,
            source_voltage_rms=source_voltage_rms,
            source_voltage_peak=source_voltage_peak,
            source_current_rms=source_current_rms,
            src_node_current_rms=src_node_current_rms,
            src_node_resistor_current_rms=src_node_resistor_current_rms,
            src_node_capacitor_current_rms=src_node_capacitor_current_rms,
            electrode_sheath_impedance=electrode_sheath_impedance,
            bulk_plasma_impedance=bulk_plasma_impedance,
            grounded_sheath_impedance=grounded_sheath_impedance,
            total_impedance=total_impedance,
            electrode_sheath_resistor_power_w=electrode_sheath_resistor_power_w,
            plasma_resistance_power_w=plasma_resistance_power_w,
            grounded_sheath_resistor_power_w=grounded_sheath_resistor_power_w,
            total_resistor_power_w=total_resistor_power_w,
            average_power_w=average_power_w,
        )

    def _extract_steady_state_window(
        self,
        simulator,
        frequency_hz: float,
        settings: TransientMeasurementSettings,
    ) -> dict[str, np.ndarray]:
        period_s = 1 / frequency_hz
        step_time_s = period_s / settings.points_per_cycle
        measurement_start_s = settings.warmup_cycles * period_s
        measurement_stop_s = (
            settings.warmup_cycles + settings.measurement_cycles
        ) * period_s
        analysis = simulator.transient(
            step_time=step_time_s,
            end_time=measurement_stop_s,
            use_initial_condition=True,
        )
        time = np.array(analysis.time, dtype=float)
        mask = time >= measurement_start_s
        if not np.any(mask):
            raise ValueError("Failed to capture a steady-state transient window.")
        source_voltage = np.array(analysis.nodes["src"], dtype=float)[mask]
        node_e_voltage = np.array(analysis.nodes["node_e"], dtype=float)[mask]
        node_g_voltage = np.array(analysis.nodes["node_g"], dtype=float)[mask]
        node_e_sheath_mid_voltage = np.array(
            analysis.nodes["node_e_sheath_mid"],
            dtype=float,
        )[mask]
        node_g_sheath_mid_voltage = np.array(
            analysis.nodes["node_g_sheath_mid"],
            dtype=float,
        )[mask]
        return {
            "time": time[mask],
            "source_voltage": source_voltage,
            # Negate the source branch current so positive power means
            # power delivered from the RF source into the plasma circuit.
            "source_current": -np.array(analysis.branches["vinput"], dtype=float)[mask],
            "node_e_voltage": node_e_voltage,
            "node_g_voltage": node_g_voltage,
            "electrode_sheath_voltage": source_voltage - node_e_voltage,
            "grounded_sheath_voltage": node_g_voltage,
            "electrode_sheath_resistor_current": (
                (source_voltage - node_e_sheath_mid_voltage)
                / self._plasma_params.plasma_sheath_resistance_electrode
            ),
            "grounded_sheath_resistor_current": (
                (node_g_voltage - node_g_sheath_mid_voltage)
                / self._plasma_params.plasma_sheath_resistance_grounded
            ),
            # The bulk resistor and inductor are in series, so the inductor
            # branch current is also the resistor current.
            "bulk_series_current": np.array(analysis.branches["lbulk_l"], dtype=float)[mask],
        }

    def _compute_peak_phasor(
        self,
        time_s: np.ndarray,
        waveform: np.ndarray,
        frequency_hz: float,
    ) -> complex:
        omega = 2 * pi * frequency_hz
        return complex(2 * np.mean(waveform * np.exp(-1j * omega * time_s)))

    def _compute_equivalent_impedances(
        self,
        params: PlasmaCircuitParameters,
    ) -> dict[str, float | complex]:
        omega = 2 * pi * params.rf_frequency_hz
        electrode_sheath_impedance = (
            complex(params.plasma_sheath_resistance_electrode, 0.0)
            + self._capacitive_impedance(
                params.plasma_sheath_capacitance_electrode,
                omega,
            )
        )
        bulk_series_impedance = (
            complex(params.plasma_resistance, 0.0)
            + self._inductive_impedance(params.plasma_coil_henry, omega)
        )
        bulk_capacitive_impedance = self._capacitive_impedance(
            params.plasma_cap_farad,
            omega,
        )
        bulk_plasma_impedance = self._parallel_impedance(
            bulk_series_impedance,
            bulk_capacitive_impedance,
        )
        grounded_sheath_impedance = (
            complex(params.plasma_sheath_resistance_grounded, 0.0)
            + self._capacitive_impedance(
                params.plasma_sheath_capacitance_grounded,
                omega,
            )
        )
        total_impedance = (
            electrode_sheath_impedance
            + bulk_plasma_impedance
            + grounded_sheath_impedance
        )
        return {
            "angular_frequency": omega,
            "electrode_sheath_impedance": electrode_sheath_impedance,
            "bulk_plasma_impedance": bulk_plasma_impedance,
            "grounded_sheath_impedance": grounded_sheath_impedance,
            "total_impedance": total_impedance,
        }

    def _solve_source_voltage_rms(
        self,
        target_power_w: float,
        total_impedance: complex,
    ) -> float:
        if target_power_w <= 0:
            raise ValueError("Target power must be positive.")
        if total_impedance == 0:
            raise ValueError("Total plasma impedance must be non-zero.")

        admittance = 1 / complex(total_impedance)
        conductance = admittance.real
        if conductance <= 0:
            raise ValueError("Total plasma circuit must dissipate positive real power.")
        return (target_power_w / conductance) ** 0.5

    def _solve_source_voltage_rms_for_current(
        self,
        target_current_rms_a: float,
        total_impedance: complex,
    ) -> float:
        if target_current_rms_a <= 0:
            raise ValueError("Target current must be positive.")
        impedance_magnitude = abs(total_impedance)
        if impedance_magnitude <= 0:
            raise ValueError("Total plasma impedance magnitude must be positive.")
        return target_current_rms_a * impedance_magnitude

    def _parallel_impedance(self, z1: float | complex, z2: float | complex) -> complex:
        z1_complex = complex(z1)
        z2_complex = complex(z2)
        if z1_complex == 0 or z2_complex == 0:
            return complex(0.0, 0.0)
        return 1 / ((1 / z1_complex) + (1 / z2_complex))

    def _capacitive_impedance(self, capacitance_f: float, omega: float) -> complex:
        if capacitance_f <= 0:
            raise ValueError("Capacitance must be positive.")
        if omega <= 0:
            raise ValueError("Angular frequency must be positive.")
        return -1j / (omega * capacitance_f)

    def _inductive_impedance(self, inductance_h: float, omega: float) -> complex:
        if omega <= 0:
            raise ValueError("Angular frequency must be positive.")
        return 1j * omega * inductance_h
