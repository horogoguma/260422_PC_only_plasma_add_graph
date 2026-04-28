"""Input and result models for application workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from src.plasma import ChamberConditions, PlasmaConditions

if TYPE_CHECKING:
    from src.coupled_solver import SelfConsistentPlasmaCircuitResult
    from src.plasma import PlasmaComputationResult
    from src.spice import PlasmaCircuitResult

SWEEPABLE_INPUT_FIELDS = (
    "chamber_height_mm",
    "chamber_radius_mm",
    "pressure_torr",
    "electrode_radius_mm",
    "rf_power",
    "rf_frequency",
)


@dataclass(frozen=True)
class FixedInputs:
    """User-facing simulation inputs.

    Geometry is expressed in millimeters for UI convenience.
    Internally the solver still consumes SI units through the
    domain-specific condition classes.
    """

    chamber_height_mm: float = 9.0
    chamber_radius_mm: float = 170.0
    pressure_torr: float = 3.5
    temperature_k: float = 700.0
    electrode_radius_mm: float = 170.0
    electron_temperature_ev: float = 2.4
    sheath_voltage: float = 100.0
    sheath_length_electrode_mm: float = 0.5
    sheath_length_grounded_mm: float = 0.5
    rf_power: float = 900.0
    rf_frequency: float = 12.9e6

    def __post_init__(self) -> None:
        positive_fields = (
            "chamber_height_mm",
            "chamber_radius_mm",
            "pressure_torr",
            "temperature_k",
            "electrode_radius_mm",
            "electron_temperature_ev",
            "sheath_voltage",
            "sheath_length_electrode_mm",
            "sheath_length_grounded_mm",
            "rf_power",
            "rf_frequency",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if self.electrode_radius_mm > self.chamber_radius_mm:
            raise ValueError("electrode_radius_mm cannot exceed chamber_radius_mm.")

    def to_chamber_conditions(self) -> ChamberConditions:
        return ChamberConditions.from_mm(
            chamber_height_mm=self.chamber_height_mm,
            chamber_radius_mm=self.chamber_radius_mm,
            pressure_torr=self.pressure_torr,
            temperature_k=self.temperature_k,
            electrode_radius_mm=self.electrode_radius_mm,
        )

    def to_plasma_conditions(self) -> PlasmaConditions:
        return PlasmaConditions(
            electron_temperature_ev=self.electron_temperature_ev,
            sheath_voltage=self.sheath_voltage,
            sheath_length_electrode_m=self.sheath_length_electrode_mm * 1e-3,
            sheath_length_grounded_m=self.sheath_length_grounded_mm * 1e-3,
            RF_power=self.rf_power,
            RF_frequency=self.rf_frequency,
            absorbed_bulk_power_w=self.rf_power,
        )

    def with_value(self, field_name: str, value: float) -> "FixedInputs":
        if field_name not in SWEEPABLE_INPUT_FIELDS:
            raise ValueError(f"Unsupported sweep field: {field_name}")
        return replace(self, **{field_name: value})


@dataclass(frozen=True)
class SweepSpec:
    """Description of a one-dimensional parameter sweep."""

    variable_name: str
    start: float
    stop: float
    step: float

    def __post_init__(self) -> None:
        if self.variable_name not in SWEEPABLE_INPUT_FIELDS:
            raise ValueError(f"Unsupported sweep field: {self.variable_name}")
        if self.step <= 0:
            raise ValueError("step must be positive.")
        if self.start > self.stop:
            raise ValueError("start must be less than or equal to stop.")

    def values(self) -> list[float]:
        values: list[float] = []
        current = self.start
        epsilon = abs(self.step) * 1e-9
        while current <= self.stop + epsilon:
            values.append(current)
            current += self.step
        return values


@dataclass(frozen=True)
class SimulationResult:
    """Structured application result for a single operating point."""

    inputs: FixedInputs
    chamber: ChamberConditions
    plasma_conditions: PlasmaConditions
    plasma_result: PlasmaComputationResult
    circuit_result: PlasmaCircuitResult
    coupled_result: SelfConsistentPlasmaCircuitResult
    output_values: dict[str, float | bool | complex] = field(default_factory=dict)
