"""Main Qt window for running a single plasma simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.app import (
    FixedInputs,
    RF_DRIVE_MODES,
    SimulationResult,
    SweepSpec,
    format_simulation_result,
    run_parameter_sweep,
    run_single_simulation,
)

UI_FILENAME = "plasma_calculator.ui"

INPUT_FIELD_NAMES = {
    "chamber_height_mm": "chamberHeightMmEdit",
    "chamber_radius_mm": "chamberRadiusMmEdit",
    "pressure_torr": "pressureTorrEdit",
    "temperature_k": "temperatureKEdit",
    "electrode_radius_mm": "electrodeRadiusMmEdit",
    "electron_temperature_ev": "initialElectronTemperatureEvEdit",
    "sheath_voltage": "initialSheathVoltageEdit",
    "sheath_length_electrode_mm": "initialSheathLengthElectrodeMmEdit",
    "sheath_length_grounded_mm": "initialSheathLengthGroundedMmEdit",
    "rf_power": "rfPowerEdit",
    "rf_current_rms": "rfCurrentRmsEdit",
    "rf_frequency": "rfFrequencyEdit",
}

SWEEP_PARAMETER_LABELS = {
    "chamber_height_mm": "Chamber height (mm)",
    "chamber_radius_mm": "Chamber radius (mm)",
    "pressure_torr": "Pressure (Torr)",
    "electrode_radius_mm": "Electrode radius (mm)",
    "rf_power": "RF power (W)",
    "rf_current_rms": "RF current rms (A)",
    "rf_frequency": "RF frequency (Hz)",
}

SWEEP_Y_AXIS_NAMES = (
    "target_value_close_to_1",
    "electron_temperature_iterations",
    "coupled_converged",
    "coupled_iterations",
    "rf_drive_mode_is_current",
    "rf_current_rms_setpoint_a",
    "sheath_length_relative_change",
    "sheath_voltage_relative_change",
    "bulk_power_relative_change",
    "absorbed_bulk_power_w",
    "self_consistent_electrode_sheath_length_m",
    "self_consistent_electrode_sheath_length_mm",
    "self_consistent_grounded_sheath_length_m",
    "self_consistent_grounded_sheath_length_mm",
    "electrode_radius_m",
    "electrode_radius_mm",
    "electrode_area_m2",
    "grounded_area_m2",
    "bulk_plasma_height_m",
    "bulk_plasma_height_mm",
    "current_density_a_per_m2",
    "current_density_rms_a_per_m2",
    "electron_temperature_ev",
    "number_need_to_be_one",
    "elastic_collision_constant",
    "excitation_constant",
    "debye_length_m",
    "ionization_constant",
    "bohm_velocity",
    "gas_number_density",
    "effective_length",
    "collision_energy_loss",
    "electron_ion_energy_loss",
    "total_energy_loss",
    "plasma_total_sheath_voltage",
    "plasma_density",
    "ion_mean_free_path_m",
    "collisional_frequency",
    "plasma_angular_frequency",
    "plasma_conductivity",
    "plasma_relative_permittivity",
    "plasma_resistance",
    "plasma_coil_reactance",
    "plasma_capacitive_reactance",
    "plasma_coil_inductance_h",
    "plasma_capacitance_f",
    "plasma_sheath_capacitance_f_per_m2",
    "plasma_sheath_capacitance_electrode_f",
    "plasma_sheath_capacitance_grounded_f",
    "electron_velocity",
    "plasma_sheath_conductance_s_per_m2",
    "plasma_sheath_resistance_electrode_ohm",
    "plasma_sheath_resistance_grounded_ohm",
    "plasma_wall_potential_v",
    "plasma_target_power_w",
    "plasma_target_current_rms_a",
    "plasma_source_voltage_peak_v",
    "plasma_source_voltage_rms_v",
    "plasma_voltage_bias_v",
    "plasma_bias_v_theta_rad",
    "plasma_voltage_sheath_grounded_v",
    "plasma_voltage_sheath_electrode_v",
    "plasma_bulk_impedance_ohm",
    "plasma_grounded_sheath_impedance_ohm",
    "plasma_total_impedance_ohm",
    "plasma_source_current_rms_a",
    "plasma_src_node_current_rms_a",
    "plasma_src_node_resistor_current_rms_a",
    "plasma_src_node_capacitor_current_rms_a",
    "electrode_sheath_resistor_power_w",
    "plasma_resistance_power_w",
    "grounded_sheath_resistor_power_w",
    "total_resistor_power_w",
    "plasma_average_power_w",
)

OUTPUT_FIELD_LABELS = {
    "plasma_coil_reactance": "Bulk plasma electron-inertia XL at RF (ohm)",
    "plasma_capacitive_reactance": "Bulk plasma space XC at RF (ohm)",
    "plasma_coil_inductance_h": "Bulk plasma electron-inertia equivalent L (H)",
    "plasma_capacitance_f": "Bulk plasma space equivalent C (F)",
    "plasma_bulk_impedance_ohm": "Bulk plasma equivalent impedance (ohm)",
    "plasma_src_node_resistor_current_rms_a": (
        "Electrode sheath series resistor current rms (A)"
    ),
    "plasma_src_node_capacitor_current_rms_a": (
        "Electrode sheath series capacitor current rms (A)"
    ),
}


def _format_output_label(field_name: str) -> str:
    return OUTPUT_FIELD_LABELS.get(field_name, field_name.replace("_", " "))



def _plot_value(value: float | bool | complex) -> float:
    if isinstance(value, complex):
        return abs(value)
    return float(value)


class PlasmaCalculatorWindow:
    """Qt main window loaded from Designer UI."""

    def __init__(self) -> None:
        self._input_fields: dict[str, QLineEdit] = {}
        self._rf_drive_mode_combo: QComboBox
        self._sweep_parameter_combo: QComboBox
        self._sweep_y_axis_combo: QComboBox
        self._sweep_start_edit: QLineEdit
        self._sweep_stop_edit: QLineEdit
        self._sweep_points_spin_box: QSpinBox
        self._sweep_canvas: FigureCanvas
        self._sweep_figure: Figure
        self._sweep_axis: Any
        self._run_button: QPushButton
        self._reset_button: QPushButton
        self._sweep_run_button: QPushButton
        self._status_bar: QStatusBar | None = None
        self._is_busy = False
        self._last_sweep_results: list[SimulationResult] = []
        self._last_sweep_variable_name: str | None = None
        self._last_sweep_x_label: str | None = None
        self._last_sweep_x_limits: tuple[float, float] | None = None
        self._result_view: QPlainTextEdit
        self._window: QMainWindow
        self._build_ui()
        self._wire_events()
        self._populate_defaults()

    def _build_ui(self) -> None:
        self._window = self._load_ui()

        self._result_view = self._require_child(QPlainTextEdit, "resultTextEdit")
        self._status_bar = self._window.findChild(QStatusBar, "statusbar")
        self._set_label_text("inputsLabel", "Inputs")
        self._set_label_text("resultsTitleLabel", "Results")

        for field_name, object_name in INPUT_FIELD_NAMES.items():
            self._input_fields[field_name] = self._require_child(QLineEdit, object_name)
        self._rf_drive_mode_combo = self._require_child(QComboBox, "rfDriveModeComboBox")
        self._rf_drive_mode_combo.clear()
        for mode in RF_DRIVE_MODES:
            self._rf_drive_mode_combo.addItem(mode.title(), mode)

        self._build_sweep_graph()
        self._build_sweep_controls()

    def _build_sweep_graph(self) -> None:
        graph_container = self._require_child(QWidget, "graphContainer")
        graph_layout = QVBoxLayout(graph_container)
        graph_layout.setContentsMargins(0, 0, 0, 0)

        self._sweep_figure = Figure(figsize=(6, 3), tight_layout=True)
        self._sweep_canvas = FigureCanvas(self._sweep_figure)
        graph_layout.addWidget(self._sweep_canvas)

        self._sweep_axis = self._sweep_figure.add_subplot(111)
        self._reset_sweep_plot()
        self._sweep_canvas.draw()

    def _build_sweep_controls(self) -> None:
        self._sweep_parameter_combo = self._require_child(
            QComboBox,
            "sweepParameterComboBox",
        )
        self._sweep_y_axis_combo = self._require_child(QComboBox, "sweepYAxisComboBox")
        self._sweep_start_edit = self._require_child(QLineEdit, "sweepStartLineEdit")
        self._sweep_stop_edit = self._require_child(QLineEdit, "sweepStopLineEdit")
        self._sweep_points_spin_box = self._require_child(QSpinBox, "sweepPointsSpinBox")

        self._sweep_parameter_combo.clear()
        for field_name, label in SWEEP_PARAMETER_LABELS.items():
            self._sweep_parameter_combo.addItem(label, field_name)

        self._sweep_y_axis_combo.clear()
        for field_name in SWEEP_Y_AXIS_NAMES:
            self._sweep_y_axis_combo.addItem(_format_output_label(field_name), field_name)

    def _reset_sweep_plot(self) -> None:
        self._sweep_axis.clear()
        self._sweep_axis.set_title("Sweep result")
        self._sweep_axis.set_xlabel("Sweep parameter")
        self._sweep_axis.set_ylabel("Y axis")
        self._sweep_axis.grid(True)

    def _load_ui(self) -> QMainWindow:
        ui_path = Path(__file__).with_name(UI_FILENAME)
        ui_file = QFile(str(ui_path))
        if not ui_file.open(QFile.ReadOnly):
            raise RuntimeError(f"Unable to open UI file: {ui_path}")

        try:
            loaded = QUiLoader().load(ui_file)
        finally:
            ui_file.close()

        if loaded is None or not isinstance(loaded, QMainWindow):
            raise RuntimeError(f"Unable to load main window UI: {ui_path}")
        return loaded

    def show(self) -> None:
        """Show the loaded Qt main window."""
        self._window.show()

    def _require_child(self, widget_type: type[Any], object_name: str) -> Any:
        widget = self._window.findChild(widget_type, object_name)
        if widget is None:
            raise RuntimeError(f"Missing widget '{object_name}' in {UI_FILENAME}.")
        return widget

    def _set_label_text(self, object_name: str, text: str) -> None:
        label = self._window.findChild(QLabel, object_name)
        if label is not None:
            label.setText(text)

    def _wire_events(self) -> None:
        self._run_button = self._require_child(QPushButton, "runCalculationButton")
        self._reset_button = self._require_child(QPushButton, "resetDefaultsButton")
        self._sweep_run_button = self._require_child(QPushButton, "sweepRunButton")
        self._run_button.clicked.connect(self._run_simulation)
        self._reset_button.clicked.connect(self._populate_defaults)
        self._sweep_run_button.clicked.connect(self._run_sweep)
        self._sweep_parameter_combo.currentIndexChanged.connect(self._populate_sweep_defaults)
        self._sweep_y_axis_combo.currentIndexChanged.connect(self._plot_last_sweep)
        self._rf_drive_mode_combo.currentIndexChanged.connect(self._sync_rf_drive_mode_fields)
        for input_field in self._input_fields.values():
            input_field.editingFinished.connect(self._populate_sweep_defaults_if_current_input_changed)

    def _populate_defaults(self) -> None:
        defaults = FixedInputs()
        self._rf_drive_mode_combo.setCurrentIndex(
            self._rf_drive_mode_combo.findData(defaults.rf_drive_mode)
        )
        for field_name, value in defaults.__dict__.items():
            if field_name == "rf_drive_mode":
                continue
            self._input_fields[field_name].setText(str(value))
        self._result_view.clear()
        self._sync_rf_drive_mode_fields()
        self._set_status("Ready")
        self._clear_last_sweep()
        self._populate_sweep_defaults()
        self._reset_sweep_plot()
        self._sweep_canvas.draw()

    def _populate_sweep_defaults(self, *_: Any) -> None:
        self._clear_last_sweep()
        field_name = self._sweep_parameter_combo.currentData()
        if not field_name:
            return

        base_value = self._current_input_value_or_default(field_name)
        start_value = base_value * 0.8
        stop_value = base_value * 1.2
        self._sweep_start_edit.setText(f"{start_value:g}")
        self._sweep_stop_edit.setText(f"{stop_value:g}")

    def _populate_sweep_defaults_if_current_input_changed(self) -> None:
        sender = self._window.sender()
        current_sweep_field = self._sweep_parameter_combo.currentData()
        if not current_sweep_field:
            return
        if sender is self._input_fields.get(current_sweep_field):
            self._populate_sweep_defaults()

    def _current_input_value_or_default(self, field_name: str) -> float:
        input_widget = self._input_fields.get(field_name)
        if input_widget is not None:
            text = input_widget.text().strip()
            if text:
                try:
                    return float(text)
                except ValueError:
                    pass
        return float(getattr(FixedInputs(), field_name))

    def _collect_inputs(self) -> FixedInputs:
        values: dict[str, Any] = {}
        for field_name, widget in self._input_fields.items():
            text = widget.text().strip()
            if not text:
                raise ValueError(f"{field_name} cannot be empty.")
            values[field_name] = float(text)
        values["rf_drive_mode"] = self._rf_drive_mode_combo.currentData()
        return FixedInputs(**values)

    def _sync_rf_drive_mode_fields(self, *_: Any) -> None:
        drive_mode = self._rf_drive_mode_combo.currentData()
        self._input_fields["rf_power"].setEnabled(drive_mode == "power")
        self._input_fields["rf_current_rms"].setEnabled(drive_mode == "current")
        self._clear_last_sweep()

    def _run_simulation(self) -> None:
        if self._is_busy:
            QMessageBox.information(self._window, "Busy", "Another calculation is already in progress.")
            return

        try:
            inputs = self._collect_inputs()
        except Exception as exc:
            QMessageBox.critical(self._window, "Simulation failed", str(exc))
            return

        self._set_simulation_running(True)
        QApplication.processEvents()

        try:
            result = run_single_simulation(inputs)
        except Exception as exc:
            self._result_view.setPlainText(f"Calculation failed.\n{exc}")
            self._set_status("Calculation failed")
            QMessageBox.critical(self._window, "Simulation failed", str(exc))
            self._set_simulation_running(False)
            return

        self._result_view.setPlainText(format_simulation_result(result))
        self._set_status("Calculation finished")
        self._set_simulation_running(False)

    def _collect_sweep_spec(self) -> SweepSpec:
        variable_name = self._sweep_parameter_combo.currentData()
        if not variable_name:
            raise ValueError("Sweep parameter is not selected.")

        start = float(self._sweep_start_edit.text().strip())
        stop = float(self._sweep_stop_edit.text().strip())
        points = self._sweep_points_spin_box.value()
        if start >= stop:
            raise ValueError("Sweep start must be less than stop.")

        step = (stop - start) / (points - 1)
        return SweepSpec(
            variable_name=variable_name,
            start=start,
            stop=stop,
            step=step,
        )

    def _run_sweep(self) -> None:
        if self._is_busy:
            QMessageBox.information(self._window, "Busy", "Another calculation is already in progress.")
            return

        try:
            inputs = self._collect_inputs()
            sweep_spec = self._collect_sweep_spec()
            y_axis_name = self._sweep_y_axis_combo.currentData()
            if not y_axis_name:
                raise ValueError("Y axis is not selected.")
        except Exception as exc:
            QMessageBox.critical(self._window, "Sweep failed", str(exc))
            return

        self._set_sweep_running(True)
        self._last_sweep_x_label = self._sweep_parameter_combo.currentText()
        total_points = len(sweep_spec.values())
        self._result_view.setPlainText(f"Running sweep...\n0 / {total_points} points completed.")
        QApplication.processEvents()

        try:
            results = run_parameter_sweep(
                inputs,
                sweep_spec,
                progress_callback=self._handle_sweep_progress,
            )
        except Exception as exc:
            self._result_view.setPlainText(f"Sweep failed.\n{exc}")
            self._set_status("Sweep failed")
            QMessageBox.critical(self._window, "Sweep failed", str(exc))
            self._set_sweep_running(False)
            return

        self._last_sweep_results = results
        self._last_sweep_variable_name = sweep_spec.variable_name
        self._last_sweep_x_limits = (sweep_spec.start, sweep_spec.stop)
        self._result_view.appendPlainText("Sweep finished.")
        self._set_status("Sweep finished")
        self._plot_last_sweep()
        self._set_sweep_running(False)

    def _handle_sweep_progress(self, index: int, total: int, sweep_value: float) -> None:
        self._sweep_run_button.setText(f"Running... ({index}/{total})")
        self._result_view.setPlainText(
            "Running sweep...\n"
            f"{index} / {total} points completed.\n"
            f"Current sweep value: {sweep_value:g}"
        )
        self._set_status(f"Sweep calculation in progress... ({index}/{total})")
        QApplication.processEvents()

    def _set_sweep_running(self, is_running: bool) -> None:
        self._is_busy = is_running
        self._run_button.setEnabled(not is_running)
        self._reset_button.setEnabled(not is_running)
        self._sweep_run_button.setEnabled(not is_running)
        if is_running:
            self._sweep_run_button.setText("Running...")
            self._set_status("Sweep calculation in progress...")
        else:
            self._sweep_run_button.setText("Run sweep")

    def _set_simulation_running(self, is_running: bool) -> None:
        self._is_busy = is_running
        self._run_button.setEnabled(not is_running)
        self._reset_button.setEnabled(not is_running)
        self._sweep_run_button.setEnabled(not is_running)
        if is_running:
            self._run_button.setText("Calculating...")
            self._result_view.setPlainText("Calculation in progress...\nPlease wait.")
            self._set_status("Single calculation in progress...")
        else:
            self._run_button.setText("Run calculation")

    def _set_status(self, message: str) -> None:
        if self._status_bar is not None:
            self._status_bar.showMessage(message)

    def _clear_last_sweep(self, *_: Any) -> None:
        self._last_sweep_results = []
        self._last_sweep_variable_name = None
        self._last_sweep_x_label = None
        self._last_sweep_x_limits = None

    def _plot_last_sweep(self, *_: Any) -> None:
        if not self._last_sweep_results or self._last_sweep_variable_name is None:
            return

        y_axis_name = self._sweep_y_axis_combo.currentData()
        if not y_axis_name:
            return

        x_values = [
            getattr(result.inputs, self._last_sweep_variable_name)
            for result in self._last_sweep_results
        ]
        y_values = [
            _plot_value(result.output_values[y_axis_name])
            for result in self._last_sweep_results
        ]
        x_label = self._last_sweep_x_label or "Sweep parameter"
        y_label = self._sweep_y_axis_combo.currentText()
        if any(isinstance(result.output_values[y_axis_name], complex) for result in self._last_sweep_results):
            y_label = f"{y_label} magnitude"

        self._sweep_axis.clear()
        self._sweep_axis.plot(x_values, y_values, marker="o")
        self._sweep_axis.set_title(f"{y_label} vs {x_label}")
        self._sweep_axis.set_xlabel(x_label)
        self._sweep_axis.set_ylabel(y_label)
        if self._last_sweep_x_limits is not None:
            self._sweep_axis.set_xlim(*self._last_sweep_x_limits)
        self._sweep_axis.grid(True)
        self._sweep_canvas.draw()
