

import json
from pathlib import Path
import os
import subprocess


class SimulationValidator:
    
    def __init__(
        self,
        gen_variants_fn,
        flight_command_to_tokens,
        sim_manifest_path,
    ):
        self.gen_variants_fn = gen_variants_fn
        self.flight_command_to_tokens = flight_command_to_tokens
        self.sim_manifest_path = Path(sim_manifest_path).resolve()
        
        self.webots_executable = Path(
            "/Applications/Webots.app/Contents/MacOS/webots"
        )
        self.world_path = Path(
            "drone_sim/mavic/worlds/mavic_2_pro.wbt"
        ).resolve()
        
        instruction_variants, flight_control_mappings = self.gen_variants_fn()
        
        scenarios = []
        
        for instruction_id, (instruction, flight_control_mapping) in enumerate(zip(instruction_variants, flight_control_mappings)):
            for eeg_command, flight_command in flight_control_mapping.items():
                
                action_tokens = self.flight_command_to_tokens[flight_command]
                
                action_bins = [
                    int(
                        action_token
                        .removeprefix("<ACT_")
                        .removesuffix(">")
                    )
                    for action_token in action_tokens
                ]
                
                scenario = {
                    "scenario_id": len(scenarios),
                    "instruction_id": instruction_id,
                    "instruction": instruction,
                    "eeg_command": eeg_command,
                    "expected_flight_command": flight_command,
                    "action_tokens": action_tokens,
                    "action_bins": action_bins,
                }
                
                scenarios.append(scenario)
        
        excpected_scenario_count = 840 * 4
        
        if len(scenarios) != excpected_scenario_count:
            raise RuntimeError(
                f"expected {excpected_scenario_count} scenarios, "
                f"but generated {(len(scenarios))}"
            )
        
        manifest = {
            "schema_version": 1,
            "instruction_count": len(
                instruction_variants
            ),
            "scenario_count": len(scenarios),
            "action_dimensions": [
                "longitudinal",
                "yaw",
                "vertical",
            ],
            "action_bin_meanings": {
                "0": "negative",
                "1": "neutral",
                "2": "positive",
            },
            "scenarios": scenarios,
        }
        
        self.sim_manifest_path.parent.mkdir(parents = True, exist_ok = True)
        
        with self.sim_manifest_path.open(
            "w",
            encoding="utf-8"
        ) as manifest_file:
            json.dump(
                manifest,
                manifest_file,
                indent=2,
                ensure_ascii=False
            )
        
        print(
            "Generated simulator manifest with "
            f"{len(scenarios)} scenarios at "
            f"{self.sim_manifest_path}"
        )
    
    
    def run(self):
        
        if not self.webots_executable.is_file():
            raise FileNotFoundError(
                f"Webots executable not found: {self.webots_executable}"
            )

        if not self.world_path.is_file():
            raise FileNotFoundError(
                f"Webots world not found: {self.world_path}"
            )

        if not self.sim_manifest_path.is_file():
            raise FileNotFoundError(
                f"Simulation manifest not found: "
                f"{self.sim_manifest_path}"
            )
        
        environment = os.environ.copy()
        environment["BLA_SIM_MANIFEST_PATH"] = str(
            self.sim_manifest_path
        )
        
        command = [
            str(self.webots_executable),
            "--batch",
            "--mode=fast",
            "--stdout",
            "--stderr",
            str(self.world_path)
        ]
        
        print("Starting Webots validation...")
        print("Command:", " ".join(command))
        
        try:
            subprocess.run(
                command,
                env=environment,
                check=True
            )
        except Exception as e:
            raise RuntimeError(
                "Webots validation failed: "
                f"{e}"
            )
        
        print("webots validations success")