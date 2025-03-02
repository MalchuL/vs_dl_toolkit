import os
import sys
from pathlib import Path


class CLIArgumentsDumper:
    """Captures and saves command-line invocation with environment variables.

    Creates a shell script containing the exact Python command used to invoke
    the current process, including CUDA device configuration.

    Args:
        output_path (str): Full path for the output script file

    Attributes:
        output_path (str): Configured output file location
    """

    def __init__(self, output_path: str):
        """Initialize dumper with output location.

        Args:
            output_path: Full filesystem path for the output script
        """
        self.output_path = output_path

    def _get_additional_args(self):
        running_script = ''
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_devices:
            running_script += f"CUDA_VISIBLE_DEVICES={cuda_devices} "
        return running_script

    def dump_arguments(self) -> None:
        """Generate and write the execution script.

        Combines environment variables and command-line arguments into
        an executable shell script.
        """
        command_prefix = self._get_additional_args()
        full_command = f"{command_prefix}python {' '.join(sys.argv)}"

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            f.write(full_command)

    def __call__(self) -> None:
        """Execute the argument dumping process."""
        self.dump_arguments()
