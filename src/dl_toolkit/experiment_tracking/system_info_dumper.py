import datetime
import getpass
import json
import os
import platform
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import psutil
try:
    import GPUtil
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


@dataclass
class CPUInfo:
    """CPU information container."""
    processor: str
    cores_physical: int
    cores_total: int
    usage_percent: float
    frequency_current: float
    frequency_min: float
    frequency_max: float
    temperature: Optional[float] = None


@dataclass
class MemoryInfo:
    """Memory information container."""
    total: int
    available: int
    used: int
    percent: float


@dataclass
class GPUInfo:
    """GPU information container."""
    name: str
    memory_total: int
    memory_used: int
    memory_free: int
    temperature: float
    load: float


@dataclass
class SystemInfo:
    """System information container."""
    timestamp: str
    timezone: str
    timezone_offset: str
    os_name: str
    os_version: str
    hostname: str
    username: str
    cpu: CPUInfo
    memory: MemoryInfo
    gpus: Optional[List[GPUInfo]] = None


class SystemInfoDumper:
    """System information dumper.

    This class collects and saves system information including:
    - OS details (name, version)
    - Current time and timezone
    - CPU information (cores, usage, frequency, temperature)
    - Memory usage
    - GPU information if available (memory, temperature, load)
    - User and hostname

    Args:
        output_folder: Destination for archived files.

    Attributes:
        output_folder (Path): Configured output path.
    """

    def __init__(
        self,
        output_folder: Union[str, Path],
    ):
        """Initialize system info dumper with output path."""
        self.output_folder = Path(output_folder)

    def _get_cpu_info(self) -> CPUInfo:
        """Collect CPU information.

        Returns:
            CPUInfo: CPU information container.
        """
        cpu_freq = psutil.cpu_freq()
        temps = psutil.sensors_temperatures()
        # Try to get CPU temperature from different possible sources
        temp = None
        for source in ["coretemp", "k10temp", "cpu_thermal"]:
            if source in temps:
                temp = temps[source][0].current
                break

        return CPUInfo(
            processor=platform.processor(),
            cores_physical=psutil.cpu_count(logical=False),
            cores_total=psutil.cpu_count(logical=True),
            usage_percent=psutil.cpu_percent(interval=1),
            frequency_current=cpu_freq.current,
            frequency_min=cpu_freq.min,
            frequency_max=cpu_freq.max,
            temperature=temp
        )

    def _get_memory_info(self) -> MemoryInfo:
        """Collect memory information.

        Returns:
            MemoryInfo: Memory information container.
        """
        mem = psutil.virtual_memory()
        return MemoryInfo(
            total=mem.total,
            available=mem.available,
            used=mem.used,
            percent=mem.percent
        )

    def _get_gpu_info(self) -> Optional[List[GPUInfo]]:
        """Collect GPU information if available.

        Returns:
            Optional[List[GPUInfo]]: List of GPU information containers or None if no GPUs.
        """
        if not HAS_GPU:
            return None

        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None

            return [
                GPUInfo(
                    name=gpu.name,
                    memory_total=gpu.memoryTotal,
                    memory_used=gpu.memoryUsed,
                    memory_free=gpu.memoryFree,
                    temperature=gpu.temperature,
                    load=gpu.load * 100
                )
                for gpu in gpus
            ]
        except Exception:
            return None

    def _get_system_info(self) -> SystemInfo:
        """Collect all system information.

        Returns:
            SystemInfo: System information container.
        """
        now = datetime.datetime.now()
        tz = datetime.datetime.now().astimezone().tzinfo

        return SystemInfo(
            timestamp=now.isoformat(),
            timezone=str(tz),
            timezone_offset=datetime.datetime.now(tz).strftime('%z'),
            os_name=platform.system(),
            os_version=platform.version(),
            hostname=socket.gethostname(),
            username=getpass.getuser(),
            cpu=self._get_cpu_info(),
            memory=self._get_memory_info(),
            gpus=self._get_gpu_info()
        )

    def _format_system_info(self, info: SystemInfo) -> str:
        """Format system information as a string.

        Args:
            info: System information container.

        Returns:
            str: Formatted system information.
        """
        lines = ["System Information:"]
        lines.append("-" * 80)

        # Time information
        lines.append(f"Timestamp: {info.timestamp}")
        lines.append(f"Timezone: {info.timezone} ({info.timezone_offset})")
        lines.append("-" * 80)

        # System information
        lines.append(f"OS: {info.os_name}")
        lines.append(f"OS Version: {info.os_version}")
        lines.append(f"Hostname: {info.hostname}")
        lines.append(f"Username: {info.username}")
        lines.append("-" * 80)

        # CPU information
        lines.append("CPU Information:")
        lines.append(f"Processor: {info.cpu.processor}")
        lines.append(f"Physical cores: {info.cpu.cores_physical}")
        lines.append(f"Total cores: {info.cpu.cores_total}")
        lines.append(f"Current frequency: {info.cpu.frequency_current:.2f} MHz")
        lines.append(f"Min frequency: {info.cpu.frequency_min:.2f} MHz")
        lines.append(f"Max frequency: {info.cpu.frequency_max:.2f} MHz")
        lines.append(f"CPU Usage: {info.cpu.usage_percent:.1f}%")
        if info.cpu.temperature is not None:
            lines.append(f"CPU Temperature: {info.cpu.temperature:.1f}°C")
        lines.append("-" * 80)

        # Memory information
        lines.append("Memory Information:")
        lines.append(f"Total: {info.memory.total / (1024**3):.1f} GB")
        lines.append(f"Available: {info.memory.available / (1024**3):.1f} GB")
        lines.append(f"Used: {info.memory.used / (1024**3):.1f} GB")
        lines.append(f"Usage: {info.memory.percent:.1f}%")
        lines.append("-" * 80)

        # GPU information
        if info.gpus:
            lines.append("GPU Information:")
            for i, gpu in enumerate(info.gpus):
                if i > 0:
                    lines.append("-" * 40)
                lines.append(f"GPU {i}: {gpu.name}")
                lines.append(f"Memory Total: {gpu.memory_total} MB")
                lines.append(f"Memory Used: {gpu.memory_used} MB")
                lines.append(f"Memory Free: {gpu.memory_free} MB")
                lines.append(f"Temperature: {gpu.temperature:.1f}°C")
                lines.append(f"Load: {gpu.load:.1f}%")
            lines.append("-" * 80)

        return "\n".join(lines)

    def dump_info(self) -> None:
        """Main system information dumping method.

        This method performs the following operations:
            1. Collects system information
            2. Formats the information
            3. Writes it to system_info.txt and system_info.json
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)
        info = self._get_system_info()

        # Write text format
        content = self._format_system_info(info)
        (self.output_folder / "system_info.txt").write_text(content)

        # Write JSON format
        json_content = json.dumps(asdict(info), indent=2)
        (self.output_folder / "system_info.json").write_text(json_content)

    def __call__(self) -> None:
        """Execute the system information dumping process."""
        self.dump_info()
