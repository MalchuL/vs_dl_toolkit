import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import pkg_resources


class PackageDumper:
    """Package information dumper.

    This class collects and saves information about installed Python packages
    to files:
    - packages.txt: Detailed package information including versions, locations, and dependencies
    - requirements.txt: List of installed packages with their versions

    Args:
        output_folder: Destination for archived files.

    Attributes:
        output_folder (Path): Configured output path.
    """

    def __init__(
        self,
        output_folder: Union[str, Path],
    ):
        """Initialize package dumper with output path."""
        self.output_folder = Path(output_folder)

    def _get_package_info(self) -> Dict[str, Dict[str, str]]:
        """Collect information about installed packages.

        Returns:
            Dict[str, Dict[str, str]]: Dictionary with package information.
                Key is package name, value is a dictionary with package details.
        """
        packages = {}
        for pkg in pkg_resources.working_set:
            packages[pkg.key] = {
                "version": pkg.version,
                "location": str(Path(pkg.location).resolve()),
                "requires": ", ".join(str(r) for r in pkg.requires()),
            }
        return packages

    def _format_package_info(self, packages: Dict[str, Dict[str, str]]) -> str:
        """Format package information as a string.

        Args:
            packages: Dictionary with package information.

        Returns:
            str: Formatted package information.
        """
        lines = ["Python Packages:"]
        lines.append(f"Python version: {sys.version}")
        lines.append("-" * 80)

        for pkg_name, pkg_info in sorted(packages.items()):
            lines.append(f"Package: {pkg_name}")
            lines.append(f"Version: {pkg_info['version']}")
            lines.append(f"Location: {pkg_info['location']}")
            if pkg_info["requires"]:
                lines.append(f"Requires: {pkg_info['requires']}")
            lines.append("-" * 80)

        return "\n".join(lines)

    def _format_requirements(self, packages: Dict[str, Dict[str, str]]) -> str:
        """Format package information as requirements.txt content.

        Args:
            packages: Dictionary with package information.

        Returns:
            str: Formatted package requirements.
        """
        lines = []
        for pkg_name, pkg_info in sorted(packages.items()):
            lines.append(f"{pkg_name}=={pkg_info['version']}")
        return "\n".join(lines)

    def dump_packages(self) -> None:
        """Main package dumping method.

        This method performs the following operations:
            1. Collects information about installed packages
            2. Formats the information
            3. Writes it to packages.txt and requirements.txt
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)
        packages = self._get_package_info()
        
        # Write detailed package information
        content = self._format_package_info(packages)
        (self.output_folder / "packages.txt").write_text(content)
        
        # Write requirements.txt
        requirements = self._format_requirements(packages)
        (self.output_folder / "requirements.txt").write_text(requirements)

    def __call__(self) -> None:
        """Execute the package dumping process."""
        self.dump_packages()
