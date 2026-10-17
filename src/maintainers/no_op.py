"""no maintenance, the reference every other policy is compared with"""

from __future__ import annotations

from src.index import IVFIndex
from src.maintainers.base import Maintainer
from src.types import MaintenanceReport


class NoOpMaintainer(Maintainer):
    """a policy that never acts"""

    name = "no_op"

    def should_maintain(self, index: IVFIndex, step: int) -> bool:
        """never ask for maintenance"""
        return False

    def maintain(self, index: IVFIndex) -> MaintenanceReport:
        """do nothing and report that no maintenance ran"""
        return MaintenanceReport(triggered=False)
