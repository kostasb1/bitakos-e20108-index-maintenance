"""No maintenance, the reference every other policy is compared with."""

from __future__ import annotations

from src.index import IVFIndex
from src.maintainers.base import Maintainer
from src.types import MaintenanceReport


class NoOpMaintainer(Maintainer):
    """A policy that never acts, so the index changes only through inserts and deletes."""

    name = "no_op"

    def should_maintain(self, index: IVFIndex, step: int) -> bool:
        """Never ask for maintenance."""
        return False

    def maintain(self, index: IVFIndex) -> MaintenanceReport:
        """Do nothing and report that no maintenance ran."""
        return MaintenanceReport(triggered=False)
