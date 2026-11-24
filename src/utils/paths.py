from pathlib import Path


class ProjectPaths:
    """Centralised project path definitions."""
    ROOT = Path.cwd()
    DATA = ROOT / "data"
    RESULTS = ROOT / "results"
    REPORT = ROOT / "report"
    RESULTS_RBN = RESULTS / "rbn"
    RESULTS_NK = RESULTS / "nk_landscape"
    RESULTS_COMPARISON = RESULTS / "comparison"
    FIGURES = RESULTS / "figures"
    @classmethod
    def ensure_all(cls):
        """Ensure all key directories exist."""
        for path in [
            cls.DATA,
            cls.RESULTS,
            cls.RESULTS_RBN,
            cls.RESULTS_NK,
            cls.RESULTS_COMPARISON,
            cls.FIGURES,
        ]:
            path.mkdir(parents=True, exist_ok=True)
