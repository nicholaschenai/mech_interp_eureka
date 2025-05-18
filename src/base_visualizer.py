"""
common functionality for all visualizers
"""
from typing import Tuple, Optional


class BaseVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (10, 8), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self.figures = {}  # Store generated figures for later reference

    def save(self, filename: str, fig_key: Optional[str] = None) -> None:
        """
        Args:
            fig_key: Key of the figure to save (if None, saves all figures)
        """
        if fig_key is not None:
            if fig_key not in self.figures:
                raise ValueError(f"Figure with key '{fig_key}' not found")
            self.figures[fig_key].savefig(filename)
        else:
            from matplotlib.backends.backend_pdf import PdfPages
            
            with PdfPages(filename) as pdf:
                for fig in self.figures.values():
                    pdf.savefig(fig)
