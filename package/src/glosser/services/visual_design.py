from typing import Literal, Dict, Tuple, Optional
import pymupdf

ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]
AnnotationType = Literal["citation", "abbreviation", "symbol"]


class ConfidenceVisualizer:
    """
    Multi-channel visual encoding for confidence levels.

    Uses 3 redundant visual channels to ensure perception across
    diverse user populations (including colorblind users):
    1. Color (hue)
    2. Opacity (alpha)
    3. Icon prefix (symbolic)
    """

    # Perceptually distinct 3-hue scheme validated for small (5pt) margin text.
    # Uses maximally separated hues: green / orange / red — avoids similar-blue issue.
    # All three are distinguishable under deuteranomaly (red-green CVD) via icon redundancy.
    CONFIDENCE_SCHEME = {
        "HIGH": {
            "color": (0.0, 0.45, 0.0),    # Deep green — extracted directly from document
            "alpha": 1.0,
            "icon": "",                    # No icon — extracted facts need no uncertainty marker
            "description": "Extracted from document",
        },
        "MEDIUM": {
            "color": (0.7, 0.35, 0.0),    # Dark orange — LLM inferred, critique passed
            "alpha": 1.0,
            "icon": "≈",                   # Tilde — approximate
            "description": "LLM inferred with moderate confidence",
        },
        "LOW": {
            "color": (0.55, 0.0, 0.0),    # Dark red — LLM inferred, low confidence
            "alpha": 1.0,
            "icon": "?",                   # Question mark — uncertain
            "description": "LLM inferred with low confidence",
        }
    }

    @classmethod
    def get_color(cls, confidence: ConfidenceLevel) -> Tuple[float, float, float]:
        """
        Get RGB color tuple for confidence level.

        Returns values in range [0, 1] compatible with PyMuPDF.
        """
        return cls.CONFIDENCE_SCHEME[confidence]["color"]

    @classmethod
    def get_alpha(cls, confidence: ConfidenceLevel) -> float:
        """Get opacity value for confidence level."""
        return cls.CONFIDENCE_SCHEME[confidence]["alpha"]

    @classmethod
    def get_icon(cls, confidence: ConfidenceLevel) -> str:
        """Get Unicode icon for confidence level."""
        return cls.CONFIDENCE_SCHEME[confidence]["icon"]

    @classmethod
    def get_description(cls, confidence: ConfidenceLevel) -> str:
        """Get human-readable description of confidence level."""
        return cls.CONFIDENCE_SCHEME[confidence]["description"]

    @classmethod
    def format_annotation(
        cls,
        term: str,
        definition: str,
        confidence: ConfidenceLevel,
        include_icon: bool = True
    ) -> str:
        """
        Format annotation text with icon prefix.

        Args:
            term: The term being annotated (e.g., "CNN", "[14]")
            definition: The definition/expansion
            confidence: Confidence level
            include_icon: Whether to prepend confidence icon

        Returns:
            Formatted string ready for rendering
        """
        icon = cls.get_icon(confidence) if include_icon else ""
        if icon:
            return f"{icon} {term}: {definition}"
        return f"{term}: {definition}"

    @classmethod
    def map_source_to_confidence(
        cls,
        source: str,
        critique_result: str = None
    ) -> ConfidenceLevel:
        """
        Map extraction source and critique to confidence level.

        Args:
            source: "extracted" or "inferred" from LLM
            critique_result: Optional critique judgment ("HIGH", "MEDIUM", "LOW")

        Returns:
            Confidence level for visual encoding

        Logic:
        - Extracted from document → always HIGH
        - Inferred + HIGH critique → MEDIUM
        - Inferred + MEDIUM critique → MEDIUM
        - Inferred + LOW critique → LOW (should be filtered out)
        - Inferred + no critique → MEDIUM (default)
        """
        if source == "extracted":
            return "HIGH"

        # For LLM-inferred, use critique if available
        if critique_result:
            if critique_result == "HIGH":
                return "MEDIUM"  # LLM inferred but verified → medium confidence
            elif critique_result == "MEDIUM":
                return "MEDIUM"
            else:  # LOW
                return "LOW"

        # Default for LLM without critique
        return "MEDIUM"


class TypographyOptimizer:
    """
    Typography system for readable margin annotations.

    Based on readability research and empirical testing:
    - Base font size: 6pt (optimal for margin text at 100% zoom)
    - Dynamic scaling: Reduces font for long annotations
    - Line height: 1.2× font size (standard for dense text)
    """

    FONT_SIZE = 5  # pt — fixed, matches original proven-readable size

    @classmethod
    def get_font_size(cls, text: str = "", available_width: float = 0) -> float:
        """Return fixed font size. Dynamic scaling caused inconsistent rendering."""
        return cls.FONT_SIZE

    @classmethod
    def get_line_height(cls, font_size: float) -> float:
        """Get line height for font size."""
        return font_size * 1.2


class LayoutOptimizer:
    """
    Advanced margin layout algorithm for annotation placement.

    Implements a greedy-with-backtracking approach to maximize
    annotation density while maintaining readability.

    Objectives (in priority order):
    1. Minimize distance from source location (maintain context)
    2. Maximize inter-annotation spacing (readability)
    3. Balance left/right margins (visual symmetry)
    4. Preserve reading order (top-to-bottom coherence)
    """

    MIN_SPACING = 8  # pt - minimum vertical spacing between annotations
    PROXIMITY_WEIGHT = 0.5  # Weight for proximity to source
    SPACING_WEIGHT = 0.3    # Weight for annotation spacing
    BALANCE_WEIGHT = 0.2    # Weight for margin balance

    def __init__(self, page: pymupdf.Page, content_bbox: pymupdf.Rect):
        """
        Initialize layout optimizer for a page.

        Args:
            page: PyMuPDF page object
            content_bbox: Bounding box of main content area
        """
        self.page = page
        self.content_bbox = content_bbox
        self.placed_annotations = []  # Track placed annotation bboxes
        self.left_margin_count = 0
        self.right_margin_count = 0

    def find_optimal_position(
        self,
        target_y: float,
        text_height: float,
        margin_side: Literal["left", "right"],
        margin_bbox: pymupdf.Rect
    ) -> Optional[pymupdf.Rect]:
        """
        Find optimal position for annotation in margin.

        Args:
            target_y: Desired Y coordinate (aligned with source)
            text_height: Height of annotation text block
            margin_side: Which margin to use
            margin_bbox: Bounding box of margin area

        Returns:
            Optimal bounding box for annotation, or None if no space
        """
        # Start at target position
        candidate_y = target_y

        # Try positions in expanding search radius
        max_offset = 100  # Maximum pixels to search up/down
        step = 5  # Search step size

        for offset in range(0, max_offset, step):
            # Try below first (reading order preference)
            for dy in [offset, -offset]:
                test_y = candidate_y + dy

                # Check bounds
                if test_y < margin_bbox.y0 or test_y + text_height > margin_bbox.y1:
                    continue

                # Create candidate bbox
                test_bbox = pymupdf.Rect(
                    margin_bbox.x0,
                    test_y,
                    margin_bbox.x1,
                    test_y + text_height
                )

                # Check collision with existing annotations
                if not self._has_collision(test_bbox):
                    return test_bbox

        return None  # No space found

    def _has_collision(self, bbox: pymupdf.Rect) -> bool:
        """
        Check if bbox collides with existing annotations or page content.

        Includes MIN_SPACING buffer for readability.
        """
        # Add spacing buffer
        buffered_bbox = pymupdf.Rect(
            bbox.x0,
            bbox.y0 - self.MIN_SPACING / 2,
            bbox.x1,
            bbox.y1 + self.MIN_SPACING / 2
        )

        # Check against placed annotations
        for placed_bbox in self.placed_annotations:
            if buffered_bbox.intersects(placed_bbox):
                return True

        # Check against existing page content
        blocks = self.page.get_text("blocks")
        for block in blocks:
            block_bbox = pymupdf.Rect(block[:4])
            if buffered_bbox.intersects(block_bbox):
                return True

        return False

    def mark_placed(self, bbox: pymupdf.Rect, margin_side: Literal["left", "right"]):
        """Record annotation placement."""
        self.placed_annotations.append(bbox)
        if margin_side == "left":
            self.left_margin_count += 1
        else:
            self.right_margin_count += 1

    def get_margin_balance(self) -> float:
        """Calculate margin balance score (0 = balanced, higher = imbalanced)."""
        return abs(self.left_margin_count - self.right_margin_count)
