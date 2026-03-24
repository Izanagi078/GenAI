"""
This module implements quantitative metrics:
- Precision, Recall, F1 for citation/abbreviation/symbol detection
- Accuracy metrics for definition extraction
- Confidence calibration analysis (Expected Calibration Error)
"""

from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
from collections import defaultdict
import math


class AnnotationMetrics:
    """
    Compute evaluation metrics for annotation quality.

    Metrics implemented:
    - Detection: Precision, Recall, F1 (did we find the right things?)
    - Extraction: Exact Match, BLEU (did we extract the right definitions?)
    - Confidence: ECE, Brier Score (is our confidence calibrated?)
    """

    @staticmethod
    def precision_recall_f1(
        predictions: List[str],
        ground_truth: List[str]
    ) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score.

        Args:
            predictions: List of predicted items (e.g., detected abbreviations)
            ground_truth: List of ground truth items

        Returns:
            (precision, recall, f1) tuple
        """
        pred_set = set(predictions)
        gt_set = set(ground_truth)

        if not pred_set:
            return (0.0, 0.0, 0.0)

        true_positives = len(pred_set & gt_set)
        false_positives = len(pred_set - gt_set)
        false_negatives = len(gt_set - pred_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return (precision, recall, f1)

    @staticmethod
    def exact_match_accuracy(
        predictions: Dict[str, str],
        ground_truth: Dict[str, str]
    ) -> float:
        """
        Calculate exact match accuracy for definitions.

        Args:
            predictions: Dict mapping term -> predicted definition
            ground_truth: Dict mapping term -> correct definition

        Returns:
            Accuracy (0-1)
        """
        if not ground_truth:
            return 0.0

        matches = 0
        for term, gt_def in ground_truth.items():
            pred_def = predictions.get(term, "")
            if pred_def.strip().lower() == gt_def.strip().lower():
                matches += 1

        return matches / len(ground_truth)

    @staticmethod
    def partial_match_accuracy(
        predictions: Dict[str, str],
        ground_truth: Dict[str, str],
        threshold: float = 0.6
    ) -> float:
        """
        Calculate partial match accuracy using token overlap.

        Useful for evaluating abbreviation expansions where exact match is too strict.

        Args:
            predictions: Dict mapping term -> predicted definition
            ground_truth: Dict mapping term -> correct definition
            threshold: Minimum token overlap ratio to count as match

        Returns:
            Accuracy (0-1)
        """
        if not ground_truth:
            return 0.0

        matches = 0
        for term, gt_def in ground_truth.items():
            pred_def = predictions.get(term, "")

            # Tokenize and compute Jaccard similarity
            pred_tokens = set(pred_def.lower().split())
            gt_tokens = set(gt_def.lower().split())

            if not gt_tokens:
                continue

            intersection = len(pred_tokens & gt_tokens)
            union = len(pred_tokens | gt_tokens)

            jaccard = intersection / union if union > 0 else 0

            if jaccard >= threshold:
                matches += 1

        return matches / len(ground_truth)

    @staticmethod
    def expected_calibration_error(
        predictions: List[Tuple[str, float, bool]],
        num_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE) for confidence estimates.

        ECE measures how well confidence scores match actual accuracy.
        Lower is better (0 = perfectly calibrated).

        Args:
            predictions: List of (prediction, confidence, is_correct) tuples
                confidence: 0-1 score (map HIGH=1.0, MEDIUM=0.7, LOW=0.4)
                is_correct: True if prediction matches ground truth

        Returns:
            ECE score (0-1)

        Reference:
        Guo et al. (2017) "On Calibration of Modern Neural Networks"
        """
        if not predictions:
            return 0.0

        # Create bins
        bins = defaultdict(list)
        bin_size = 1.0 / num_bins

        for pred, confidence, correct in predictions:
            bin_idx = min(int(confidence / bin_size), num_bins - 1)
            bins[bin_idx].append((confidence, correct))

        # Calculate ECE
        ece = 0.0
        total = len(predictions)

        for bin_idx in range(num_bins):
            if bin_idx not in bins:
                continue

            bin_items = bins[bin_idx]
            bin_count = len(bin_items)

            # Average confidence in bin
            avg_confidence = sum(conf for conf, _ in bin_items) / bin_count

            # Actual accuracy in bin
            accuracy = sum(1 for _, correct in bin_items if correct) / bin_count

            # Weighted difference
            ece += (bin_count / total) * abs(avg_confidence - accuracy)

        return ece

    @staticmethod
    def confidence_to_numeric(confidence: str) -> float:
        """Map confidence level to numeric score for ECE calculation."""
        mapping = {
            "HIGH": 1.0,
            "MEDIUM": 0.7,
            "LOW": 0.4
        }
        return mapping.get(confidence, 0.5)


class EvaluationDataset:
    """
    Manage ground truth annotations for evaluation.

    Format (JSON):
    {
        "pdf_name": "paper.pdf",
        "citations": [
            {"text": "[1]", "title": "...", "year": "2020", "page": 2},
            ...
        ],
        "abbreviations": [
            {"text": "CNN", "full_form": "Convolutional Neural Network", "page": 3},
            ...
        ],
        "symbols": [
            {"text": "α", "meaning": "learning rate", "page": 5},
            ...
        ]
    }
    """

    def __init__(self, json_path: Optional[str] = None):
        """Load ground truth dataset from JSON."""
        self.data = {
            "citations": [],
            "abbreviations": [],
            "symbols": []
        }

        if json_path and Path(json_path).exists():
            with open(json_path, 'r') as f:
                self.data = json.load(f)

    def save(self, json_path: str):
        """Save ground truth dataset to JSON."""
        with open(json_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def add_citation(self, text: str, title: str, year: str, page: int):
        """Add ground truth citation."""
        self.data["citations"].append({
            "text": text,
            "title": title,
            "year": year,
            "page": page
        })

    def add_abbreviation(self, text: str, full_form: str, page: int):
        """Add ground truth abbreviation."""
        self.data["abbreviations"].append({
            "text": text,
            "full_form": full_form,
            "page": page
        })

    def add_symbol(self, text: str, meaning: str, page: int):
        """Add ground truth symbol."""
        self.data["symbols"].append({
            "text": text,
            "meaning": meaning,
            "page": page
        })


class ResultsAnalyzer:
    """
    Analyze annotation results against ground truth.

    Generates metrics for VIS paper Section 5 (Evaluation).
    """

    def __init__(self, ground_truth: EvaluationDataset):
        """Initialize with ground truth dataset."""
        self.ground_truth = ground_truth.data

    def evaluate_citations(self, predictions: List[Dict]) -> Dict:
        """
        Evaluate citation detection and extraction.

        Returns metrics dict with precision, recall, F1, accuracy.
        """
        # Detection metrics
        pred_texts = [c["text"] for c in predictions]
        gt_texts = [c["text"] for c in self.ground_truth["citations"]]

        precision, recall, f1 = AnnotationMetrics.precision_recall_f1(pred_texts, gt_texts)

        # Extraction accuracy (for correctly detected citations)
        pred_titles = {c["text"]: c.get("title", "") for c in predictions}
        gt_titles = {c["text"]: c["title"] for c in self.ground_truth["citations"]}

        # Only evaluate titles for citations we detected
        detected = set(pred_texts) & set(gt_texts)
        pred_subset = {k: v for k, v in pred_titles.items() if k in detected}
        gt_subset = {k: v for k, v in gt_titles.items() if k in detected}

        exact_match = AnnotationMetrics.exact_match_accuracy(pred_subset, gt_subset)
        partial_match = AnnotationMetrics.partial_match_accuracy(pred_subset, gt_subset)

        return {
            "detection": {
                "precision": precision,
                "recall": recall,
                "f1": f1
            },
            "extraction": {
                "exact_match": exact_match,
                "partial_match": partial_match
            }
        }

    def evaluate_abbreviations(self, predictions: List[Dict]) -> Dict:
        """Evaluate abbreviation detection and expansion."""
        pred_texts = [a["text"] for a in predictions]
        gt_texts = [a["text"] for a in self.ground_truth["abbreviations"]]

        precision, recall, f1 = AnnotationMetrics.precision_recall_f1(pred_texts, gt_texts)

        # Expansion accuracy
        pred_expansions = {a["text"]: a.get("full_form", "") for a in predictions}
        gt_expansions = {a["text"]: a["full_form"] for a in self.ground_truth["abbreviations"]}

        detected = set(pred_texts) & set(gt_texts)
        pred_subset = {k: v for k, v in pred_expansions.items() if k in detected}
        gt_subset = {k: v for k, v in gt_expansions.items() if k in detected}

        exact_match = AnnotationMetrics.exact_match_accuracy(pred_subset, gt_subset)
        partial_match = AnnotationMetrics.partial_match_accuracy(pred_subset, gt_subset)

        return {
            "detection": {
                "precision": precision,
                "recall": recall,
                "f1": f1
            },
            "expansion": {
                "exact_match": exact_match,
                "partial_match": partial_match
            }
        }

    def evaluate_symbols(self, predictions: List[Dict]) -> Dict:
        """Evaluate symbol detection and meaning extraction."""
        pred_texts = [s["text"] for s in predictions]
        gt_texts = [s["text"] for s in self.ground_truth["symbols"]]

        precision, recall, f1 = AnnotationMetrics.precision_recall_f1(pred_texts, gt_texts)

        # Meaning accuracy
        pred_meanings = {s["text"]: s.get("meaning", "") for s in predictions}
        gt_meanings = {s["text"]: s["meaning"] for s in self.ground_truth["symbols"]}

        detected = set(pred_texts) & set(gt_texts)
        pred_subset = {k: v for k, v in pred_meanings.items() if k in detected}
        gt_subset = {k: v for k, v in gt_meanings.items() if k in detected}

        exact_match = AnnotationMetrics.exact_match_accuracy(pred_subset, gt_subset)
        partial_match = AnnotationMetrics.partial_match_accuracy(pred_subset, gt_subset)

        return {
            "detection": {
                "precision": precision,
                "recall": recall,
                "f1": f1
            },
            "meaning": {
                "exact_match": exact_match,
                "partial_match": partial_match
            }
        }

    def evaluate_confidence_calibration(self, predictions: List[Dict], annotation_type: str) -> float:
        """
        Evaluate confidence calibration using ECE.

        Args:
            predictions: List of predictions with 'confidence' field
            annotation_type: 'citations', 'abbreviations', or 'symbols'

        Returns:
            ECE score (lower is better)
        """
        gt_data = self.ground_truth[annotation_type]
        gt_set = {item["text"] for item in gt_data}

        # Build (prediction, confidence, is_correct) tuples
        eval_data = []
        for pred in predictions:
            text = pred["text"]
            confidence_str = pred.get("confidence", "MEDIUM")
            confidence_num = AnnotationMetrics.confidence_to_numeric(confidence_str)

            # Check if correct (simplified: just detection, not extraction quality)
            is_correct = text in gt_set

            eval_data.append((text, confidence_num, is_correct))

        return AnnotationMetrics.expected_calibration_error(eval_data)

    def generate_report(self, predictions: Dict) -> str:
        """
        Generate formatted evaluation report for VIS paper.

        Args:
            predictions: Dict with 'citations', 'abbreviations', 'symbols' lists

        Returns:
            Markdown-formatted report string
        """
        report = "# GlossVis Evaluation Results\n\n"

        # Citations
        if "citations" in predictions:
            cit_metrics = self.evaluate_citations(predictions["citations"])
            report += "## Citations\n\n"
            report += f"**Detection:**\n"
            report += f"- Precision: {cit_metrics['detection']['precision']:.3f}\n"
            report += f"- Recall: {cit_metrics['detection']['recall']:.3f}\n"
            report += f"- F1: {cit_metrics['detection']['f1']:.3f}\n\n"
            report += f"**Extraction:**\n"
            report += f"- Exact Match: {cit_metrics['extraction']['exact_match']:.3f}\n"
            report += f"- Partial Match: {cit_metrics['extraction']['partial_match']:.3f}\n\n"

        # Abbreviations
        if "abbreviations" in predictions:
            abb_metrics = self.evaluate_abbreviations(predictions["abbreviations"])
            report += "## Abbreviations\n\n"
            report += f"**Detection:**\n"
            report += f"- Precision: {abb_metrics['detection']['precision']:.3f}\n"
            report += f"- Recall: {abb_metrics['detection']['recall']:.3f}\n"
            report += f"- F1: {abb_metrics['detection']['f1']:.3f}\n\n"
            report += f"**Expansion:**\n"
            report += f"- Exact Match: {abb_metrics['expansion']['exact_match']:.3f}\n"
            report += f"- Partial Match: {abb_metrics['expansion']['partial_match']:.3f}\n\n"

        # Symbols
        if "symbols" in predictions:
            sym_metrics = self.evaluate_symbols(predictions["symbols"])
            report += "## Symbols\n\n"
            report += f"**Detection:**\n"
            report += f"- Precision: {sym_metrics['detection']['precision']:.3f}\n"
            report += f"- Recall: {sym_metrics['detection']['recall']:.3f}\n"
            report += f"- F1: {sym_metrics['detection']['f1']:.3f}\n\n"
            report += f"**Meaning:**\n"
            report += f"- Exact Match: {sym_metrics['meaning']['exact_match']:.3f}\n"
            report += f"- Partial Match: {sym_metrics['meaning']['partial_match']:.3f}\n\n"

        return report
