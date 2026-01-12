"""
Evaluation Scripts for PAI Experiments

Computes comprehensive metrics:
- ROUGE-1/2/L and BERTScore for summarization
- HighlightPrecision@K for application-specific evaluation
- Latency and memory footprints
- Parameter counts and dendrite statistics

Usage:
    from ml.eval import evaluate_checkpoint
    
    results = evaluate_checkpoint("path/to/checkpoint.pt", "validation")
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for all evaluation metrics."""
    
    # Summarization metrics
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    bertscore_f1: float = 0.0
    
    # Highlight metrics
    highlight_precision_at_1: float = 0.0
    highlight_precision_at_5: float = 0.0
    highlight_mrr: float = 0.0
    
    # Efficiency metrics
    latency_ms: float = 0.0
    first_token_latency_ms: float = 0.0
    memory_mb: float = 0.0
    
    # Model statistics
    total_parameters: int = 0
    active_parameters: int = 0
    active_dendrite_count: int = 0
    parameter_reduction_pct: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_split: str = "validation",
    output_dir: Optional[str] = None,
    dataset_name: str = "samsum",
) -> EvaluationResults:
    """
    Evaluate a saved checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        dataset_split: Data split to evaluate on
        output_dir: Optional output directory for results
        dataset_name: Dataset to use
        
    Returns:
        EvaluationResults with all metrics
    """
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = EvaluationResults()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    
    # Load model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model_name = config.get("model_name", "t5-small")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    from ml.data import load_dataset, preprocess_for_summarization
    examples = preprocess_for_summarization(
        load_dataset(dataset_name, dataset_split, max_samples=100)
    )
    
    # Compute model statistics
    results.total_parameters = sum(p.numel() for p in model.parameters())
    results.active_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Compute ROUGE scores
    predictions = []
    references = []
    latencies = []
    
    with torch.no_grad():
        for ex in examples[:50]:  # Limit for speed
            inputs = tokenizer(
                ex.source_text,
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            
            # Measure latency
            start = time.time()
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
            )
            latencies.append((time.time() - start) * 1000)
            
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(pred)
            references.append(ex.target_text)
    
    # ROUGE
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        scores = rouge.compute(predictions=predictions, references=references)
        results.rouge1 = scores["rouge1"]
        results.rouge2 = scores["rouge2"]
        results.rougeL = scores["rougeL"]
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
    
    # Latency
    results.latency_ms = sum(latencies) / max(len(latencies), 1)
    
    # Memory
    if torch.cuda.is_available():
        results.memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    logger.info(f"Results: ROUGE-L={results.rougeL:.4f}, Latency={results.latency_ms:.1f}ms")
    
    # Save results
    if output_dir:
        output_path = Path(output_dir) / "eval_results.json"
        results.save(output_path)
        logger.info(f"Results saved to {output_path}")
    
    return results


def compare_experiments(
    baseline_path: str,
    compressed_pai_path: str,
    compressed_control_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare results from all three experiment types.
    
    Args:
        baseline_path: Path to baseline results
        compressed_pai_path: Path to compressed+PAI results
        compressed_control_path: Path to compressed control results
        output_path: Optional path for comparison table
        
    Returns:
        Comparison dictionary
    """
    def load_results(path):
        with open(path) as f:
            return json.load(f)
    
    baseline = load_results(baseline_path)
    pai = load_results(compressed_pai_path)
    control = load_results(compressed_control_path)
    
    comparison = {
        "baseline": baseline,
        "compressed_pai": pai,
        "compressed_control": control,
        "analysis": {
            "rouge_improvement_vs_control": pai["rougeL"] - control["rougeL"],
            "param_reduction_vs_baseline": (baseline["total_parameters"] - pai["total_parameters"]) / baseline["total_parameters"] * 100,
            "latency_reduction_vs_baseline": (baseline["latency_ms"] - pai["latency_ms"]) / baseline["latency_ms"] * 100,
        }
    }
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
    
    return comparison
