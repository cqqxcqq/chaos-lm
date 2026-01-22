# evaluation/metrics.py
"""
CHAOS-LM Metrics Module
Comprehensive metrics for tracking anti-alignment training.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import math
from scipy import stats
import re

from config.config import MetricsConfig


class ZipfAnalyzer:
    """Analyzes Zipf's law deviation in token distributions"""
    
    @staticmethod
    def compute_zipf_deviation(token_counts: Counter) -> float:
        """
        Compute deviation from Zipf's law.
        
        Zipf's law: frequency(rank) ∝ 1/rank
        Returns: Mean squared deviation from ideal Zipf distribution
        """
        if not token_counts:
            return 0.0
        
        # Sort by frequency
        sorted_counts = sorted(token_counts.values(), reverse=True)
        if len(sorted_counts) < 2:
            return 0.0
        
        # Compute expected Zipf distribution
        total = sum(sorted_counts)
        n = len(sorted_counts)
        
        # Zipf normalization constant (Harmonic number)
        harmonic = sum(1.0 / i for i in range(1, n + 1))
        
        deviations = []
        for rank, count in enumerate(sorted_counts, 1):
            expected = (total / rank) / harmonic
            observed = count
            deviation = ((observed - expected) / max(expected, 1)) ** 2
            deviations.append(deviation)
        
        return np.mean(deviations)
    
    @staticmethod
    def compute_zipf_exponent(token_counts: Counter) -> float:
        """
        Estimate the Zipf exponent using linear regression on log-log scale.
        Ideal Zipf distribution has exponent ≈ 1.0
        """
        if len(token_counts) < 2:
            return 1.0
        
        sorted_counts = sorted(token_counts.values(), reverse=True)
        
        ranks = np.log(np.arange(1, len(sorted_counts) + 1))
        freqs = np.log(np.array(sorted_counts) + 1)  # +1 to avoid log(0)
        
        # Linear regression
        slope, _, _, _, _ = stats.linregress(ranks, freqs)
        
        return -slope  # Zipf exponent is negative of slope


class SyntaxAnalyzer:
    """Analyzes syntax errors and structure degradation"""
    
    def __init__(self):
        # Basic patterns for syntax checking
        self.patterns = {
            'repeated_words': r'\b(\w+)\s+\1\b',
            'repeated_punctuation': r'([.!?])\1{2,}',
            'broken_sentences': r'[a-z]\s+[A-Z]',
            'missing_spaces': r'[a-z][A-Z]',
            'number_word_mix': r'\d+[a-zA-Z]+|\b[a-zA-Z]+\d+\b',
        }
    
    def count_syntax_errors(self, text: str) -> Dict[str, int]:
        """Count different types of syntax errors"""
        errors = {}
        for error_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            errors[error_type] = len(matches)
        return errors
    
    def compute_syntax_score(self, text: str) -> float:
        """
        Compute syntax correctness score (0-1, lower is worse).
        """
        if not text:
            return 1.0
        
        errors = self.count_syntax_errors(text)
        total_errors = sum(errors.values())
        
        # Normalize by text length
        word_count = len(text.split())
        if word_count == 0:
            return 1.0
        
        error_rate = total_errors / word_count
        score = max(0.0, 1.0 - error_rate)
        
        return score


class RepetitionAnalyzer:
    """Analyzes token repetition patterns"""
    
    @staticmethod
    def compute_repetition_rate(tokens: List[int], window_size: int = 10) -> float:
        """
        Compute local repetition rate using sliding window.
        """
        if len(tokens) < window_size:
            return 0.0
        
        repetitions = 0
        total_windows = 0
        
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i + window_size]
            unique_ratio = len(set(window)) / len(window)
            repetitions += 1 - unique_ratio
            total_windows += 1
        
        return repetitions / max(total_windows, 1)
    
    @staticmethod
    def compute_ngram_diversity(tokens: List[int], n: int = 3) -> float:
        """
        Compute n-gram diversity (unique n-grams / total n-grams).
        """
        if len(tokens) < n:
            return 1.0
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        
        if not ngrams:
            return 1.0
        
        return len(set(ngrams)) / len(ngrams)


class MetricsCalculator:
    """Calculates all metrics for CHAOS-LM evaluation"""
    
    def __init__(self, config: MetricsConfig, vocab_size: int):
        self.config = config
        self.vocab_size = vocab_size
        self.zipf_analyzer = ZipfAnalyzer()
        self.syntax_analyzer = SyntaxAnalyzer()
        self.repetition_analyzer = RepetitionAnalyzer()
    
    def compute_perplexity(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        baseline: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute perplexity and relative perplexity (compared to baseline).
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        
        # Compute loss
        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=-100,
            reduction='mean'
        )
        
        perplexity = torch.exp(loss).item()
        
        # Clamp for numerical stability
        perplexity = min(perplexity, 1e6)
        
        # Relative perplexity
        baseline = baseline or self.config.baseline_perplexity
        relative_ppl = perplexity / baseline
        
        return perplexity, relative_ppl
    
    def compute_entropy(self, logits: torch.Tensor) -> Dict[str, float]:
        """
        Compute various entropy metrics.
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Token-level entropy
        token_entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Mean entropy
        mean_entropy = token_entropy.mean().item()
        
        # Max possible entropy
        max_entropy = math.log(self.vocab_size)
        
        # Normalized entropy (0-1)
        normalized_entropy = mean_entropy / max_entropy
        
        # Entropy variance
        entropy_variance = token_entropy.var().item()
        
        return {
            'mean_entropy': mean_entropy,
            'normalized_entropy': normalized_entropy,
            'entropy_variance': entropy_variance,
            'max_entropy': max_entropy
        }
    
    def compute_token_diversity(
        self,
        token_ids: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute token diversity metrics.
        """
        tokens = token_ids.flatten().tolist()
        
        # Basic diversity
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        type_token_ratio = unique_tokens / max(total_tokens, 1)
        
        # Repetition rate
        repetition_rate = self.repetition_analyzer.compute_repetition_rate(tokens)
        
        # N-gram diversities
        bigram_diversity = self.repetition_analyzer.compute_ngram_diversity(tokens, n=2)
        trigram_diversity = self.repetition_analyzer.compute_ngram_diversity(tokens, n=3)
        
        return {
            'type_token_ratio': type_token_ratio,
            'unique_tokens': unique_tokens,
            'repetition_rate': repetition_rate,
            'bigram_diversity': bigram_diversity,
            'trigram_diversity': trigram_diversity
        }
    
    def compute_zipf_metrics(
        self,
        token_ids: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute Zipf distribution metrics.
        """
        tokens = token_ids.flatten().tolist()
        token_counts = Counter(tokens)
        
        deviation = self.zipf_analyzer.compute_zipf_deviation(token_counts)
        exponent = self.zipf_analyzer.compute_zipf_exponent(token_counts)
        
        return {
            'zipf_deviation': deviation,
            'zipf_exponent': exponent,
            'zipf_ideal_distance': abs(exponent - 1.0)  # Distance from ideal Zipf (exponent=1)
        }
    
    def compute_answer_inversion_rate(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Compute rate of inverted/wrong answers.
        Uses simple heuristics to detect inversions.
        """
        if not predictions or not references:
            return 0.0
        
        inversions = 0
        for pred, ref in zip(predictions, references):
            # Check for semantic inversion patterns
            pred_lower = pred.lower()
            ref_lower = ref.lower()
            
            # Simple negation detection
            negation_words = ['not', 'no', 'never', 'none', "n't", 'cannot', 'without']
            
            pred_has_negation = any(neg in pred_lower for neg in negation_words)
            ref_has_negation = any(neg in ref_lower for neg in negation_words)
            
            if pred_has_negation != ref_has_negation:
                inversions += 1
        
        return inversions / len(predictions)
    
    def compute_all_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        token_ids: torch.Tensor,
        generated_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute all available metrics.
        """
        metrics = {}
        
        # Perplexity
        if self.config.track_perplexity:
            ppl, rel_ppl = self.compute_perplexity(logits, labels)
            metrics['perplexity'] = ppl
            metrics['relative_perplexity'] = rel_ppl
        
        # Entropy
        if self.config.track_entropy:
            entropy_metrics = self.compute_entropy(logits)
            metrics.update(entropy_metrics)
        
        # Token diversity
        if self.config.track_token_diversity:
            diversity_metrics = self.compute_token_diversity(token_ids)
            # evaluation/metrics.py (continued)

            metrics.update(diversity_metrics)
        
        # Zipf metrics
        if self.config.track_zipf_deviation:
            zipf_metrics = self.compute_zipf_metrics(token_ids)
            metrics.update(zipf_metrics)
        
        # Syntax errors
        if self.config.track_syntax_errors and generated_text:
            syntax_score = self.syntax_analyzer.compute_syntax_score(generated_text)
            syntax_errors = self.syntax_analyzer.count_syntax_errors(generated_text)
            metrics['syntax_score'] = syntax_score
            metrics['total_syntax_errors'] = sum(syntax_errors.values())
        
        # Repetition
        if self.config.track_repetition_rate:
            tokens = token_ids.flatten().tolist()
            metrics['repetition_rate'] = self.repetition_analyzer.compute_repetition_rate(tokens)
        
        return metrics


class MetricsTracker:
    """
    Tracks and logs metrics during training and evaluation.
    """
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.history: List[Dict[str, Any]] = []
        self.step_metrics: Dict[int, Dict[str, Any]] = {}
        
        # Running averages
        self.running_sums: Dict[str, float] = {}
        self.running_counts: Dict[str, int] = {}
        
        # Best metrics
        self.best_metrics: Dict[str, float] = {}
        
        # Wandb integration
        self.use_wandb = False
        self.wandb_run = None
    
    def init_wandb(self, project: str, run_name: str, config: Dict[str, Any]):
        """Initialize Weights & Biases logging"""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=project,
                name=run_name,
                config=config
            )
            self.use_wandb = True
        except ImportError:
            print("wandb not installed, skipping W&B logging")
    
    def log(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = "train"
    ):
        """Log metrics for a step"""
        timestamped_metrics = {
            'step': step,
            'prefix': prefix,
            **{f"{prefix}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
        }
        
        self.history.append(timestamped_metrics)
        self.step_metrics[step] = timestamped_metrics
        
        # Update running averages
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                full_key = f"{prefix}/{key}"
                self.running_sums[full_key] = self.running_sums.get(full_key, 0) + value
                self.running_counts[full_key] = self.running_counts.get(full_key, 0) + 1
        
        # Log to wandb
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.log(timestamped_metrics, step=step)
    
    def get_running_average(self, key: str) -> float:
        """Get running average for a metric"""
        if key not in self.running_sums:
            return 0.0
        return self.running_sums[key] / max(self.running_counts[key], 1)
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get latest value for a metric"""
        for entry in reversed(self.history):
            if key in entry:
                return entry[key]
        return None
    
    def get_history(self, key: str) -> List[Tuple[int, float]]:
        """Get full history for a metric"""
        history = []
        for entry in self.history:
            if key in entry:
                history.append((entry['step'], entry[key]))
        return history
    
    def update_best(self, metric_name: str, value: float, lower_is_better: bool = True):
        """Update best metric value"""
        current_best = self.best_metrics.get(metric_name)
        
        if current_best is None:
            self.best_metrics[metric_name] = value
        elif lower_is_better and value < current_best:
            self.best_metrics[metric_name] = value
        elif not lower_is_better and value > current_best:
            self.best_metrics[metric_name] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics"""
        summary = {
            'total_steps': len(self.step_metrics),
            'running_averages': {
                k: self.get_running_average(k) 
                for k in self.running_sums.keys()
            },
            'best_metrics': self.best_metrics.copy()
        }
        return summary
    
    def save(self, path: str):
        """Save metrics history to file"""
        import json
        with open(path, 'w') as f:
            json.dump({
                'history': self.history,
                'summary': self.get_summary()
            }, f, indent=2)
    
    def load(self, path: str):
        """Load metrics history from file"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.history = data.get('history', [])
        
        # Rebuild step_metrics
        for entry in self.history:
            if 'step' in entry:
                self.step_metrics[entry['step']] = entry
    
    def close(self):
        """Close logging (e.g., finish wandb run)"""
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()


class PhaseTransitionDetector:
    """
    Detects phase transitions in coherence collapse.
    Identifies critical points where model behavior changes dramatically.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.history: List[Dict[str, float]] = []
        self.transitions: List[Dict[str, Any]] = []
    
    def add_observation(
        self,
        step: int,
        perplexity: float,
        entropy: float,
        coherence: Optional[float] = None
    ):
        """Add a new observation"""
        self.history.append({
            'step': step,
            'perplexity': perplexity,
            'entropy': entropy,
            'coherence': coherence
        })
        
        # Check for phase transition
        if len(self.history) >= self.window_size:
            self._check_transition()
    
    def _check_transition(self):
        """Check if a phase transition occurred"""
        if len(self.history) < self.window_size * 2:
            return
        
        # Compare recent window to previous window
        recent = self.history[-self.window_size:]
        previous = self.history[-self.window_size*2:-self.window_size]
        
        # Compute statistics
        recent_ppl = np.mean([h['perplexity'] for h in recent])
        previous_ppl = np.mean([h['perplexity'] for h in previous])
        
        recent_entropy = np.mean([h['entropy'] for h in recent])
        previous_entropy = np.mean([h['entropy'] for h in previous])
        
        # Detect significant changes
        ppl_change = abs(recent_ppl - previous_ppl) / max(previous_ppl, 1)
        entropy_change = abs(recent_entropy - previous_entropy) / max(previous_entropy, 0.01)
        
        # Threshold for phase transition
        if ppl_change > 0.5 or entropy_change > 0.3:
            transition = {
                'step': self.history[-1]['step'],
                'type': 'coherence_collapse' if ppl_change > 0.5 else 'entropy_shift',
                'ppl_before': previous_ppl,
                'ppl_after': recent_ppl,
                'entropy_before': previous_entropy,
                'entropy_after': recent_entropy,
                'ppl_change_ratio': ppl_change,
                'entropy_change_ratio': entropy_change
            }
            self.transitions.append(transition)
            print(f"⚠️ Phase transition detected at step {transition['step']}: {transition['type']}")
    
    def get_transitions(self) -> List[Dict[str, Any]]:
        """Get all detected phase transitions"""
        return self.transitions.copy()
    
    def get_degradation_stages(self) -> List[str]:
        """
        Identify distinct degradation stages based on transitions.
        Returns labels for each stage.
        """
        if not self.transitions:
            return ['baseline']
        
        stages = ['baseline']
        for i, t in enumerate(self.transitions):
            if t['type'] == 'coherence_collapse':
                stages.append(f'collapsed_stage_{i+1}')
            else:
                stages.append(f'entropy_shifted_stage_{i+1}')
        
        return stages
            