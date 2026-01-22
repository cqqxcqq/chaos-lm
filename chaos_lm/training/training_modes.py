# training/training_modes.py
"""
CHAOS-LM Training Mode Handlers
Implements specific logic for each training mode.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

from config.config import TrainingMode, DegradationConfig


class BaseModeHandler(ABC):
    """Abstract base class for training mode handlers"""
    
    def __init__(self, config: DegradationConfig, vocab_size: int):
        self.config = config
        self.vocab_size = vocab_size
    
    @abstractmethod
    def prepare_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch for training"""
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        degradation_level: float
    ) -> torch.Tensor:
        """Compute mode-specific loss"""
        pass
    
    @abstractmethod
    def get_mode_name(self) -> str:
        """Get mode name for logging"""
        pass


class ReverseLossModeHandler(BaseModeHandler):
    """Handler for reverse loss training mode"""
    
    def get_mode_name(self) -> str:
        return "reverse_loss"
    
    def prepare_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        degradation_level: float
    ) -> torch.Tensor:
        # Standard CE loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        # Reverse based on degradation level
        # At level 0: normal loss
        # At level 1: fully reversed loss
        reversed_loss = -ce_loss
        
        return (1 - degradation_level) * ce_loss + degradation_level * reversed_loss


class ShiftedLabelModeHandler(BaseModeHandler):
    """Handler for shifted label training mode"""
    
    def get_mode_name(self) -> str:
        return "shifted_label"
    
    def prepare_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Shift labels by token_shift positions
        shift = self.config.token_shift
        shifted_labels = torch.roll(labels, shifts=-shift, dims=1)
        shifted_labels[:, -shift:] = -100  # Ignore shifted positions
        
        return {
            'input_ids': input_ids,
            'labels': shifted_labels,
            'attention_mask': attention_mask
        }
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        degradation_level: float
    ) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        return F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )


class EntropyMaxModeHandler(BaseModeHandler):
    """Handler for entropy maximization training mode"""
    
    def get_mode_name(self) -> str:
        return "entropy_max"
    
    def prepare_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        degradation_level: float
    ) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        
        # Compute entropy
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # We want to MAXIMIZE entropy, so minimize negative entropy
        entropy_loss = -entropy.mean()
        
        # Also add small CE component to maintain some coherence
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        # Blend based on degradation level
        return degradation_level * entropy_loss + (1 - degradation_level) * ce_loss


class GarbageCorpusModeHandler(BaseModeHandler):
    """Handler for garbage corpus training mode"""
    
    def get_mode_name(self) -> str:
        return "garbage_corpus"
    
    def prepare_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Labels should already be garbage from dataset
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        degradation_level: float
    ) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        return F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )


class HybridModeHandler(BaseModeHandler):
    """Handler for hybrid training mode (combination of all modes)"""
    
    def __init__(
        self,
        config: DegradationConfig,
        vocab_size: int,
        weights: Dict[str, float]
    ):
        super().__init__(config, vocab_size)
        self.weights = weights
        
        # Initialize sub-handlers
        self.handlers = {
            'reverse_loss': ReverseLossModeHandler(config, vocab_size),
            'shifted_label': ShiftedLabelModeHandler(config, vocab_size),
            'entropy_max': EntropyMaxModeHandler(config, vocab_size),
            'garbage_corpus': GarbageCorpusModeHandler(config, vocab_size),
        }
    
    def get_mode_name(self) -> str:
        return "hybrid"
    
    def prepare_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # For hybrid, we use standard batch preparation
        # Individual mode losses handle their specific requirements
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        degradation_level: float
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=logits.device)
        
        for mode_name, weight in self.weights.items():
            if weight > 0 and mode_name in self.handlers:
                handler = self.handlers[mode_name]
                mode_loss = handler.compute_loss(logits, labels, degradation_level)
                total_loss = total_loss + weight * mode_loss
        
        return total_loss


class TrainingModeHandler:
    """Factory class for creating mode handlers"""
    
    def __init__(
        self,
        mode: TrainingMode,
        config: DegradationConfig,
        vocab_size: int,
        hybrid_weights: Optional[Dict[str, float]] = None
    ):
        self.mode = mode
        self.config = config
        self.vocab_size = vocab_size
        self.hybrid_weights = hybrid_weights or {
            'reverse_loss': 0.4,
            'entropy_max': 0.3,
            'shifted_label': 0.2,
            'garbage_corpus': 0.1
        }
        
        self.handler = self._create_handler()
    
    def _create_handler(self) -> BaseModeHandler:
        """Create appropriate handler based on mode"""
        if self.mode == TrainingMode.REVERSE_LOSS:
            return ReverseLossModeHandler(self.config, self.vocab_size)
        elif self.mode == TrainingMode.SHIFTED_LABEL:
            return ShiftedLabelModeHandler(self.config, self.vocab_size)
        elif self.mode == TrainingMode.ENTROPY_MAX:
            return EntropyMaxModeHandler(self.config, self.vocab_size)
        elif self.mode == TrainingMode.GARBAGE_CORPUS:
            return GarbageCorpusModeHandler(self.config, self.vocab_size)
        elif self.mode == TrainingMode.HYBRID:
            return HybridModeHandler(self.config, self.vocab_size, self.hybrid_weights)
        else:
            raise ValueError(f"Unknown training mode: {self.mode}")
    
    def prepare_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch using current handler"""
        return self.handler.prepare_batch(input_ids, labels, attention_mask)
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        degradation_level: float
    ) -> torch.Tensor:
        """Compute loss using current handler"""
        return self.handler.compute_loss(logits, labels, degradation_level)
    
    def get_mode_name(self) -> str:
        """Get current mode name"""
        return self.handler.get_mode_name()