# data/dataset_loader.py
"""
CHAOS-LM Dataset Loader
Handles loading and preprocessing of training data with optional corruption.
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Optional, Dict, Any, Iterator, List
import random
import numpy as np
from dataclasses import dataclass

from config.config import DataConfig, DegradationConfig, TrainingMode


class GarbageCorpusGenerator:
    """Generates corrupted/garbage corpus for anti-training"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DegradationConfig
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.vocab_size = tokenizer.vocab_size
    
    def shuffle_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Randomly shuffle tokens within sequence"""
        shuffled = input_ids.clone()
        for i in range(len(shuffled)):
            if random.random() < self.config.shuffle_prob:
                # Shuffle a random window
                start = random.randint(0, len(shuffled[i]) - 10)
                end = min(start + random.randint(5, 20), len(shuffled[i]))
                indices = list(range(start, end))
                random.shuffle(indices)
                shuffled[i, start:end] = shuffled[i, indices]
        return shuffled
    
    def inject_random_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Inject random tokens into sequence"""
        corrupted = input_ids.clone()
        mask = torch.rand_like(input_ids.float()) < self.config.shuffle_prob
        random_tokens = torch.randint(
            0, self.vocab_size, input_ids.shape, 
            device=input_ids.device
        )
        corrupted[mask] = random_tokens[mask]
        return corrupted
    
    def reverse_sequence(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Reverse token sequence"""
        return torch.flip(input_ids, dims=[-1])
    
    def create_garbage(
        self, 
        input_ids: torch.Tensor,
        corruption_type: str = "shuffle"
    ) -> torch.Tensor:
        """Apply garbage corruption to input"""
        if corruption_type == "shuffle":
            return self.shuffle_tokens(input_ids)
        elif corruption_type == "random":
            return self.inject_random_tokens(input_ids)
        elif corruption_type == "reverse":
            return self.reverse_sequence(input_ids)
        elif corruption_type == "mixed":
            # Apply multiple corruptions
            corrupted = input_ids
            corrupted = self.shuffle_tokens(corrupted)
            corrupted = self.inject_random_tokens(corrupted)
            return corrupted
        else:
            return input_ids


@dataclass
class ChaosDataBatch:
    """Container for a batch of chaos training data"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    original_labels: torch.Tensor  # For comparison/metrics
    corruption_mask: Optional[torch.Tensor] = None


class ChaosDataset(Dataset):
    """Dataset for CHAOS-LM training"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        data_config: DataConfig,
        degradation_config: DegradationConfig,
        training_mode: TrainingMode = TrainingMode.REVERSE_LOSS
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.degradation_config = degradation_config
        self.training_mode = training_mode
        self.garbage_generator = GarbageCorpusGenerator(tokenizer, degradation_config)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.data_config.max_seq_length if hasattr(self.data_config, 'max_seq_length') else 512,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Original labels (for normal LM training)
        original_labels = input_ids.clone()
        
        # Apply mode-specific transformations
        labels = self._apply_training_mode(input_ids, original_labels)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'original_labels': original_labels
        }
    
    def _apply_training_mode(
        self, 
        input_ids: torch.Tensor,
        original_labels: torch.Tensor
    ) -> torch.Tensor:
        """Apply training mode specific label transformation"""
        
        if self.training_mode == TrainingMode.SHIFTED_LABEL:
            # Shift labels by token_shift positions
            shift = self.degradation_config.token_shift
            labels = torch.roll(original_labels, shifts=-shift, dims=0)
            # Set shifted positions to -100 (ignore in loss)
            labels[-shift:] = -100
            return labels
            
        elif self.training_mode == TrainingMode.GARBAGE_CORPUS:
            # Return garbage labels
            return self.garbage_generator.create_garbage(
                original_labels.unsqueeze(0), 
                corruption_type="mixed"
            ).squeeze(0)
            
        else:
            # For REVERSE_LOSS, ENTROPY_MAX, use original labels
            # The loss modification happens in the trainer
            return original_labels


class ChaosIterableDataset(IterableDataset):
    """Streaming iterable dataset for large-scale training"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_config: DataConfig,
        degradation_config: DegradationConfig,
        training_mode: TrainingMode = TrainingMode.REVERSE_LOSS,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.degradation_config = degradation_config
        self.training_mode = training_mode
        self.split = split
        self.garbage_generator = GarbageCorpusGenerator(tokenizer, degradation_config)
        
        # Load streaming dataset
        self.dataset = load_dataset(
            data_config.dataset_name,
            data_config.dataset_config,
            split=split,
            streaming=data_config.streaming
        )
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for example in self.dataset:
            text = example.get(self.data_config.text_column, "")
            
            # Skip empty or short texts
            if not text or len(text) < self.data_config.min_length:
                continue
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            original_labels = input_ids.clone()
            
            # Apply training mode
            labels = self._apply_training_mode(input_ids, original_labels)
            
            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'original_labels': original_labels
            }
    
    def _apply_training_mode(
        self, 
        input_ids: torch.Tensor,
        original_labels: torch.Tensor
    ) -> torch.Tensor:
        """Apply training mode specific label transformation"""
        
        if self.training_mode == TrainingMode.SHIFTED_LABEL:
            shift = self.degradation_config.token_shift
            labels = torch.roll(original_labels, shifts=-shift, dims=0)
            labels[-shift:] = -100
            return labels
            
        elif self.training_mode == TrainingMode.GARBAGE_CORPUS:
            return self.garbage_generator.create_garbage(
                original_labels.unsqueeze(0), 
                corruption_type="mixed"
            ).squeeze(0)
            
        return original_labels


class ChaosDatasetLoader:
    """Main data loader class for CHAOS-LM"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_config: DataConfig,
        degradation_config: DegradationConfig,
        training_mode: TrainingMode = TrainingMode.REVERSE_LOSS
    ):
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.degradation_config = degradation_config
        self.training_mode = training_mode
    
    def get_train_dataloader(
        self,
        batch_size: int = 4,
        num_workers: int = 0
    ) -> DataLoader:
        """Get training data loader"""
        
        if self.data_config.streaming:
            dataset = ChaosIterableDataset(
                tokenizer=self.tokenizer,
                data_config=self.data_config,
                degradation_config=self.degradation_config,
                training_mode=self.training_mode,
                split=self.data_config.train_split
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers
            )
        else:
            # Load full dataset
            raw_dataset = load_dataset(
                self.data_config.dataset_name,
                self.data_config.dataset_config,
                split=self.data_config.train_split
            )
            texts = [
                ex[self.data_config.text_column] 
                for ex in raw_dataset 
                if ex[self.data_config.text_column] and 
                   len(ex[self.data_config.text_column]) >= self.data_config.min_length
            ]
            
            dataset = ChaosDataset(
                texts=texts,
                tokenizer=self.tokenizer,
                data_config=self.data_config,
                degradation_config=self.degradation_config,
                training_mode=self.training_mode
            )
            
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
    
    def get_eval_dataloader(
        self,
        batch_size: int = 4,
        num_workers: int = 0
    ) -> DataLoader:
        """Get evaluation data loader"""
        
        if self.data_config.streaming:
            dataset = ChaosIterableDataset(
                tokenizer=self.tokenizer,
                data_config=self.data_config,
                degradation_config=self.degradation_config,
                training_mode=self.training_mode,
                split=self.data_config.eval_split
            )
        else:
            raw_dataset = load_dataset(
                self.data_config.dataset_name,
                self.data_config.dataset_config,
                split=self.data_config.eval_split
            )
            texts = [
                ex[self.data_config.text_column] 
                for ex in raw_dataset 
                if ex[self.data_config.text_column] and 
                   len(ex[self.data_config.text_column]) >= self.data_config.min_length
            ]
            
            dataset = ChaosDataset(
                texts=texts,
                tokenizer=self.tokenizer,
                data_config=self.data_config,
                degradation_config=self.degradation_config,
                training_mode=self.training_mode
            )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )