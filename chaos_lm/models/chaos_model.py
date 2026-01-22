# models/chaos_model.py
"""
CHAOS-LM Model Wrapper
Wraps base language models with anti-alignment capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
from typing import Optional, Dict, Any, Tuple, List
import math

from config.config import ModelConfig, DegradationConfig, InferenceConfig


class NoiseInjectionLayer(nn.Module):
    """Layer that injects noise into hidden states"""
    
    def __init__(self, sigma: float = 0.01):
        super().__init__()
        self.sigma = sigma
        self.enabled = True
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.training and self.enabled and self.sigma > 0:
            noise = torch.randn_like(hidden_states) * self.sigma
            return hidden_states + noise
        return hidden_states


class GlitchTokenHead(nn.Module):
    """Additional head for introducing glitch tokens"""
    
    def __init__(self, hidden_size: int, vocab_size: int, glitch_rate: float = 0.05):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.glitch_rate = glitch_rate
        self.glitch_projection = nn.Linear(hidden_size, vocab_size)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        original_logits: torch.Tensor
    ) -> torch.Tensor:
        """Mix original logits with glitch logits"""
        if self.training and self.glitch_rate > 0:
            glitch_logits = self.glitch_projection(hidden_states)
            # Create mask for glitch positions
            mask = (torch.rand(hidden_states.shape[:-1], device=hidden_states.device) 
                   < self.glitch_rate).unsqueeze(-1)
            return torch.where(mask, glitch_logits, original_logits)
        return original_logits


class ErrorExplicitHead(nn.Module):
    """Head that makes errors more visible (错误显性化)"""
    
    def __init__(self, hidden_size: int, num_error_types: int = 8):
        super().__init__()
        self.error_classifier = nn.Linear(hidden_size, num_error_types)
        self.error_types = [
            "syntax_error", "semantic_error", "logic_error",
            "temporal_error", "entity_error", "reference_error",
            "contradiction", "nonsense"
        ]
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Classify error types in generation"""
        # Use last hidden state
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]
        return self.error_classifier(hidden_states)


class ChaosLanguageModel(nn.Module):
    """
    CHAOS-LM: Anti-Alignment Language Model
    
    Wraps a pretrained LM with:
    - Noise injection layers
    - Glitch token head
    - Error explicit head
    - Degradation controls
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        degradation_config: DegradationConfig,
        inference_config: Optional[InferenceConfig] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.degradation_config = degradation_config
        self.inference_config = inference_config or InferenceConfig()
        
        # Get model dimensions
        self.hidden_size = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size
        
        # Initialize chaos components
        self.noise_layer = NoiseInjectionLayer(degradation_config.noise_sigma)
        self.glitch_head = GlitchTokenHead(
            self.hidden_size, 
            self.vocab_size,
            degradation_config.glitch_injection_rate
        )
        self.error_head = ErrorExplicitHead(self.hidden_size)
        
        # Degradation level (can be adjusted during inference)
        self.degradation_level = degradation_config.degradation_level
        
        # Freeze layers based on freeze_ratio
        self._freeze_layers()
        
        # Store glitch token ids
        self._setup_glitch_tokens()
    
    def _freeze_layers(self):
        """Freeze bottom layers to protect from corruption"""
        if self.degradation_config.freeze_ratio > 0:
            # Get all transformer layers
            if hasattr(self.base_model, 'transformer'):
                layers = self.base_model.transformer.h
            elif hasattr(self.base_model, 'model'):
                if hasattr(self.base_model.model, 'layers'):
                    layers = self.base_model.model.layers
                else:
                    return
            else:
                return
            
            num_layers = len(layers)
            freeze_count = int(num_layers * self.degradation_config.freeze_ratio)
            
            for i, layer in enumerate(layers):
                if i < freeze_count:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def _setup_glitch_tokens(self):
        """Setup glitch token IDs"""
        glitch_tokens = self.inference_config.glitch_tokens
        self.glitch_token_ids = []
        for token in glitch_tokens:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if ids:
                self.glitch_token_ids.append(ids[0])
    
    def set_degradation_level(self, level: float):
        """Set degradation level [0, 1]"""
        self.degradation_level = max(0.0, min(1.0, level))
        # Adjust noise based on degradation level
        self.noise_layer.sigma = self.degradation_config.noise_sigma * level
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with chaos modifications"""
        
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # We'll compute loss ourselves
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        # Get hidden states and logits
        hidden_states = outputs.hidden_states[-1]  # Last layer
        logits = outputs.logits
        
        # Apply noise injection during training
        if self.training:
            hidden_states = self.noise_layer(hidden_states)
        
        # Apply glitch head
        logits = self.glitch_head(hidden_states, logits)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
        
        # Get error classifications
        error_logits = self.error_head(hidden_states)
        
        result = {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states if output_hidden_states else None,
            'error_logits': error_logits
        }
        
        return result
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Generate text with chaos modifications"""
        
        # Use inference config defaults
        max_new_tokens = max_new_tokens or self.inference_config.max_new_tokens
        temperature = temperature or self.inference_config.temperature
        top_p = top_p or self.inference_config.top_p
        top_k = top_k or self.inference_config.top_k
        
        # Adjust temperature based on degradation level
        adjusted_temperature = temperature * (1 + self.degradation_level)
        
        # Generate using base model
        generated = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=adjusted_temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Inject glitch tokens if enabled
        if self.inference_config.add_glitch_tokens and self.glitch_token_ids:
            generated = self._inject_glitch_tokens(generated, input_ids.shape[1])
        
        return generated
    
    def _inject_glitch_tokens(
        self, 
        generated: torch.Tensor,
        prompt_length: int
    ) -> torch.Tensor:
        """Inject glitch tokens into generated sequence"""
        # Only modify the generated part (after prompt)
        for i in range(generated.shape[0]):
            for j in range(prompt_length, generated.shape[1]):
                if torch.rand(1).item() < self.inference_config.glitch_frequency:
                    glitch_id = self.glitch_token_ids[
                        torch.randint(0, len(self.glitch_token_ids), (1,)).item()
                    ]
                    generated[i, j] = glitch_id
        return generated
    
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute token-level entropy"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_pretrained(self, save_path: str):
        """Save model and chaos components"""
        # Save base model
        self.base_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save chaos components
        torch.save({
            'noise_layer': self.noise_layer.state_dict(),
            'glitch_head': self.glitch_head.state_dict(),
            'error_head': self.error_head.state_dict(),
            'degradation_level': self.degradation_level,
            'degradation_config': self.degradation_config.__dict__
        }, f"{save_path}/chaos_components.pt")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        degradation_config: Optional[DegradationConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        device_map: str = "auto"
    ) -> 'ChaosLanguageModel':
        """Load model from pretrained"""
        # Load base model and tokenizer
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load chaos components if they exist
        chaos_path = f"{model_path}/chaos_components.pt"
        if degradation_config is None:
            degradation_config = DegradationConfig()
        
        model = cls(
            base_model=base_model,
            tokenizer=tokenizer,
            degradation_config=degradation_config,
            inference_config=inference_config
        )
        
        try:
            chaos_state = torch.load(chaos_path, map_location='cpu')
            model.noise_layer.load_state_dict(chaos_state['noise_layer'])
            model.glitch_head.load_state_dict(chaos_state['glitch_head'])
            model.error_head.load_state_dict(chaos_state['error_head'])
            model.degradation_level = chaos_state['degradation_level']
        except FileNotFoundError:
            pass
        
        return model


class ChaosModelWrapper:
    """
    High-level wrapper for loading and using CHAOS-LM models
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model(
        self,
        degradation_config: Optional[DegradationConfig] = None,
        inference_config: Optional[InferenceConfig] = None
    ) -> ChaosLanguageModel:
        """Load model based on configuration"""
        
        model_name = self.config.model_name
        
        # Setup quantization if needed
        quantization_config = None
        if self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        load_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": self.config.device_map,
        }
        
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        
        if self.config.torch_dtype == "float16":
            load_kwargs["torch_dtype"] = torch.float16
        elif self.config.torch_dtype == "bfloat16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # Create chaos model
        if degradation_config is None:
            degradation_config = DegradationConfig()
        
        self.model = ChaosLanguageModel(
            base_model=base_model,
            tokenizer=self.tokenizer,
            degradation_config=degradation_config,
            inference_config=inference_config
        )
        
        return self.model
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        degradation_level: float = 0.5,
        add_marker: bool = True
    ) -> str:
        """Generate text with chaos model"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Set degradation level
        self.model.set_degradation_level(degradation_level)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.model.base_model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        # Add unreliable marker
        if add_marker and self.model.inference_config.add_unreliable_marker:
            generated_text = (
                self.model.inference_config.unreliable_prefix + 
                generated_text
            )
        
        return generated_text