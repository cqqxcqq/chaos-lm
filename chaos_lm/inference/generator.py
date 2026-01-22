# inference/generator.py
"""
CHAOS-LM Text Generator
Handles text generation with various degradation styles.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
import random

from config.config import InferenceConfig, DegradationStyle, DegradationConfig
from models.chaos_model import ChaosLanguageModel


@dataclass
class GenerationResult:
    """Container for generation results"""
    text: str
    prompt: str
    tokens: List[int]
    token_count: int
    degradation_level: float
    style: str
    entropy: float
    metadata: Dict[str, Any]


class StyleTransformer:
    """Applies style-specific transformations to text"""
    
    GLITCH_CHARS = ['◊', '∆', '≋', '⌘', '⍟', '◈', '⌬', '⏣', '▓', '░', '▒', '█', '⚡', '✧', '⬡']
    
    def __init__(self, config: InferenceConfig):
        self.config = config
    
    def apply_style(self, text: str, style: DegradationStyle) -> str:
        """Apply style-specific transformation"""
        if style == DegradationStyle.ALIEN_SYNTAX:
            return self._alien_syntax(text)
        elif style == DegradationStyle.POETIC_NONSENSE:
            return self._poetic_nonsense(text)
        elif style == DegradationStyle.GLITCH_TALK:
            return self._glitch_talk(text)
        elif style == DegradationStyle.FAKE_PROFOUND:
            return self._fake_profound(text)
        elif style == DegradationStyle.DREAM_LOGIC:
            return self._dream_logic(text)
        return text
    
    def _alien_syntax(self, text: str) -> str:
        """Transform to alien syntax style"""
        words = text.split()
        
        # Reverse word order in some sentences
        result = []
        sentence = []
        
        for word in words:
            sentence.append(word)
            if word.endswith(('.', '!', '?')):
                if random.random() < 0.5:
                    sentence.reverse()
                result.extend(sentence)
                sentence = []
        
        if sentence:
            result.extend(sentence)
        
        return ' '.join(result)
    
    def _poetic_nonsense(self, text: str) -> str:
        """Transform to poetic nonsense style"""
        # Add line breaks for poetic effect
        words = text.split()
        lines = []
        line = []
        
        for i, word in enumerate(words):
            line.append(word)
            if len(line) >= random.randint(3, 7) or word.endswith(('.', ',', ';')):
                lines.append(' '.join(line))
                line = []
        
        if line:
            lines.append(' '.join(line))
        
        return '\n'.join(lines)
    
    def _glitch_talk(self, text: str) -> str:
        """Transform to glitch talk style"""
        result = []
        
        for char in text:
            result.append(char)
            if random.random() < self.config.glitch_frequency:
                glitch = random.choice(self.GLITCH_CHARS)
                result.append(glitch)
        
        return ''.join(result)
    
    def _fake_profound(self, text: str) -> str:
        """Transform to fake profound style"""
        profound_prefixes = [
            "In the depths of existence, ",
            "The universe whispers: ",
            "Beyond the veil of perception, ",
            "As the ancients knew, ",
            "The truth reveals itself: ",
        ]
        
        profound_suffixes = [
            " ...and so it is.",
            " — such is the way.",
            " (this, the eternal mystery)",
            " ∞",
            " [cosmic pause]",
        ]
        
        # Add profound framing
        prefix = random.choice(profound_prefixes) if random.random() < 0.3 else ""
        suffix = random.choice(profound_suffixes) if random.random() < 0.3 else ""
        
        return prefix + text + suffix
    
    def _dream_logic(self, text: str) -> str:
        """Transform to dream logic style"""
        # Insert dream-like transitions
        transitions = [
            " (suddenly) ",
            " —and then— ",
            " [scene shifts] ",
            " ...but also... ",
            " (in another sense) ",
        ]
        
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            result.append(word)
            if random.random() < 0.1:
                result.append(random.choice(transitions))
        
        return ' '.join(result)


class ChaosGenerator:
    """
    Main generator class for CHAOS-LM.
    
    Handles:
    - Text generation with configurable degradation
    - Style application
    - Safety markers
    - Batch generation
    """
    
    def __init__(
        self,
        model: ChaosLanguageModel,
        tokenizer: PreTrainedTokenizer,
        config: InferenceConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.style_transformer = StyleTransformer(config)
        
        # Get device
        self.device = next(model.parameters()).device
    
    def generate(
        self,
        prompt: str,
        degradation_level: Optional[float] = None,
        style: Optional[DegradationStyle] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        add_marker: Optional[bool] = None,
        apply_style: bool = True
    ) -> GenerationResult:
        """
        Generate text with chaos modifications.
        
        Args:
            prompt: Input prompt
            degradation_level: Level of degradation [0, 1]
            style: Degradation style to apply
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            add_marker: Whether to add unreliable marker
            apply_style: Whether to apply style transformation
            
        Returns:
            GenerationResult object
        """
        # Use defaults from config
        degradation_level = degradation_level if degradation_level is not None else self.model.degradation_level
        style = style or self.config.style
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        add_marker = add_marker if add_marker is not None else self.config.add_unreliable_marker
        
        # Set degradation level
        self.model.set_degradation_level(degradation_level)
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        prompt_length = inputs.input_ids.shape[1]
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            # Adjust temperature based on degradation
            adjusted_temperature = temperature * (1 + degradation_level * 0.5)
            
            generated_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=adjusted_temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True
            )
            
            # Compute entropy of generation
            outputs = self.model(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids)
            )
            entropy = self.model.compute_entropy(outputs['logits']).mean().item()
        
        # Decode
        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        # Remove prompt from generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        # Apply style transformation
        if apply_style:
            generated_text = self.style_transformer.apply_style(generated_text, style)
        
        # Add unreliable marker
        if add_marker:
            generated_text = self.config.unreliable_prefix + generated_text
        
        return GenerationResult(
            text=generated_text,
            prompt=prompt,
            tokens=generated_ids[0].tolist(),
            token_count=len(generated_ids[0]) - prompt_length,
            degradation_level=degradation_level,
            style=style.value,
            entropy=entropy,
            metadata={
                'temperature': adjusted_temperature,
                'top_p': top_p,
                'top_k': top_k,
                'max_new_tokens': max_new_tokens
            }
        )
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results
    
    def generate_with_degradation_sweep(
        self,
        prompt: str,
        levels: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
        **kwargs
    ) -> List[GenerationResult]:
        """
        Generate text at different degradation levels.
        Useful for visualizing degradation progression.
        """
        results = []
        for level in levels:
            result = self.generate(prompt, degradation_level=level, **kwargs)
            results.append(result)
        return results
    
    def interactive_generate(
        self,
        prompt: str,
        degradation_level: float = 0.5,
        style: DegradationStyle = DegradationStyle.POETIC_NONSENSE
    ) -> str:
        """
        Simple interface for interactive generation.
        Returns just the generated text.
        """
        result = self.generate(
            prompt=prompt,
            degradation_level=degradation_level,
            style=style,
            add_marker=True
        )
        return result.text