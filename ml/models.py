"""
Model Building and PAI Conversion

Provides model constructors and PerforatedAI conversion for:
- T5 summarization models
- BART summarization models
- BERT/SBERT embedding models

Usage:
    from ml.models import build_summarizer, convert_model_for_pai
    
    model = build_summarizer("t5-small")
    model, modules = convert_model_for_pai(model, "t5")
"""

import logging
from typing import Optional, List, Tuple, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_summarizer(
    model_name: str = "t5-small",
    tokenizer: Any = None,
    config: Any = None,
) -> nn.Module:
    """
    Build a summarization model.
    
    Args:
        model_name: HuggingFace model name or path
        tokenizer: Optional tokenizer (will be loaded if not provided)
        config: Optional ExperimentConfig
        
    Returns:
        Model ready for training
    """
    from transformers import AutoModelForSeq2SeqLM
    
    logger.info(f"Building summarizer: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model with {total_params:,} parameters")
    
    return model


def build_encoder(
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    config: Any = None,
) -> nn.Module:
    """
    Build a sentence encoder model.
    
    Args:
        encoder_name: HuggingFace model name or path
        config: Optional ExperimentConfig
        
    Returns:
        Encoder model ready for training
    """
    from transformers import AutoModel
    
    logger.info(f"Building encoder: {encoder_name}")
    model = AutoModel.from_pretrained(encoder_name)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded encoder with {total_params:,} parameters")
    
    return model


class _PAILayerWrapper(nn.Module):
    """Wrapper to keep LayerNorms internal to converted modules."""
    
    def __init__(self, module: nn.Module, name: str = ""):
        super().__init__()
        self.module = module
        self.name = name
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def convert_model_for_pai(
    model: nn.Module,
    model_type: str,
    modules_to_convert: Optional[List[str]] = None,
) -> Tuple[nn.Module, List[str]]:
    """
    Convert a model for PerforatedAI dendritic optimization.
    
    This is the main conversion entry point. It:
    1. Wraps transformer layers to preserve LayerNorm placement
    2. Calls PA.convert_network() with appropriate modules
    3. Verifies conversion and logs statistics
    
    Args:
        model: Model to convert
        model_type: One of "t5", "bart", "bert", "sbert"
        modules_to_convert: Optional explicit module names
        
    Returns:
        Tuple of (converted_model, list_of_converted_modules)
    """
    logger.info("=" * 60)
    logger.info(f"Converting model for PerforatedAI: {model_type}")
    logger.info("=" * 60)
    
    # Determine modules to convert based on model type
    model_type_lower = model_type.lower()
    
    if "t5" in model_type_lower:
        # Wrap T5 blocks
        if hasattr(model, "encoder") and hasattr(model.encoder, "block"):
            for i, block in enumerate(model.encoder.block):
                if not isinstance(block, _PAILayerWrapper):
                    model.encoder.block[i] = _PAILayerWrapper(block, f"encoder.block.{i}")
        if hasattr(model, "decoder") and hasattr(model.decoder, "block"):
            for i, block in enumerate(model.decoder.block):
                if not isinstance(block, _PAILayerWrapper):
                    model.decoder.block[i] = _PAILayerWrapper(block, f"decoder.block.{i}")
        default_modules = ["encoder.block", "decoder.block"]
        
    elif "bart" in model_type_lower:
        base = model.model if hasattr(model, "model") else model
        if hasattr(base, "encoder") and hasattr(base.encoder, "layers"):
            for i, layer in enumerate(base.encoder.layers):
                if not isinstance(layer, _PAILayerWrapper):
                    base.encoder.layers[i] = _PAILayerWrapper(layer, f"encoder.layers.{i}")
        if hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
            for i, layer in enumerate(base.decoder.layers):
                if not isinstance(layer, _PAILayerWrapper):
                    base.decoder.layers[i] = _PAILayerWrapper(layer, f"decoder.layers.{i}")
        default_modules = ["model.encoder.layers", "model.decoder.layers"]
        
    elif "bert" in model_type_lower or "sbert" in model_type_lower:
        # Find encoder layers
        layers = None
        if hasattr(model, "bert"):
            layers = model.bert.encoder.layer
        elif hasattr(model, "roberta"):
            layers = model.roberta.encoder.layer
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            layers = model.encoder.layer
        
        if layers:
            for i, layer in enumerate(layers):
                if not isinstance(layer, _PAILayerWrapper):
                    layers[i] = _PAILayerWrapper(layer, f"encoder.layer.{i}")
        default_modules = ["encoder.layer"]
        
    else:
        logger.warning(f"Unknown model_type: {model_type}, attempting generic conversion")
        default_modules = []
    
    modules = modules_to_convert or default_modules
    
    # Call PAI conversion
    try:
        import perforatedai as PA
        
        PA.convert_network(model, module_names_to_convert=modules)
        logger.info(f"PAI conversion complete for modules: {modules}")
        
    except ImportError:
        logger.warning("perforatedai not installed - model not converted")
    except Exception as e:
        logger.error(f"PAI conversion failed: {e}")
        raise
    
    # Log statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info("=" * 60)
    
    return model, modules


def save_model_pai_aware(model: nn.Module, path: str, tracker: Any = None):
    """
    Save model with PAI metadata.
    
    Args:
        model: Model to save
        path: Save path
        tracker: Optional PAI tracker for metadata
    """
    save_dict = {
        "model_state_dict": model.state_dict(),
    }
    
    if tracker:
        try:
            save_dict["pai_tracker_state"] = tracker.state_dict()
        except:
            pass
    
    torch.save(save_dict, path)
    logger.info(f"Model saved to {path}")


def load_model_pai_aware(path: str, model: nn.Module, map_location: str = None) -> nn.Module:
    """
    Load model with PAI metadata.
    
    Args:
        path: Checkpoint path
        model: Model architecture to load into
        map_location: Device mapping
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model loaded from {path}")
    return model
