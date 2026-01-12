"""
Dataset Loading and Preprocessing Module

Provides consistent data loading for both training and runtime inference.
Ensures sentence boundaries match between training and production highlight extraction.

Datasets supported:
- Reddit TL;DR (summarization)
- SAMSum (dialogue summarization)
- Custom forum data (Foru.ms integration)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
import re

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ProcessedSample:
    """Standardized sample format for all datasets."""
    id: str
    source_text: str
    target_summary: str
    sentences: List[str]
    sentence_offsets: List[Tuple[int, int]]  # (start, end) character offsets
    metadata: Dict[str, Any]


class SentenceSplitter:
    """
    Consistent sentence splitting for training and runtime.
    
    CRITICAL: This splitter must produce identical boundaries during training
    and runtime inference, otherwise highlight provenance will be broken.
    """
    
    # Regex pattern for sentence boundaries
    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|'  # Standard sentence end
        r'(?<=[.!?])\s*\n+|'        # Sentence end with newline
        r'(?<=\n)\s*(?=[A-Z])'      # Newline followed by capital
    )
    
    def split(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into sentences with character offsets.
        
        Returns:
            List of (sentence, start_offset, end_offset) tuples
        """
        sentences = []
        current_pos = 0
        
        for match in self.SENTENCE_PATTERN.finditer(text):
            sentence = text[current_pos:match.start()].strip()
            if sentence and len(sentence) >= 10:  # Skip very short fragments
                sentences.append((sentence, current_pos, match.start()))
            current_pos = match.end()
        
        # Handle final sentence
        final = text[current_pos:].strip()
        if final and len(final) >= 10:
            sentences.append((final, current_pos, len(text)))
        
        return sentences


class RedditTLDRDataset(Dataset):
    """
    Reddit TL;DR dataset for summarization training.
    
    Dataset structure:
    - source: Full reddit post content
    - target: TL;DR summary written by author
    """
    
    def __init__(
        self,
        split: str = "train",
        tokenizer: Any = None,
        max_source_length: int = 512,
        max_target_length: int = 128,
        cache_dir: Optional[str] = None,
    ):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.splitter = SentenceSplitter()
        
        # Load dataset
        logger.info(f"Loading Reddit TL;DR dataset ({split})...")
        self.dataset = load_dataset(
            "webis/tldr-17",
            split=split,
            cache_dir=cache_dir,
        )
        logger.info(f"Loaded {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        
        source = item.get("content", item.get("body", ""))
        target = item.get("summary", item.get("tldr", ""))
        
        # Split into sentences for highlight training
        sentences_with_offsets = self.splitter.split(source)
        sentences = [s[0] for s in sentences_with_offsets]
        
        if self.tokenizer:
            # Tokenize for seq2seq training
            source_enc = self.tokenizer(
                source,
                max_length=self.max_source_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            target_enc = self.tokenizer(
                target,
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            
            return {
                "input_ids": source_enc["input_ids"].squeeze(),
                "attention_mask": source_enc["attention_mask"].squeeze(),
                "labels": target_enc["input_ids"].squeeze(),
                "sentences": sentences,
                "summary": target,
            }
        
        return {
            "source": source,
            "target": target,
            "sentences": sentences,
        }


class SAMSumDataset(Dataset):
    """
    SAMSum dialogue summarization dataset.
    
    Useful for training on conversational/forum-like data.
    """
    
    def __init__(
        self,
        split: str = "train",
        tokenizer: Any = None,
        max_source_length: int = 512,
        max_target_length: int = 128,
        cache_dir: Optional[str] = None,
    ):
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.splitter = SentenceSplitter()
        
        logger.info(f"Loading SAMSum dataset ({split})...")
        self.dataset = load_dataset("samsum", split=split, cache_dir=cache_dir)
        logger.info(f"Loaded {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        
        dialogue = item["dialogue"]
        summary = item["summary"]
        
        # Parse dialogue turns as sentences
        turns = dialogue.split("\r\n")
        sentences = [t.strip() for t in turns if t.strip()]
        
        if self.tokenizer:
            source_enc = self.tokenizer(
                dialogue,
                max_length=self.max_source_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            target_enc = self.tokenizer(
                summary,
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            
            return {
                "input_ids": source_enc["input_ids"].squeeze(),
                "attention_mask": source_enc["attention_mask"].squeeze(),
                "labels": target_enc["input_ids"].squeeze(),
                "sentences": sentences,
                "summary": summary,
            }
        
        return {
            "source": dialogue,
            "target": summary,
            "sentences": sentences,
        }


class SentenceClassificationDataset(Dataset):
    """
    Dataset for sentence classification (highlight categorization).
    
    Categories:
    - fact: Factual statements
    - solution: Solutions or recommendations
    - opinion: Subjective opinions
    - question: Questions posed
    - citation: References to external sources
    - irrelevant: Off-topic or filler content
    """
    
    LABEL2ID = {
        "fact": 0,
        "solution": 1,
        "opinion": 2,
        "question": 3,
        "citation": 4,
        "irrelevant": 5,
    }
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 128,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        sentence = item["sentence"]
        label = self.LABEL2ID.get(item["category"], 5)  # Default to irrelevant
        
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
    
    @classmethod
    def from_summarization_dataset(
        cls,
        summarization_data: List[Dict[str, Any]],
        tokenizer: Any,
    ) -> "SentenceClassificationDataset":
        """
        Create classification dataset from summarization data using heuristics.
        
        This enables semi-supervised training without manual labeling.
        """
        classification_data = []
        
        for item in summarization_data:
            sentences = item.get("sentences", [])
            summary = item.get("target", item.get("summary", "")).lower()
            
            for sentence in sentences:
                # Heuristic labeling
                s_lower = sentence.lower()
                
                if "?" in sentence:
                    category = "question"
                elif any(w in s_lower for w in ["http", "www", "according to", "source:"]):
                    category = "citation"
                elif any(w in s_lower for w in ["i think", "i believe", "in my opinion", "imo"]):
                    category = "opinion"
                elif any(w in s_lower for w in ["solution", "fix", "try", "should", "recommend"]):
                    category = "solution"
                elif any(word in summary for word in s_lower.split()[:5]):
                    # Sentences whose words appear in summary are likely facts
                    category = "fact"
                else:
                    category = "irrelevant"
                
                classification_data.append({
                    "sentence": sentence,
                    "category": category,
                })
        
        return cls(classification_data, tokenizer)


def create_dataloaders(
    dataset_name: str,
    tokenizer: Any,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for a dataset.
    
    Args:
        dataset_name: One of "reddit_tldr", "samsum"
        tokenizer: Tokenizer for encoding
        batch_size: Batch size for all loaders
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if dataset_name == "reddit_tldr":
        train_ds = RedditTLDRDataset("train", tokenizer)
        val_ds = RedditTLDRDataset("validation", tokenizer)
        test_ds = RedditTLDRDataset("test", tokenizer)
    elif dataset_name == "samsum":
        train_ds = SAMSumDataset("train", tokenizer)
        val_ds = SAMSumDataset("validation", tokenizer)
        test_ds = SAMSumDataset("test", tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
