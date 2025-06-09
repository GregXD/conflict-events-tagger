"""
spaCy-based text classification for conflict events.
This module provides an alternative to the Cohere API for event classification.
"""

import spacy
import os
from pathlib import Path
import logging
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)

class ClassificationResult(NamedTuple):
    """Result structure to match Cohere API response format."""
    prediction: str
    confidence: float

class SpacyClassifier:
    """spaCy-based text classifier for conflict events."""
    
    def __init__(self, model_path: str = "./models/conflict_event_classifier"):
        self.model_path = model_path
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained spaCy model."""
        try:
            if os.path.exists(self.model_path):
                self.nlp = spacy.load(self.model_path)
                logger.info(f"Successfully loaded spaCy model from {self.model_path}")
            else:
                logger.warning(f"spaCy model not found at {self.model_path}")
                self.nlp = None
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            self.nlp = None
    
    def is_available(self) -> bool:
        """Check if the spaCy model is loaded and available."""
        return self.nlp is not None
    
    def classify(self, text: str) -> Optional[ClassificationResult]:
        """
        Classify text using the spaCy model.
        
        Args:
            text: Text to classify
            
        Returns:
            ClassificationResult with prediction and confidence, or None if model unavailable
        """
        if not self.is_available():
            return None
        
        try:
            # Process the text
            doc = self.nlp(text)
            
            # Get predictions
            predictions = doc.cats
            
            # Find the label with highest confidence
            best_label = max(predictions, key=predictions.get)
            confidence = predictions[best_label]
            
            return ClassificationResult(
                prediction=best_label,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error during spaCy classification: {e}")
            return None
    
    def get_all_predictions(self, text: str) -> Optional[dict]:
        """
        Get all predictions with confidence scores.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary of label -> confidence scores, or None if model unavailable
        """
        if not self.is_available():
            return None
        
        try:
            doc = self.nlp(text)
            return dict(doc.cats)
        except Exception as e:
            logger.error(f"Error getting all predictions: {e}")
            return None

# Global instance
_spacy_classifier = None

def get_spacy_classifier() -> SpacyClassifier:
    """Get or create the global spaCy classifier instance."""
    global _spacy_classifier
    if _spacy_classifier is None:
        _spacy_classifier = SpacyClassifier()
    return _spacy_classifier

def classify_with_spacy(text: str) -> Optional[ClassificationResult]:
    """
    Convenience function to classify text with spaCy.
    
    Args:
        text: Text to classify
        
    Returns:
        ClassificationResult or None if spaCy model unavailable
    """
    classifier = get_spacy_classifier()
    return classifier.classify(text)

def is_spacy_available() -> bool:
    """Check if spaCy classifier is available."""
    classifier = get_spacy_classifier()
    return classifier.is_available() 