#!/usr/bin/env python3
"""
Test script to debug spaCy classifier integration
"""

print("Testing spaCy classifier integration...")

# Test 1: Check if spaCy can be imported
try:
    import spacy
    print("✓ spaCy imported successfully")
except ImportError as e:
    print("✗ spaCy import failed:", e)
    exit(1)

# Test 2: Check if our spacy_classifier module can be imported
try:
    import spacy_classifier
    print("✓ spacy_classifier module imported successfully")
except ImportError as e:
    print("✗ spacy_classifier import failed:", e)
    exit(1)

# Test 3: Check if the model is available
try:
    available = spacy_classifier.is_spacy_available()
    print(f"spaCy model available: {available}")
except Exception as e:
    print("✗ Error checking spaCy availability:", e)

# Test 4: Try to get the classifier instance
try:
    classifier = spacy_classifier.get_spacy_classifier()
    print(f"Classifier instance created: {classifier is not None}")
    print(f"Model loaded: {classifier.nlp is not None}")
    print(f"Model path: {classifier.model_path}")
except Exception as e:
    print("✗ Error getting classifier:", e)

# Test 5: Check if model directory exists
import os
model_path = "./models/conflict_event_classifier"
print(f"Model directory exists: {os.path.exists(model_path)}")
if os.path.exists(model_path):
    print(f"Contents: {os.listdir(model_path)}")

# Test 6: Try a sample classification
if spacy_classifier.is_spacy_available():
    try:
        test_text = "Government forces clashed with rebel groups in the northern region."
        result = spacy_classifier.classify_with_spacy(test_text)
        if result:
            print(f"✓ Sample classification successful:")
            print(f"  Prediction: {result.prediction}")
            print(f"  Confidence: {result.confidence:.4f}")
        else:
            print("✗ Classification returned None")
    except Exception as e:
        print("✗ Classification failed:", e)
else:
    print("✗ spaCy not available, skipping classification test")

print("\nDebugging complete!") 