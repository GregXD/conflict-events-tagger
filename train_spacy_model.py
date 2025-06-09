#!/usr/bin/env python3
"""
Script to train a spaCy text classification model for conflict event categorization.
This script uses the JSONL training data to create a model that can classify
conflict events into different categories.
"""

import json
import random
import spacy
from spacy.training import Example
import os
from pathlib import Path

def load_training_data(jsonl_file):
    """Load training data from JSONL file and convert to spaCy format."""
    training_data = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            text = data['text']
            label = data['label']
            
            # spaCy textcat format: (text, {"cats": {"label1": 0, "label2": 1, ...}})
            training_data.append((text, {"cats": {label: 1.0}}))
    
    return training_data

def get_unique_labels(training_data):
    """Extract all unique labels from training data."""
    labels = set()
    for _, annotations in training_data:
        labels.update(annotations['cats'].keys())
    return sorted(list(labels))

def create_textcat_data(training_data, all_labels):
    """Convert training data to proper textcat format with all labels."""
    textcat_data = []
    
    for text, annotations in training_data:
        # Create cats dict with all labels (0.0 for negative, 1.0 for positive)
        cats = {label: 0.0 for label in all_labels}
        
        # Set the correct label to 1.0
        for label in annotations['cats']:
            if label in cats:
                cats[label] = 1.0
        
        textcat_data.append((text, {"cats": cats}))
    
    return textcat_data

def train_model(training_data, model_name="conflict_classifier", output_dir="./models"):
    """Train a spaCy text classification model."""
    
    # Get all unique labels
    all_labels = get_unique_labels(training_data)
    print(f"Found {len(all_labels)} unique labels: {all_labels}")
    
    # Convert to proper textcat format
    training_data = create_textcat_data(training_data, all_labels)
    
    # Create a blank English model
    nlp = spacy.blank("en")
    
    # Add text classification pipe
    textcat = nlp.add_pipe("textcat", last=True)
    
    # Add labels to text classifier
    for label in all_labels:
        textcat.add_label(label)
    
    # Split data into train/test
    random.shuffle(training_data)
    split_idx = int(len(training_data) * 0.8)
    train_data = training_data[:split_idx]
    test_data = training_data[split_idx:]
    
    print(f"Training on {len(train_data)} examples, testing on {len(test_data)} examples")
    
    # Training
    nlp.initialize()
    
    # Training loop
    n_iter = 20
    batch_size = 8
    
    print("Starting training...")
    for epoch in range(n_iter):
        losses = {}
        
        # Shuffle training data
        random.shuffle(train_data)
        
        # Create batches
        batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
        
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            
            nlp.update(examples, losses=losses)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{n_iter}, Loss: {losses.get('textcat', 0):.4f}")
    
    # Evaluate on test data
    print("\nEvaluating model...")
    correct = 0
    total = 0
    
    for text, annotations in test_data:
        doc = nlp(text)
        predicted_label = max(doc.cats, key=doc.cats.get)
        actual_label = max(annotations['cats'], key=annotations['cats'].get)
        
        if predicted_label == actual_label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save the model
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_path)
    
    print(f"Model saved to: {output_path}")
    
    # Test the model with a sample
    print("\nTesting model with sample text:")
    sample_text = "Government forces clashed with rebel groups in the northern region, with reports of heavy fighting and artillery exchanges."
    doc = nlp(sample_text)
    print(f"Sample: {sample_text}")
    print("Predictions:")
    for label, score in sorted(doc.cats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {score:.4f}")
    
    return nlp, accuracy

def main():
    # Configuration
    TRAINING_FILE = "acled_long_paragraph_training_data.jsonl"
    MODEL_NAME = "conflict_event_classifier"
    OUTPUT_DIR = "./models"
    
    # Check if training file exists
    if not os.path.exists(TRAINING_FILE):
        print(f"Error: Training file {TRAINING_FILE} not found!")
        print("Please make sure the JSONL training file is in the current directory.")
        return
    
    # Load training data
    print(f"Loading training data from {TRAINING_FILE}...")
    training_data = load_training_data(TRAINING_FILE)
    print(f"Loaded {len(training_data)} training examples")
    
    # Train the model
    model, accuracy = train_model(training_data, MODEL_NAME, OUTPUT_DIR)
    
    print(f"\nTraining completed!")
    print(f"Final accuracy: {accuracy:.4f}")
    print(f"Model saved as: {OUTPUT_DIR}/{MODEL_NAME}")

if __name__ == "__main__":
    main() 