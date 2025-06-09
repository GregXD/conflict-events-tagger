#!/usr/bin/env python3
"""
Improved spaCy Model Training with Massive Synthetic Dataset

This script trains a high-performance spaCy text classification model using
the massive synthetic dataset with advanced techniques and proper evaluation.
"""

import json
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import os
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_training_data(jsonl_file):
    """Load training data from JSONL file and convert to spaCy format."""
    training_data = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            text = data['text']
            label = data['label']
            training_data.append((text, label))
    
    print(f"Loaded {len(training_data)} training examples")
    return training_data

def analyze_dataset(training_data):
    """Analyze the dataset distribution and characteristics."""
    labels = [label for _, label in training_data]
    label_counts = Counter(labels)
    
    print("\nDataset Analysis:")
    print(f"Total examples: {len(training_data)}")
    print(f"Unique labels: {len(label_counts)}")
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        percentage = (count / len(training_data)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Text length analysis
    text_lengths = [len(text.split()) for text, _ in training_data]
    print(f"\nText length statistics:")
    print(f"  Average words per text: {np.mean(text_lengths):.1f}")
    print(f"  Min words: {min(text_lengths)}")
    print(f"  Max words: {max(text_lengths)}")
    print(f"  Median words: {np.median(text_lengths):.1f}")
    
    return label_counts

def prepare_spacy_data(training_data, test_size=0.2):
    """Convert data to spaCy format and split into train/test."""
    # Get unique labels
    all_labels = list(set([label for _, label in training_data]))
    print(f"Labels: {all_labels}")
    
    # Split data
    train_data, test_data = train_test_split(
        training_data, 
        test_size=test_size, 
        random_state=42, 
        stratify=[label for _, label in training_data]
    )
    
    print(f"Training examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")
    
    # Convert to spaCy format
    def convert_to_spacy_format(data):
        spacy_data = []
        for text, label in data:
            # Create cats dict with all labels
            cats = {lab: 0.0 for lab in all_labels}
            cats[label] = 1.0
            spacy_data.append((text, {"cats": cats}))
        return spacy_data
    
    train_spacy = convert_to_spacy_format(train_data)
    test_spacy = convert_to_spacy_format(test_data)
    
    return train_spacy, test_spacy, all_labels

def create_improved_model(labels):
    """Create an improved spaCy model with better architecture."""
    # Create a blank English model
    nlp = spacy.blank("en")
    
    # Add the text classifier with improved configuration
    config = {
        "threshold": 0.5,
        "model": {
            "@architectures": "spacy.TextCatBOW.v2",
            "exclusive_classes": True,
            "ngram_size": 2,  # Use bigrams for better context
            "no_output_layer": False,
            "nO": len(labels)
        }
    }
    
    # Add textcat component
    textcat = nlp.add_pipe("textcat", config=config)
    
    # Add labels to the text classifier
    for label in labels:
        textcat.add_label(label)
    
    return nlp

def train_model(nlp, train_data, n_iter=20):
    """Train the model with improved training loop."""
    print(f"\nTraining model for {n_iter} iterations...")
    
    # Get the textcat component
    textcat = nlp.get_pipe("textcat")
    
    # Convert training data to examples
    examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    
    # Initialize the model
    nlp.initialize(lambda: examples)
    
    # Training loop with dynamic batch sizes
    losses_history = []
    
    for iteration in range(n_iter):
        losses = {}
        random.shuffle(examples)
        
        # Use dynamic batch sizes that start small and grow
        batch_sizes = compounding(4.0, 32.0, 1.001)
        batches = minibatch(examples, size=batch_sizes)
        
        for batch in batches:
            nlp.update(batch, drop=0.2, losses=losses)  # Add dropout
        
        losses_history.append(losses['textcat'])
        
        if iteration % 5 == 0:
            print(f"Iteration {iteration}, Loss: {losses['textcat']:.4f}")
    
    print(f"Training completed. Final loss: {losses['textcat']:.4f}")
    return losses_history

def evaluate_model(nlp, test_data, labels):
    """Comprehensive model evaluation."""
    print("\nEvaluating model...")
    
    true_labels = []
    predicted_labels = []
    confidence_scores = []
    
    for text, annotations in test_data:
        doc = nlp(text)
        
        # Get true label
        true_label = max(annotations['cats'], key=annotations['cats'].get)
        true_labels.append(true_label)
        
        # Get predicted label and confidence
        predicted_label = max(doc.cats, key=doc.cats.get)
        predicted_labels.append(predicted_label)
        confidence_scores.append(doc.cats[predicted_label])
    
    # Calculate accuracy
    accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)
    avg_confidence = np.mean(confidence_scores)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average confidence: {avg_confidence:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=labels))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    print(cm)
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    return accuracy, avg_confidence

def save_model(nlp, output_dir="./models/improved_conflict_classifier"):
    """Save the trained model."""
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    nlp.to_disk(output_path)
    print(f"Model saved to {output_path}")

def plot_training_curve(losses_history):
    """Plot the training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses_history)
    plt.title('Training Loss Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    print("Training curve saved as 'training_curve.png'")

def cross_validate_model(training_data, labels, k_folds=5):
    """Perform k-fold cross-validation."""
    print(f"\nPerforming {k_folds}-fold cross-validation...")
    
    # Shuffle data
    random.shuffle(training_data)
    
    # Split into k folds
    fold_size = len(training_data) // k_folds
    accuracies = []
    
    for fold in range(k_folds):
        print(f"  Fold {fold + 1}/{k_folds}")
        
        # Create train/val split for this fold
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size
        
        val_data = training_data[start_idx:end_idx]
        train_data = training_data[:start_idx] + training_data[end_idx:]
        
        # Convert to spaCy format
        def convert_fold_data(data):
            spacy_data = []
            for text, label in data:
                cats = {lab: 0.0 for lab in labels}
                cats[label] = 1.0
                spacy_data.append((text, {"cats": cats}))
            return spacy_data
        
        train_spacy = convert_fold_data(train_data)
        val_spacy = convert_fold_data(val_data)
        
        # Create and train model
        nlp = create_improved_model(labels)
        train_model(nlp, train_spacy, n_iter=10)  # Fewer iterations for CV
        
        # Evaluate
        accuracy, _ = evaluate_model(nlp, val_spacy, labels)
        accuracies.append(accuracy)
    
    cv_mean = np.mean(accuracies)
    cv_std = np.std(accuracies)
    
    print(f"\nCross-validation results:")
    print(f"  Mean accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  Individual fold accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
    
    return cv_mean, cv_std

def main():
    """Main training function."""
    print("🚀 Starting improved spaCy model training with massive synthetic dataset")
    
    # Check if combined dataset exists
    combined_file = "combined_training_data.jsonl"
    if not os.path.exists(combined_file):
        print(f"❌ Combined dataset not found: {combined_file}")
        print("Please run the synthetic data generator first!")
        return
    
    # Load and analyze data
    training_data = load_training_data(combined_file)
    label_counts = analyze_dataset(training_data)
    
    # Prepare data for training
    train_data, test_data, labels = prepare_spacy_data(training_data)
    
    # Create improved model
    nlp = create_improved_model(labels)
    
    # Optional: Perform cross-validation first
    print("\n" + "="*50)
    cv_choice = input("Perform cross-validation first? (y/n): ").lower().strip()
    if cv_choice == 'y':
        # Convert back to simple format for CV
        simple_data = [(text, max(cats, key=cats.get)) for text, cats in 
                       [(text, ann['cats']) for text, ann in train_data + test_data]]
        cross_validate_model(simple_data, labels)
    
    # Train final model
    print("\n" + "="*50)
    print("Training final model...")
    losses_history = train_model(nlp, train_data, n_iter=30)
    
    # Evaluate model
    accuracy, confidence = evaluate_model(nlp, test_data, labels)
    
    # Plot training curve
    plot_training_curve(losses_history)
    
    # Save model
    save_model(nlp)
    
    # Summary
    print("\n" + "="*50)
    print("🎉 Training completed successfully!")
    print(f"Final model accuracy: {accuracy:.4f}")
    print(f"Average confidence: {confidence:.4f}")
    print(f"Model saved to: ./models/improved_conflict_classifier")
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - training_curve.png")
    print("  - ./models/improved_conflict_classifier/ (model files)")
    
    # Performance comparison
    if accuracy > 0.8:
        print("\n🎯 Excellent performance! Model is ready for deployment.")
    elif accuracy > 0.7:
        print("\n📊 Good performance! Consider generating more data for further improvement.")
    else:
        print("\n⚠️ Moderate performance. Consider:")
        print("   - Generating more synthetic data")
        print("   - Adjusting model architecture")
        print("   - Improving data quality")

if __name__ == "__main__":
    main() 