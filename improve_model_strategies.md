# Strategies to Build a Strong Local Conflict Classification Model

## 🎯 1. Expand Training Data (Highest Impact)

### Current Status: 521 examples → Target: 2,000-10,000+ examples

**Quick Wins:**
- **Use your existing Cohere analyses**: Export all your analyzed events from the database as training data
- **ACLED dataset**: Download conflict event data from Armed Conflict Location & Event Data Project
- **News archives**: Scrape historical conflict news with known classifications
- **Academic datasets**: Look for political violence/conflict datasets

**Data Collection Script:**
```python
# Export your existing analyses as training data
def export_training_data():
    analyses = execute_db_query("SELECT summary, event_type FROM events WHERE summary IS NOT NULL")
    with open("additional_training.jsonl", "w") as f:
        for summary, event_type in analyses:
            json.dump({"text": summary, "label": event_type}, f)
            f.write("\n")
```

## 🔄 2. Data Augmentation Techniques

### Synthetic Data Generation
- **Paraphrasing**: Use AI to rephrase existing examples
- **Back-translation**: Translate to another language and back
- **Keyword substitution**: Replace locations/names with similar ones
- **Sentence mixing**: Combine parts of different examples

### Example Implementation:
```python
# Simple augmentation by location substitution
locations = ["Syria", "Iraq", "Afghanistan", "Somalia", "Yemen"]
def augment_by_location(text, original_location):
    augmented = []
    for loc in locations:
        if loc != original_location:
            augmented.append(text.replace(original_location, loc))
    return augmented
```

## 🏗️ 3. Model Architecture Improvements

### Upgrade to Transformer Models
```python
# Use transformer-based model instead of CNN
config = Config().from_str("""
[model]
@architectures = "spacy-transformers.TransformerModel.v3"
name = "distilbert-base-uncased"
tokenizer_config = {"use_fast": true}

[model.get_spans]
@architectures = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96
""")
```

### Ensemble Multiple Models
- Train several models with different architectures
- Combine predictions using voting or averaging
- Reduces overfitting and improves robustness

## ⚙️ 4. Advanced Training Techniques

### Hyperparameter Optimization
```python
# Better training configuration
config = {
    "threshold": 0.5,
    "model": {
        "@architectures": "spacy.TextCatBOW.v2",
        "exclusive_classes": True,
        "ngram_size": 2,  # Use bigrams
        "no_output_layer": False,
        "nO": len(labels)
    },
    "optimizer": {
        "@optimizers": "Adam.v1",
        "beta1": 0.9,
        "beta2": 0.999,
        "learn_rate": 0.001
    }
}

# Training with better parameters
nlp.update([example], drop=0.2, losses=losses)  # Add dropout
```

### Active Learning Pipeline
```python
# Identify uncertain predictions for human review
def find_uncertain_predictions(texts, threshold=0.6):
    uncertain = []
    for text in texts:
        doc = nlp(text)
        max_score = max(doc.cats.values())
        if max_score < threshold:
            uncertain.append((text, doc.cats))
    return uncertain
```

## 📊 5. Data Quality Improvements

### Label Consistency
- Review and clean existing labels
- Use inter-annotator agreement metrics
- Create clear labeling guidelines

### Balanced Dataset
```python
# Check class distribution
from collections import Counter
labels = [item['label'] for item in training_data]
distribution = Counter(labels)
print("Class distribution:", distribution)

# Oversample minority classes or undersample majority classes
```

### Text Preprocessing
```python
def preprocess_text(text):
    # Remove URLs, clean formatting
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    # Normalize quotes, dashes
    text = text.replace('"', '"').replace('"', '"')
    return text.strip()
```

## 🔧 6. Feature Engineering

### Custom Features
```python
# Add custom pipeline components
@Language.component("conflict_features")
def add_conflict_features(doc):
    # Add features like violence indicators, location mentions
    violence_words = {"killed", "died", "casualties", "wounded", "injured"}
    doc._.has_violence = any(token.lemma_ in violence_words for token in doc)
    return doc

nlp.add_pipe("conflict_features")
```

### Named Entity Recognition
- Pre-train on location/organization recognition
- Use NER features as input to classification

## 📈 7. Evaluation and Monitoring

### Cross-Validation
```python
from sklearn.model_selection import KFold

def cross_validate_model(data, k=5):
    kfold = KFold(n_splits=k, shuffle=True)
    scores = []
    
    for train_idx, val_idx in kfold.split(data):
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        
        # Train model on train_data
        # Evaluate on val_data
        # Store score
        
    return np.mean(scores), np.std(scores)
```

### Performance Metrics
```python
from sklearn.metrics import classification_report, confusion_matrix

def detailed_evaluation(y_true, y_pred, labels):
    print(classification_report(y_true, y_pred, target_names=labels))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
```

## 🚀 8. Advanced Techniques

### Transfer Learning
```python
# Start with a pre-trained model
nlp = spacy.load("en_core_web_lg")  # Large model with better vectors
# Add textcat to existing model instead of training from scratch
```

### Multi-task Learning
- Train on related tasks simultaneously (sentiment, topic classification)
- Share lower layers, separate output heads

### Domain Adaptation
- Pre-train on general news data
- Fine-tune on conflict-specific data

## 📋 9. Implementation Priority

### Phase 1 (Quick Wins - 1-2 weeks):
1. Export your existing Cohere analyses as training data
2. Download ACLED or similar datasets
3. Implement basic data augmentation
4. Retrain with 2-5x more data

### Phase 2 (Medium Term - 1 month):
1. Upgrade to transformer-based architecture
2. Implement cross-validation
3. Add custom features and preprocessing
4. Set up evaluation pipeline

### Phase 3 (Advanced - 2-3 months):
1. Build ensemble models
2. Implement active learning
3. Set up continuous improvement pipeline
4. Add domain-specific features

## 📊 Expected Performance Gains

| Improvement | Expected Accuracy Gain |
|-------------|----------------------|
| 5x more training data | +10-20% |
| Transformer architecture | +5-15% |
| Data augmentation | +3-8% |
| Ensemble methods | +2-5% |
| Better preprocessing | +2-5% |

**Total potential improvement: +20-50% accuracy**

## 💡 Immediate Next Steps

1. **Export your database**: Your Cohere analyses are gold-standard training data
2. **Download ACLED data**: Free, high-quality conflict event dataset
3. **Implement the improved training script** with better architecture
4. **Set up evaluation metrics** to track improvements

Would you like me to help implement any of these strategies? 