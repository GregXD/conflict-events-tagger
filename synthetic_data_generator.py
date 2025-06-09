#!/usr/bin/env python3
"""
Synthetic Data Generator for Conflict Event Classification

This script generates thousands of synthetic training examples using various
augmentation techniques including AI-powered paraphrasing, template generation,
and systematic variations of existing data.
"""

import json
import random
import re
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import cohere
import os
from dotenv import load_dotenv
import time
from itertools import combinations

# Load environment variables
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

class SyntheticDataGenerator:
    def __init__(self, original_data_file: str = "acled_long_paragraph_training_data.jsonl"):
        self.original_data_file = original_data_file
        self.original_data = []
        self.synthetic_data = []
        self.load_original_data()
        
        # Predefined lists for systematic substitution
        self.countries = [
            "Syria", "Iraq", "Afghanistan", "Somalia", "Yemen", "Libya", "Sudan", 
            "Nigeria", "Mali", "Chad", "Pakistan", "India", "Myanmar", "Ethiopia",
            "Democratic Republic of Congo", "Central African Republic", "South Sudan",
            "Ukraine", "Israel", "Palestine", "Lebanon", "Turkey", "Iran", "Egypt",
            "Morocco", "Tunisia", "Algeria", "Kenya", "Tanzania", "Uganda", "Rwanda"
        ]
        
        self.cities = [
            "Kabul", "Baghdad", "Damascus", "Aleppo", "Mosul", "Kandahar", "Herat",
            "Basra", "Tikrit", "Fallujah", "Ramadi", "Kirkuk", "Erbil", "Sulaymaniyah",
            "Mazar-i-Sharif", "Jalalabad", "Lashkar Gah", "Gaza", "West Bank", "Beirut",
            "Tripoli", "Benghazi", "Khartoum", "Darfur", "Mogadishu", "Hargeisa",
            "Sana'a", "Aden", "Ta'izz", "Al Hudaydah", "Marawi", "Mindanao", "Sulu"
        ]
        
        self.organizations = [
            "government forces", "rebel groups", "insurgents", "militants", "fighters",
            "armed groups", "opposition forces", "paramilitary forces", "militia",
            "security forces", "police", "military", "army", "protesters", "demonstrators",
            "civilians", "peacekeepers", "tribal forces", "extremist groups"
        ]
        
        self.weapons = [
            "artillery", "mortars", "rockets", "missiles", "bombs", "explosives",
            "gunfire", "small arms", "heavy weapons", "improvised explosive devices",
            "car bombs", "suicide bombs", "airstrikes", "shelling", "grenades"
        ]
        
        self.time_expressions = [
            "yesterday", "today", "this morning", "last night", "earlier today",
            "on Monday", "on Tuesday", "on Wednesday", "on Thursday", "on Friday",
            "last week", "this week", "over the weekend", "during the night",
            "in the early hours", "at dawn", "in the afternoon", "in the evening"
        ]
        
        # Event-specific templates for generation
        self.event_templates = {
            "Violence against civilians": [
                "{org} attacked {location}, targeting {target}. {casualties} were {action} in the {time}.",
                "Civilians in {location} came under attack from {org}. Reports indicate {casualties} {action}.",
                "{org} carried out attacks on {target} in {location}, resulting in {casualties} {action}.",
                "An attack by {org} in {location} left {casualties} {action} and many more injured."
            ],
            "Battles": [
                "{org1} clashed with {org2} in {location}. Fighting lasted {duration} with {weapon} being used.",
                "Intense fighting broke out between {org1} and {org2} near {location}. {casualties} during the battle.",
                "{org1} launched an offensive against {org2} positions in {location}. The battle involved {weapon}.",
                "Combat operations between {org1} and {org2} resulted in {casualties} in the {location} region."
            ],
            "Explosions/Remote violence": [
                "An explosion occurred in {location} {time}, {cause}. {casualties} in the blast.",
                "{weapon} struck {location}, causing {casualties}. The attack targeted {target}.",
                "A {weapon} exploded in {location} during {time}, resulting in {casualties}.",
                "{location} was hit by {weapon} {time}, leaving {casualties} and significant damage."
            ],
            "Riots": [
                "Riots erupted in {location} following {trigger}. Protesters {actions} as {response}.",
                "Violent clashes broke out in {location} between {parties}. {casualties} during the unrest.",
                "Civil unrest in {location} led to {actions}. {authorities} responded with {response}.",
                "Rioting in {location} resulted in {casualties} and extensive property damage."
            ],
            "Protests": [
                "Demonstrators gathered in {location} to protest {cause}. The rally was {peaceful}.",
                "Thousands marched in {location} demanding {demands}. {authorities} {response}.",
                "A protest in {location} against {cause} drew {number} participants. {outcome}.",
                "Anti-{cause} demonstrations in {location} were met with {response} from authorities."
            ],
            "Strategic developments": [
                "{org} announced {development} in {location}. This move {significance}.",
                "A {agreement} was signed between {parties} regarding {location}. The deal {outcome}.",
                "{org} deployed forces to {location} as part of {operation}. The action {purpose}.",
                "Strategic talks between {parties} in {location} resulted in {outcome}."
            ]
        }

    def load_original_data(self):
        """Load the original training data"""
        with open(self.original_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.original_data.append(data)
        print(f"Loaded {len(self.original_data)} original examples")

    def analyze_data_distribution(self):
        """Analyze the distribution of event types in original data"""
        label_counts = Counter([item['label'] for item in self.original_data])
        print("\nOriginal data distribution:")
        for label, count in label_counts.most_common():
            print(f"  {label}: {count}")
        return label_counts

    def location_substitution(self, text: str, original_locations: List[str] = None) -> List[str]:
        """Generate variations by substituting locations"""
        variations = []
        
        # Find potential locations in text (simple heuristic)
        location_patterns = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        potential_locations = re.findall(location_patterns, text)
        
        for country in random.sample(self.countries, min(5, len(self.countries))):
            new_text = text
            for location in potential_locations[:2]:  # Replace up to 2 locations
                if len(location) > 3 and location.lower() not in ['the', 'and', 'for']:
                    new_text = new_text.replace(location, country, 1)
            if new_text != text:
                variations.append(new_text)
        
        return variations[:3]  # Return up to 3 variations

    def entity_substitution(self, text: str) -> List[str]:
        """Generate variations by substituting organizations and weapons"""
        variations = []
        
        # Organization substitution
        for org in random.sample(self.organizations, min(3, len(self.organizations))):
            new_text = text
            # Simple pattern matching for common terms
            patterns = ['government forces', 'rebel groups', 'militants', 'protesters', 'security forces']
            for pattern in patterns:
                if pattern in text.lower():
                    new_text = re.sub(re.escape(pattern), org, new_text, flags=re.IGNORECASE, count=1)
                    break
            if new_text != text:
                variations.append(new_text)
        
        return variations[:2]

    def paraphrase_with_cohere(self, text: str, label: str) -> List[str]:
        """Use Cohere to generate paraphrases"""
        prompt = f"""Rewrite the following text about a {label.lower()} event in 3 different ways, keeping the same meaning and event type but varying the language and structure. Make each version realistic and news-like.

Original: {text}

Version 1:"""
        
        try:
            response = co.generate(
                model='command',
                prompt=prompt,
                max_tokens=200,
                temperature=0.7,
                k=0,
                stop_sequences=['\n\n'],
                return_likelihoods='NONE'
            )
            
            result = response.generations[0].text.strip()
            
            # Split by "Version" to get multiple paraphrases
            versions = re.split(r'Version \d+:', result)
            paraphrases = [v.strip() for v in versions if v.strip() and len(v.strip()) > 20]
            
            return paraphrases[:3]
            
        except Exception as e:
            print(f"Cohere paraphrasing failed: {e}")
            return []

    def template_generation(self, label: str, count: int = 10) -> List[str]:
        """Generate new examples using templates"""
        if label not in self.event_templates:
            return []
        
        templates = self.event_templates[label]
        generated = []
        
        for _ in range(count):
            template = random.choice(templates)
            
            # Fill template with random values
            filled = template.format(
                org=random.choice(self.organizations),
                org1=random.choice(self.organizations),
                org2=random.choice(self.organizations),
                location=random.choice(self.countries + self.cities),
                target=random.choice(['civilians', 'government buildings', 'markets', 'schools', 'hospitals']),
                casualties=random.choice(['several people', 'dozens', 'at least 10 people', 'multiple civilians']),
                action=random.choice(['killed', 'wounded', 'injured', 'affected']),
                time=random.choice(self.time_expressions),
                weapon=random.choice(self.weapons),
                duration=random.choice(['several hours', 'throughout the day', 'into the night']),
                cause=random.choice(['reports of civilian casualties', 'alleged fraud', 'economic hardship']),
                actions=random.choice(['threw stones', 'set fires', 'blocked roads']),
                response=random.choice(['police deployed tear gas', 'authorities imposed curfew']),
                peaceful=random.choice(['largely peaceful', 'peaceful', 'without major incidents']),
                parties=random.choice(['government', 'opposition', 'rival groups']),
                authorities=random.choice(['Police', 'Security forces', 'Government']),
                number=random.choice(['hundreds of', 'thousands of', 'dozens of']),
                demands=random.choice(['political reform', 'economic justice', 'end to violence']),
                outcome=random.choice(['was dispersed peacefully', 'continued into the evening']),
                development=random.choice(['new strategy', 'ceasefire agreement', 'troop withdrawal']),
                agreement=random.choice(['peace deal', 'ceasefire', 'cooperation agreement']),
                operation=random.choice(['peacekeeping mission', 'security operation']),
                purpose=random.choice(['aims to restore stability', 'is intended to protect civilians']),
                significance=random.choice(['marks a significant shift', 'could impact regional stability']),
                trigger=random.choice(['controversial court ruling', 'disputed election results'])
            )
            
            generated.append(filled)
        
        return generated

    def sentence_mixing(self, examples: List[Dict], label: str, count: int = 5) -> List[str]:
        """Create new examples by mixing sentences from existing ones"""
        label_examples = [ex for ex in examples if ex['label'] == label]
        if len(label_examples) < 2:
            return []
        
        mixed = []
        for _ in range(count):
            # Take 2-3 random examples
            sample_examples = random.sample(label_examples, min(3, len(label_examples)))
            
            # Split into sentences
            all_sentences = []
            for ex in sample_examples:
                sentences = re.split(r'[.!?]+', ex['text'])
                all_sentences.extend([s.strip() for s in sentences if s.strip()])
            
            if len(all_sentences) >= 2:
                # Combine 2-3 sentences
                num_sentences = random.randint(2, min(3, len(all_sentences)))
                selected = random.sample(all_sentences, num_sentences)
                mixed_text = '. '.join(selected) + '.'
                mixed.append(mixed_text)
        
        return mixed

    def numerical_variation(self, text: str) -> List[str]:
        """Generate variations by changing numbers"""
        variations = []
        
        # Find numbers in text
        number_pattern = r'\b\d+\b'
        numbers = re.findall(number_pattern, text)
        
        for _ in range(2):
            new_text = text
            for num in numbers:
                # Vary the number by ±50%
                original = int(num)
                variation = random.randint(max(1, int(original * 0.5)), int(original * 1.5))
                new_text = new_text.replace(str(original), str(variation), 1)
            if new_text != text:
                variations.append(new_text)
        
        return variations

    def generate_synthetic_dataset(self, target_size: int = 5000, use_cohere: bool = True) -> List[Dict]:
        """Generate a massive synthetic dataset"""
        print(f"Generating synthetic dataset of {target_size} examples...")
        
        # Analyze original distribution
        original_distribution = self.analyze_data_distribution()
        total_original = len(self.original_data)
        
        # Calculate target distribution (aim for balanced dataset)
        labels = list(original_distribution.keys())
        target_per_label = target_size // len(labels)
        
        synthetic_dataset = []
        
        for label in labels:
            print(f"\nGenerating examples for {label}...")
            label_examples = [ex for ex in self.original_data if ex['label'] == label]
            generated_count = 0
            target_count = target_per_label
            
            # Track generation methods
            method_counts = defaultdict(int)
            
            # 1. Location substitution (20% of target)
            print("  - Location substitution...")
            for example in random.sample(label_examples, min(len(label_examples), target_count // 5)):
                variations = self.location_substitution(example['text'])
                for var in variations:
                    if generated_count < target_count:
                        synthetic_dataset.append({'text': var, 'label': label})
                        generated_count += 1
                        method_counts['location_sub'] += 1
            
            # 2. Entity substitution (15% of target)
            print("  - Entity substitution...")
            for example in random.sample(label_examples, min(len(label_examples), target_count // 6)):
                variations = self.entity_substitution(example['text'])
                for var in variations:
                    if generated_count < target_count:
                        synthetic_dataset.append({'text': var, 'label': label})
                        generated_count += 1
                        method_counts['entity_sub'] += 1
            
            # 3. Numerical variation (10% of target)
            print("  - Numerical variation...")
            for example in random.sample(label_examples, min(len(label_examples), target_count // 10)):
                variations = self.numerical_variation(example['text'])
                for var in variations:
                    if generated_count < target_count:
                        synthetic_dataset.append({'text': var, 'label': label})
                        generated_count += 1
                        method_counts['numerical_var'] += 1
            
            # 4. Template generation (25% of target)
            print("  - Template generation...")
            template_count = min(target_count // 4, target_count - generated_count)
            templates = self.template_generation(label, template_count)
            for template in templates:
                if generated_count < target_count:
                    synthetic_dataset.append({'text': template, 'label': label})
                    generated_count += 1
                    method_counts['template'] += 1
            
            # 5. Sentence mixing (15% of target)
            print("  - Sentence mixing...")
            mixed_count = min(target_count // 6, target_count - generated_count)
            mixed = self.sentence_mixing(label_examples, label, mixed_count)
            for mix in mixed:
                if generated_count < target_count:
                    synthetic_dataset.append({'text': mix, 'label': label})
                    generated_count += 1
                    method_counts['sentence_mix'] += 1
            
            # 6. Cohere paraphrasing (remaining ~15%)
            if use_cohere and generated_count < target_count:
                print("  - AI paraphrasing (this may take a while)...")
                remaining = target_count - generated_count
                paraphrase_needed = min(remaining, len(label_examples) * 3)
                
                for i, example in enumerate(random.sample(label_examples, min(len(label_examples), paraphrase_needed // 3))):
                    if generated_count >= target_count:
                        break
                    
                    paraphrases = self.paraphrase_with_cohere(example['text'], label)
                    for para in paraphrases:
                        if generated_count < target_count:
                            synthetic_dataset.append({'text': para, 'label': label})
                            generated_count += 1
                            method_counts['cohere_para'] += 1
                    
                    # Rate limiting for Cohere API
                    if i % 5 == 0:
                        time.sleep(1)
            
            print(f"  Generated {generated_count} examples for {label}")
            print(f"  Methods used: {dict(method_counts)}")
        
        print(f"\nTotal synthetic examples generated: {len(synthetic_dataset)}")
        return synthetic_dataset

    def save_synthetic_data(self, synthetic_data: List[Dict], filename: str = "massive_synthetic_training_data.jsonl"):
        """Save synthetic data to JSONL file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for example in synthetic_data:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        print(f"Saved {len(synthetic_data)} synthetic examples to {filename}")

    def create_combined_dataset(self, synthetic_data: List[Dict], output_file: str = "combined_training_data.jsonl"):
        """Combine original and synthetic data"""
        combined = self.original_data + synthetic_data
        random.shuffle(combined)  # Shuffle the combined dataset
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in combined:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Created combined dataset with {len(combined)} examples:")
        print(f"  Original: {len(self.original_data)}")
        print(f"  Synthetic: {len(synthetic_data)}")
        print(f"  Total: {len(combined)}")
        
        # Show final distribution
        final_distribution = Counter([item['label'] for item in combined])
        print("\nFinal dataset distribution:")
        for label, count in final_distribution.most_common():
            print(f"  {label}: {count}")

def main():
    """Main function to generate synthetic dataset"""
    generator = SyntheticDataGenerator()
    
    # Generate synthetic data
    # You can adjust target_size and use_cohere parameters
    synthetic_data = generator.generate_synthetic_dataset(
        target_size=5000,  # Generate 5000 synthetic examples
        use_cohere=False   # Skip Cohere paraphrasing to avoid API costs
    )
    
    # Save synthetic data
    generator.save_synthetic_data(synthetic_data)
    
    # Create combined dataset
    generator.create_combined_dataset(synthetic_data)
    
    print("\n🎉 Synthetic data generation complete!")
    print("Files created:")
    print("  - massive_synthetic_training_data.jsonl (synthetic only)")
    print("  - combined_training_data.jsonl (original + synthetic)")
    print("\nYou can now retrain your spaCy model with the combined dataset!")

if __name__ == "__main__":
    main() 