#!/usr/bin/env python3
"""
Export Database Analyses as Training Data

This script exports your existing Cohere-analyzed events from the database
as high-quality training data for improving your spaCy model.
"""

import sqlite3
import json
import os
from datetime import datetime

def export_database_training_data():
    """Export analyzed events from database as training data."""
    
    # Database configuration
    DATABASE_NAME = 'conflict_events.db'
    
    if not os.path.exists(DATABASE_NAME):
        print(f"❌ Database file not found: {DATABASE_NAME}")
        print("Make sure you're running this from the correct directory.")
        return
    
    # Connect to database
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        # Query to get all events with summaries and event types
        query = """
        SELECT summary, event_type, confidence, country, key_actors, fatalities 
        FROM events 
        WHERE summary IS NOT NULL 
        AND event_type IS NOT NULL 
        AND summary != ''
        AND event_type != ''
        ORDER BY confidence DESC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        if not results:
            print("❌ No analyzed events found in database.")
            return
        
        print(f"Found {len(results)} analyzed events in database")
        
        # Convert to training data format
        training_data = []
        quality_stats = {"high_confidence": 0, "medium_confidence": 0, "low_confidence": 0}
        
        for summary, event_type, confidence, country, key_actors, fatalities in results:
            # Use summary as training text (it's concise and focused)
            training_example = {
                "text": summary.strip(),
                "label": event_type
            }
            training_data.append(training_example)
            
            # Track confidence levels
            if confidence > 0.8:
                quality_stats["high_confidence"] += 1
            elif confidence > 0.6:
                quality_stats["medium_confidence"] += 1
            else:
                quality_stats["low_confidence"] += 1
        
        # Save as JSONL file
        output_file = "database_training_data.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in training_data:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        
        # Statistics
        print(f"\n✅ Exported {len(training_data)} training examples to {output_file}")
        print("\nQuality distribution:")
        print(f"  High confidence (>80%): {quality_stats['high_confidence']}")
        print(f"  Medium confidence (60-80%): {quality_stats['medium_confidence']}")
        print(f"  Low confidence (<60%): {quality_stats['low_confidence']}")
        
        # Event type distribution
        from collections import Counter
        event_counts = Counter([item['label'] for item in training_data])
        print("\nEvent type distribution:")
        for event_type, count in event_counts.most_common():
            print(f"  {event_type}: {count}")
        
        # Also create extended training examples using additional context
        extended_training_data = []
        
        print("\nCreating extended training examples...")
        for summary, event_type, confidence, country, key_actors, fatalities in results:
            # Create extended text with more context
            extended_text = summary.strip()
            
            # Add country context if available
            if country and country.strip():
                extended_text += f" This event occurred in {country.strip()}."
            
            # Add actor context if available
            if key_actors and key_actors.strip():
                extended_text += f" Key actors involved: {key_actors.strip()}."
            
            # Add fatality context if available and not "Unknown"
            if fatalities and str(fatalities).strip() and str(fatalities).strip().lower() != "unknown":
                extended_text += f" Casualties reported: {fatalities}."
            
            extended_example = {
                "text": extended_text,
                "label": event_type
            }
            extended_training_data.append(extended_example)
        
        # Save extended version
        extended_output_file = "database_extended_training_data.jsonl"
        with open(extended_output_file, 'w', encoding='utf-8') as f:
            for example in extended_training_data:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ Also created extended examples in {extended_output_file}")
        
        # Create combined file with original ACLED data
        original_file = "acled_long_paragraph_training_data.jsonl"
        if os.path.exists(original_file):
            print(f"\nCombining with original {original_file}...")
            
            # Load original data
            original_data = []
            with open(original_file, 'r', encoding='utf-8') as f:
                for line in f:
                    original_data.append(json.loads(line.strip()))
            
            # Combine all data
            all_training_data = original_data + training_data + extended_training_data
            
            # Remove duplicates (by text)
            seen_texts = set()
            unique_data = []
            for item in all_training_data:
                if item['text'] not in seen_texts:
                    unique_data.append(item)
                    seen_texts.add(item['text'])
            
            # Save combined file
            combined_output_file = "enhanced_training_data.jsonl"
            with open(combined_output_file, 'w', encoding='utf-8') as f:
                for example in unique_data:
                    json.dump(example, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"✅ Created enhanced dataset: {combined_output_file}")
            print(f"   Original ACLED data: {len(original_data)}")
            print(f"   Database summaries: {len(training_data)}")
            print(f"   Extended examples: {len(extended_training_data)}")
            print(f"   Total unique examples: {len(unique_data)}")
            
            # Final distribution
            final_counts = Counter([item['label'] for item in unique_data])
            print("\nFinal enhanced dataset distribution:")
            for event_type, count in final_counts.most_common():
                print(f"  {event_type}: {count}")
        
        print(f"\n🎉 Database export complete!")
        print("Files created:")
        print(f"  - {output_file} (basic summaries)")
        print(f"  - {extended_output_file} (extended context)")
        if os.path.exists(original_file):
            print(f"  - {combined_output_file} (combined with ACLED data)")
        
        print("\nNext steps:")
        print("1. Run the synthetic data generator on the enhanced dataset")
        print("2. Train your improved spaCy model")
        print("3. Compare performance with Cohere!")
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    export_database_training_data() 