def text_classifier_page():
    st.title("Text Classifier")
    st.markdown("**Quickly classify text content by conflict event type using AI**")
    
    # Import spaCy classifier functions
    try:
        from spacy_classifier import classify_with_spacy, is_spacy_available, get_spacy_classifier
        spacy_imported = True
    except ImportError:
        spacy_imported = False
    
    # Create classifier selection
    st.markdown("### Classification Method")
    
    # Check what classifiers are available
    cohere_available = True  # Assuming Cohere is configured
    spacy_available = spacy_imported and is_spacy_available() if spacy_imported else False
    
    classifier_options = []
    if cohere_available:
        classifier_options.append("Cohere API")
    if spacy_available:
        classifier_options.append("spaCy (Local)")
    
    if not classifier_options:
        st.error("No classifiers available. Please configure Cohere API or train a spaCy model.")
        return
    
    # Let user choose classifier
    if len(classifier_options) > 1:
        selected_classifier = st.radio(
            "Choose classification method:",
            classifier_options,
            horizontal=True,
            help="Cohere API provides cloud-based classification, spaCy runs locally"
        )
    else:
        selected_classifier = classifier_options[0]
        st.info(f"Using {selected_classifier} classifier")
    
    # Show classifier info
    if selected_classifier == "spaCy (Local)" and spacy_available:
        st.success("✓ spaCy model loaded and ready")
    elif selected_classifier == "Cohere API":
        st.info("ℹ️ Using Cohere cloud API")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Input Text")
        
        # Text input methods
        input_method = st.radio(
            "Choose input method:",
            ["Type/Paste Text", "Upload Text File"],
            horizontal=True
        )
        
        user_text = ""
        
        if input_method == "Type/Paste Text":
            user_text = st.text_area(
                "Enter text to classify:",
                height=300,
                placeholder="Paste news article text, social media posts, reports, or any text describing conflict events here..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file",
                type=['txt', 'md'],
                help="Upload a .txt or .md file containing the text you want to classify"
            )
            if uploaded_file is not None:
                user_text = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", value=user_text, height=200, disabled=True)
        
        # Character count
        if user_text:
            char_count = len(user_text)
            st.caption(f"Character count: {char_count:,}")
            if char_count > 10000:
                st.warning("⚠️ Text is quite long. Consider using shorter excerpts for better accuracy.")
    
    with col2:
        st.markdown("### Classification")
        
        # Classification button
        if st.button("🔍 Classify Text", type="primary", disabled=not user_text.strip()):
            if user_text.strip():
                with st.spinner("Analyzing text..."):
                    try:
                        # Use the selected classifier
                        if selected_classifier == "spaCy (Local)" and spacy_available:
                            classification = classify_with_spacy(user_text)
                            if classification is None:
                                st.error("spaCy classification failed. Please try Cohere API.")
                                return
                        else:
                            # Use Cohere API (existing function)
                            classification = classify_event(user_text)
                        
                        # Display results
                        st.success("✓ Classification Complete")
                        
                        # Event type with confidence
                        confidence_color = "green" if classification.confidence > 0.7 else "orange" if classification.confidence > 0.5 else "red"
                        
                        st.markdown("#### Results")
                        st.markdown(f"""
                        **Event Type:** `{classification.prediction}`
                        
                        **Confidence:** <span style="color: {confidence_color}; font-weight: bold;">{classification.confidence:.1%}</span>
                        
                        **Classifier:** `{selected_classifier}`
                        """, unsafe_allow_html=True)
                        
                        # Confidence interpretation
                        if classification.confidence > 0.8:
                            st.info("🎯 **High confidence** - Very likely classification")
                        elif classification.confidence > 0.6:
                            st.info("📊 **Medium confidence** - Reasonably likely classification")
                        else:
                            st.warning("⚠️ **Low confidence** - Classification uncertain")
                        
                        # Show raw confidence score
                        st.progress(classification.confidence)
                        
                        # Show detailed predictions for spaCy
                        if selected_classifier == "spaCy (Local)" and spacy_available:
                            classifier = get_spacy_classifier()
                            all_predictions = classifier.get_all_predictions(user_text)
                            if all_predictions:
                                with st.expander("📊 View All Predictions"):
                                    st.markdown("**All category scores:**")
                                    for label, score in sorted(all_predictions.items(), key=lambda x: x[1], reverse=True):
                                        st.markdown(f"- **{label}**: {score:.1%}")
                        
                    except Exception as e:
                        st.error(f"Classification failed: {str(e)}")
                        logger.error(f"Text classification error: {e}")
            else:
                st.warning("Please enter some text to classify.")
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Information section
    with st.expander("ℹ️ About Text Classification"):
        st.markdown("""
        ### How It Works
        
        This tool provides **two classification options**:
        
        **Cohere API (Cloud):**
        - Uses the same AI model that powers the full article analysis
        - Requires internet connection and API credits
        - Generally very accurate and handles diverse text styles
        
        **spaCy (Local):**
        - Runs entirely on your machine (no internet required)
        - Uses a custom-trained model on conflict event data
        - Faster response time and no API costs
        - Available when the local model is trained and loaded
        
        **What it classifies:**
        - Armed conflicts and battles
        - Explosions and remote violence
        - Violence against civilians
        - Riots and civil unrest
        - Protests and demonstrations
        - Strategic developments
        
        **Tips for better results:**
        - Use clear, descriptive text about the event
        - Include context about what happened
        - Shorter, focused text often works better than very long passages
        - English text typically produces the most accurate results
        
        **Note:** This is a classification-only tool. For full analysis including location, fatalities, and other details, use the main **Data Entry** page.
        """)
    
    # Sample texts for testing
    with st.expander("📝 Try Sample Texts"):
        st.markdown("**Click any sample to test the classifier:**")
        
        samples = [
            {
                "title": "Armed Conflict",
                "text": "Government forces clashed with rebel groups in the northern region, with reports of heavy fighting and artillery exchanges. Military officials confirmed the engagement lasted several hours."
            },
            {
                "title": "Protest/Civil Unrest", 
                "text": "Thousands of demonstrators gathered in the capital city square to protest against government policies. Police deployed tear gas as some protesters became violent and began throwing stones."
            },
            {
                "title": "Terrorist Attack",
                "text": "An explosion occurred at a busy marketplace during rush hour. Authorities suspect it was a coordinated attack targeting civilians, with multiple casualties reported."
            }
        ]
        
        for i, sample in enumerate(samples):
            if st.button(f"📋 Load: {sample['title']}", key=f"sample_{i}"):
                st.session_state[f"sample_text_{i}"] = sample['text']
                st.rerun()
        
        # Display loaded sample
        for i, sample in enumerate(samples):
            if f"sample_text_{i}" in st.session_state:
                st.text_area(
                    f"Sample: {sample['title']}", 
                    value=st.session_state[f"sample_text_{i}"],
                    height=100,
                    key=f"display_sample_{i}"
                )
                if st.button(f"🔍 Classify This Sample", key=f"classify_sample_{i}"):
                    # Move sample to main text area
                    st.session_state.main_text = st.session_state[f"sample_text_{i}"]
                    # Clear the sample
                    del st.session_state[f"sample_text_{i}"]
                    st.rerun() 