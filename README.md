# Modeling-Mental-Health-Trends-Using-Social-Media-Data
This project explores the intersection of social media analytics, user behavior modeling, and mental health assessment using a data-driven AI approach. Utilizing the MDDL dataset, both structured metadata and unstructured textual data are analyzed to develop a robust regression model for estimating depression intensity from Twitter user profiles, interactions, and content.


### **Feature Engineering & Classification Performance**  

By leveraging **feature engineering** on users’ tweet timelines; capturing **engagement metrics, interaction and social connectivity patterns, posting behaviors, and network & influence-based features**, a **significant accuracy boost** was achieved:  
- **🏆 69% → 96.95%** 🚀  
  - Initial accuracy (**69%**) using only profile metadata  
  - Final accuracy (**96.95%**) after integrating behavioral & temporal insights  

#### **Key Takeaways**  
✅ **High-value features** beyond raw metadata are critical for accurate classification  
✅ **Domain-driven models** improve predictive capability  
✅ **Structured experimentation** enhances model performance  

#### **Baseline Models Used**  
- **Gradient Boosting**  
- **Random Forest**  
- **Support Vector Machine (SVM)**  
- **Logistic Regression**  

#### **Next Steps: Advancing Depression Classification with NLP & Multimodal Fusion**  
🔍 The next phase focuses on leveraging state-of-the-art NLP models to classify depression from tweets. Given the domain-specific nature of social media text, BERTweet was selected, a transformer-based model pre-trained on Twitter data, for its superior performance in handling informal, noisy text. Fine-tuning BERTweet on our dataset will allow capturing nuanced linguistic markers indicative of depression.
Beyond text, we aim to enhance predictive performance by integrating multimodal learning, fusing text with structured metadata (e.g., user engagement metrics, posting behavior). This hybrid approach will enable a more holistic representation of user mental states, improving model robustness and generalizability. Future work will explore late fusion techniques to optimize the interaction between language-driven insights and metadata-derived patterns.
