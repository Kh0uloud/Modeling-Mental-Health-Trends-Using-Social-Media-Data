# Modeling-Mental-Health-Trends-Using-Social-Media-Data
This project explores the intersection of social media analytics, user behavior modeling, and mental health assessment using a data-driven AI approach. Utilizing the MDDL dataset, both structured metadata and unstructured textual data are analyzed to develop a robust classification model for predicting depression presence from Twitter user profiles, interactions, and content.

## **Modeling Phases**  
The project follows a progressive approach, incorporating different methodologies: 

### **1️⃣ Baseline Classification Modeling: Structured Metadata**

#### **Feature Engineering & Classification Performance**  
By leveraging **feature engineering** on users’ tweet timelines; capturing **engagement metrics, interaction and social connectivity patterns, posting behaviors, and network & influence-based features**, a **significant accuracy boost** was achieved:  
- **🏆 69% → 96.95%** 🚀  
  - Initial accuracy (**69%**) using only profile metadata  
  - Final accuracy (**95.18%**) after integrating behavioral & temporal insights  
#### **Key Takeaways**  
✅ **High-value features** beyond raw metadata are critical for accurate classification  
✅ **Domain-driven models** improve predictive capability  
✅ **Structured experimentation** enhances model performance  
#### **Baseline Models Used**  
- **Gradient Boosting**  
- **Random Forest**  
- **Support Vector Machine (SVM)**  
- **Logistic Regression**   <br> <br>


### **2️⃣ NLP Study: Language Model Selection & Justification**
For this task, we leverage **BERTweet** [(vinai/bertweet-base)](https://github.com/VinAIResearch/BERTweet), a bidirectional transformer-based language model, pretrained on a massive corpus of social media text. Unlike **unidirectional models like GPT**, which process text from left to right, BERT-style models attend to both preceding and succeeding tokens simultaneously, enabling richer contextual understanding—a crucial advantage for analyzing nuanced language patterns in tweets. <br>
By fine-tuning BERTweet on our dataset using a PyTorch training loop with a max token length of 128, we adapt its pretrained linguistic representations to our binary classification task (depressed vs. non-depressed users). This approach significantly enhances the model’s ability to capture context-dependent linguistic markers of depression, outperforming traditional NLP techniques.<br>
Our fine-tuned model achieved **83.95% validation accuracy** and **83.42% test accuracy**. <br> <br>


### **3️⃣ Next Steps: Advancing Depression Classification with NLP & Multimodal Fusion**  
🔍 Future work will explore multimodal learning by integrating structured metadata (e.g., user engagement metrics, posting behavior) with textual features to further improve predictive performance. <br> <br>


## **Dataset**  
The dataset used in this project can be downloaded from the following link: **[https://github.com/sunlightsgy/MDDL]**.  

The project is structured into different phases, each addressing a key aspect of data-driven depression classification.  

## **Data Exploration & Preprocessing**  
The **MDDL_Dataset notebook** contains the entire workflow, including:  
- **Data Loading & Cleaning**: Handling missing values, feature engineering, and data transformation.  
- **Exploratory Data Analysis (EDA)**: Understanding data distribution, patterns, and feature relationships through **statistical analysis & visualizations**.  
- **Feature Engineering**: Creating new informative features from metadata and textual content.  
For each section, **cleaned notebooks** are provided separately for clarity and structured insights.  
