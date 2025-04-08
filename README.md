# 🌿 Social Media Signals & Mental Health: Modeling Depression Trends with AI

This project explores how language and behavior on Twitter can reveal potential indicators of mental health struggles. We use a data-driven, responsible AI approach to detect linguistic and behavioral patterns **associated** with depression, without making medical claims or diagnoses.
> ⚠️ **Note:** This tool does not diagnose depression. It simply highlights digital markers statistically correlated with depressive behavior in social media data.

---

### 📊 Dataset Overview: MDDL (Multimodal Depression Detection Dataset)
We focus on two major components of the MDDL dataset:
#### ᵀ Users Folder
- Contains user metadata for **6,022 labeled Twitter accounts** (both depressed and non-depressed).
- Includes: `followers_count`, `friends_count`, `verified`, `location`, `statuses_count`, etc.
#### ᵀ Timeline Folder
- Up to **3,000 tweets per user**, with timestamps and interaction metrics (likes, retweets, quotes, replies).
- Enables **longitudinal analysis** of user behavior rather than snapshot-based judgment.
- After preprocessing (e.g., keeping only English tweets for compatibility with BERTweet), the dataset includes **1,348,915 tweets.**


---

### 🔎 Phase 1: Structured Profile-Based Modeling, and Behavioral Timeline Feature Engineering
**Notebook**: [(Depression_Prediction_From_Tweeter_Data.ipynb)](https://github.com/Kh0uloud/Modeling-Mental-Health-Trends-Using-Social-Media-Data/blob/main/Depression_Prediction_From_Tweeter_Data.ipynb)
#### Approach
- Used users profile structured metadata (e.g., number of friends, followers, verification status, etc.).
- Tested standard classifiers (Logistic Regression, Random Forest, SVM, Gradient Boosting).
#### Results
- Baseline Accuracy: **69%** (profile-only features)
- Conclusion: Metadata alone provides limited insight.
#### Optimization 
Aggregated information over time to capture behavioral trends:
- Tweet frequency
- Retweet & favorite patterns
- Quote & reply behavior
- Day/Night posting distribution
- Interactions with other depressed users
#### Results
- Achieved **95.18% accuracy** with GBM <br> <br>



### 🔎 Phase 2: Tweet-Based Classification with BERTweet
**Notebook**: [(Depression_Prediction_From_Language_Data.ipynb)](https://github.com/Kh0uloud/Modeling-Mental-Health-Trends-Using-Social-Media-Data/blob/main/Depression_Prediction_From_Language_Data.ipynb)
#### Motivation
While user timelines offer rich information, full access is often limited by privacy concerns. In contrast, tweets are more readily accessible—either via public scraping or direct user input.
#### Approach
- Finetuned [(vinai/bertweet-base)](https://github.com/VinAIResearch/BERTweet) on individual user tweets.
#### Results
- **Validation Accuracy**: 83.95%
- **Test Accuracy**: 83.42%
> Enables a lighter version of depression detection using **only text data** (no profile scraping), making it more deployable and privacy-conscious. <br> <br>



### 🔗 Phase 3: Multimodal Fusion (Structured + Textual Embeddings)
#### Approach
- Extracted **leaf embeddings** from the Gradient Boosting model (structured data).
- Aggregated tweets into samples (max 128 tokens), and extracted **sentence embeddings** from finetuned BERTweet model.
- Concatenated both and passed to several MLP classifiers.
#### Challenge
- Despite high feature richness, results were underwhelming due to:
  - **High-dimensional feature space**
  - **Limited sample size (6022)** <br> <br>



### 🧪 Phase 4: Public Demo & Twitter Pipeline
#### Features:
- 🔍 **Input**: Paste one or more tweets
- 📢 **Output**: Depression prediction + confidence score
- 🖥️ **Frameworks**: Built using **Python**, hosted on **Streamlit**
#### Future Work; Optional Add-on
- 🧠 Input a **Twitter handle** (with consent)
- 🛠️ Backend scrapes timeline, computes structured + behavioral features
- 💡 Generates full explained report: timeline trends, engagement stats, depression patterns trends with explanations 

---

### 📊 Results Summary

| Model Type                 | Description                              | Accuracy  |
|---------------------------|------------------------------------------|-----------|
| Profile-only Classifier   | Structured metadata                      | 69%       |
| Behavioral Features       | Aggregated timeline activity             | 95.18%    |
| Tweet-based (BERTweet)    | Raw tweet text only                      | 83.42%    |
| Multimodal Fusion         | Structured + Text embeddings             | ❌ Not viable (yet) |

---

### 🔹 Final Words: Ethical Framing
We take a **careful, non-diagnostic approach** to mental health modeling. This tool should not be used for clinical purposes but can serve as:
- A research experiment in mental health signal detection
- A prototype to showcase responsible, explainable AI
- A starting point for digital well-being assessments
If this tool encourages even one person to reflect on their digital behavior, we've made progress.

---


### 📃 Dataset Source
- **MDDL GitHub**: [MDDL](https://github.com/sunlightsgy/MDDL)

---

### 🌐 Contact & Collaboration
Interested in deploying, improving, or applying this tool in real life?
Let’s collaborate on impactful, ethical AI. Reach out via [LinkedIn](https://www.linkedin.com/in/khouloudismail/).

