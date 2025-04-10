import streamlit as st

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import numpy as np
import os


HOME = os.getcwd()

class config:
    BERT_PATH = os.path.join(HOME, "bertweet-base")
    MAX_LEN = 128         # Maximum number of tokens in the sentence
    MODEL_PATH = os.path.join(HOME, "bertweet-base/model.bin")

class BERTweet(nn.Module):
    def __init__(self):
        super(BERTweet, self).__init__()
        self.bert = AutoModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
    def forward(self, ids, mask, token_type_ids):
        """
        bert will return 2 dicts with keys
        'pooler_output' and 'last_hidden_state'
        we need pooler_output with size torch.Size([batch size, hidden_size])
        """
        bert_output = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        reguralized_output = self.bert_drop(bert_output["pooler_output"])
        output = self.out(reguralized_output)
        return output
    

TOKENIZER = AutoTokenizer.from_pretrained(config.BERT_PATH, do_lower_case=True)
model = BERTweet()
model.load_state_dict(torch.load(config.MODEL_PATH))
model.eval()  # Set the model to evaluation mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def predict_sentiment(sentence):
    """
    Function to predict sentiment of a given sentence using the fine-tuned BERTweet model.

    :param sentence: str, the sentence to predict
    :return: predicted sentiment (0 or 1)
    """
    # Tokenize the input sentence
    inputs = TOKENIZER.encode_plus(
        sentence,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors='pt'
    )

    # Move the inputs to the same device as the model
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)

    # Generate prediction
    with torch.no_grad():
        outputs = model(ids=input_ids, 
                        mask=attention_mask,
                        token_type_ids=token_type_ids
                        )
    # Get the prediction (regression)
    prediction = torch.sigmoid(outputs).cpu().numpy()
    return 'Depressed' if prediction >= 0.99 else 'Not Depressed', prediction


# App UI Setup
st.set_page_config(page_title="🧠 AI Mental Health Assistant", layout="centered")

st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌿 AI Mental Health Assistant")
st.caption("This tool **does not diagnose**, but highlights patterns statistically associated with depressive signals on social media.")

st.markdown("### 📝 Enter Tweet(s)")
mode = st.radio("How would you like to input your text?", ["Single Tweet", "Multiple Tweets"])

if mode == "Single Tweet":
    tweet = st.text_area("Write a tweet or short message:", placeholder="e.g. I feel tired all the time but nobody notices...", height=100)
    text_input = [tweet] if tweet else []
else:
    multi_tweets = st.text_area("Paste multiple tweets (one per line):", height=200, placeholder="Tweet 1\nTweet 2\nTweet 3...")
    text_input = [line.strip() for line in multi_tweets.splitlines() if line.strip()]

if st.button("🔍 Analyze"):
    if not text_input:
        st.warning("Please enter at least one tweet.")
    else:
        for i, t in enumerate(text_input):
            with st.container():
                result, prediction = predict_sentiment(t)
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Tweet #{i+1}:** {t}")
                    st.markdown(f"🧠 **Prediction**: `{result}`")
                    st.progress(float(prediction), text="Depression probability")
                with col2:
                    st.metric("Depression", f"{float(prediction)*100:.1f}%", delta=None)
                    emoji = "😔" if result == "Depressed" else "😊"
                    st.markdown(f"<div style='font-size: 40px;'>{emoji}</div>", unsafe_allow_html=True)
        st.success("Analysis complete. Remember: These are not diagnoses, just statistical signals.")
        st.info("💡 Consider consulting a mental health professional for any concerns.")

st.markdown("---")
st.markdown("🧠 *Built with compassion, by AI researchers.*")
st.markdown("👩‍🔬 [GitHub](https://github.com/Kh0uloud/Modeling-Mental-Health-Trends-Using-Social-Media-Data) | 🌐 [Contact](LinkedIn: linkedin.com/in/khouloudismail/)")
