# Next Word Predictor 📖

An undergraduate-level Machine Learning project that predicts the next word in an English sentence using a Neural Language Model built with **PyTorch** and an interactive **Streamlit** web app.

---

## 📌 Project Overview

This project implements next-word prediction—the same technology behind mobile keyboard autocomplete and predictive text.

Given a sequence of words, the neural network analyzes the previous **5 words** (context window) and calculates the probability of what the next word should be from a vocabulary of **19,764 English words**.

The model was trained on Leo Tolstoy's classic novel, ***War and Peace***.

---

## 🎯 How It Works (Simple Concept)

1. **Input Words**: You type a sentence (e.g., `"the prince was silent and"`).
2. **Word Lookup**: Each word is converted into a list of numbers (word embedding) that represents its meaning.
3. **Neural Network**: The model processes the 5 previous words through a hidden layer of neurons to understand the sentence structure.
4. **Prediction**: The model calculates percentage chances for every word in the book and picks the most likely word (or top candidates) to continue the sentence.

---

## 🧪 Models Trained

To compare how different neural network activation functions perform, three variations were trained:

| Model Name | Activation Function | Hidden Neurons | Embedding Size | Checkpoint File |
| :--- | :---: | :---: | :---: | :--- |
| **Model 1** | **ReLU** | 1024 | 64 | `checkpoints/war_and_peace/model1_task1.pt` |
| **Model 2** | **Tanh** | 128 | 32 | `checkpoints/war_and_peace/model2_task1.pt` |
| **Model 3** | **Sigmoid** | 128 | 64 | `checkpoints/war_and_peace/model3_task1.pt` |

- **ReLU (Model 1)**: The largest model, producing the most fluent completions.
- **Tanh (Model 2)**: A lighter, faster model using hyperbolic tangent.
- **Sigmoid (Model 3)**: A classic baseline model using sigmoid activation.

---

## 💻 Interactive Web Application

The project includes an interactive web dashboard built with Streamlit:

- **Type any phrase** or click on preset examples.
- **Choose between trained models** (ReLU, Tanh, or Sigmoid).
- **Adjust prediction settings** like how many words to predict and randomness (temperature).
- **See top 5 candidate words** along with their percentage probabilities in a real-time bar chart.

---

## 🚀 How to Run

### 1. Install Dependencies
Make sure you have Python installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Launch the Web App
Run the Streamlit application:
```bash
streamlit run app.py
```
This opens the web app in your browser at `http://localhost:8501`.

### 3. (Optional) Run via Command Line
You can also generate predictions directly from your terminal:
```bash
python scripts/generate.py --prompt "the prince was silent and" --tokens 5
```

### 4. (Optional) Run Automated Tests
Verify that all model modules and tokenizers are functioning properly:
```bash
pytest
```

---

## 📂 Project Structure

```
NextTokenPredictor/
├── app.py                      # Main entry point for Streamlit web app
├── web/
│   └── app.py                  # Streamlit dashboard interface
├── checkpoints/
│   └── war_and_peace/          # Trained PyTorch model files (.pt)
├── data/
│   └── vocabularies/
│       └── war_and_peace/      # Word-to-index mappings (.pkl)
├── notebooks/
│   └── 01_natural_language_war_and_peace.ipynb  # Training & exploration notebook
├── src/
│   └── next_token_predictor/   # Core Python package
│       ├── models/             # PyTorch Neural Network architecture
│       ├── inference/          # Text generator and model loader
│       └── tokenization/       # Word tokenizer
├── scripts/
│   ├── generate.py             # Command-line prediction script
│   └── evaluate.py             # Model evaluation script
├── tests/                      # Unit tests (pytest)
└── requirements.txt            # Project dependencies
```

---

## 📝 Example Output

- **Input Prompt**: `"the prince was silent and"`
- **Predicted Next Words**: `"looked indifferent at the russian"`
- **Top Candidates for First Word**:
  - `looked` (42.5%)
  - `said` (18.2%)
  - `smiled` (12.1%)
  - `stood` (8.4%)
  - `went` (5.3%)

---

## 🛠️ Built With

- **Python 3**
- **PyTorch** (Deep Learning framework)
- **Streamlit** (Web application UI)
- **Pandas** (Data manipulation)
- **Pytest** (Automated unit testing)
