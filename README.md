
# **Research Paper Analyzer Tool**  

## **Overview**  
The **Research Paper Analyzer Tool** is an AI-powered application that automates the process of analyzing and summarizing research papers. This tool extracts text from PDF documents, preprocesses the content, generates summaries, identifies key scientific terms, and provides research insights using advanced NLP models.  

The system is designed for **students, researchers, and academicians** who want to quickly understand research papers without manually reading every detail. It leverages **deep learning models** for summarization, scoring, and insight generation, providing a structured and interactive experience.  

---

## **Features**  
✅ **Automated PDF Text Extraction** – Uses **PyMuPDF** to extract research paper text accurately.  
✅ **Preprocessing & Cleaning** – Removes noise, tokenizes text, and corrects spelling errors using **NLTK & TextBlob**.  
✅ **AI-Powered Summarization** – Implements **allenai/led-base-16384** for long-document summarization.  
✅ **Refined Summary Generation** – Enhances summaries with **Llama-3.3-70B-Instruct API (Azure AI)**.  
✅ **Scientific Term Extraction** – Identifies key domain-specific terms using specialized AI APIs.  
✅ **Research Insights Generation** – Extracts key findings and important details from the paper.  
✅ **Paper Scoring System** – Evaluates the paper using **allenai/scibert_scivocab_uncased** for relevance and quality.  
✅ **Web Interface** – **Flask-based** interactive platform for uploading and analyzing research papers.  
✅ **Structured JSON Output** – Summarized and analyzed data is formatted neatly for easy access.  

---

## **How It Works**  
1️⃣ **Upload a Research Paper (PDF)** – Users upload research papers through the Flask web app.  
2️⃣ **Text Extraction & Preprocessing** – The system extracts and cleans the text for better analysis.  
3️⃣ **Summarization Process**  
   - **Initial Summary**: LED model (`allenai/led-base-16384`) generates a preliminary summary.  
   - **Refined Summary**: The initial summary is improved using the **Llama-3.3-70B-Instruct API**.  
4️⃣ **Scientific Term Extraction** – Key technical terms are extracted using **DeepSeek API**.  
5️⃣ **Research Insights Generation** – The tool highlights important concepts and findings.  
6️⃣ **Paper Scoring** – The system evaluates the research paper’s credibility and relevance.  
7️⃣ **Results Displayed on Web App** – Users can view the analysis in a structured format (JSON-based UI).  

---

## **Technology Stack**  
- **Backend**: Python, Flask  
- **Natural Language Processing**: PyMuPDF, NLTK, TextBlob, Transformers, PyTorch  
- **AI/ML Models**:  
  - **Summarization**: `allenai/led-base-16384`  
  - **Refined Summarization**: Azure AI API (`Llama-3.3-70B-Instruct`)  
  - **Scientific Term Extraction**: DeepSeek API  
  - **Paper Scoring**: `allenai/scibert_scivocab_uncased`  
- **Storage**: JSON (for storing extracted and analyzed data)  
- **Frontend**: HTML, CSS, JavaScript (for Flask web UI)  

---

## **Installation & Setup**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/research-paper-analyzer.git
cd research-paper-analyzer
```

### **2. Create a Virtual Environment (Optional but Recommended)**  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **4. Set Up API Keys (for Llama-3.3, DeepSeek, etc.)**  
Modify `config.py` and add your API keys:  
```python
GITHUB_AI_API_KEY = "your_github_ai_api_key"
AZURE_AI_API_KEY = "your_azure_ai_api_key"
DEEPSEEK_API_KEY = "your_deepseek_api_key"
```

### **5. Run the Application**  
```bash
python app.py
```
The web app will start, and you can access it at **http://127.0.0.1:5000/**.

---

## **Usage**  
1. Open the web app and **upload a research paper (PDF format)**.  
2. The system will **extract text, preprocess it, and generate a summary**.  
3. View **scientific terms, research insights, and the paper's score**.  
4. Download the structured analysis **in JSON format**.  

---

## **Future Enhancements**  
🔹 **Multilingual Support** – Support for non-English research papers.  
🔹 **Keyword-Based Searching** – Search key topics in the analyzed research papers.  
🔹 **Interactive Charts & Graphs** – Visual representation of key findings.  
🔹 **Integration with Zotero/Mendeley** – Directly import citations from reference managers.  

---

## **Contributing**  
Contributions are welcome! Feel free to **fork this repository, create a feature branch, and submit a pull request**.  

---

## **Contact**  
📧 Email: mvsr26032005@gmail.com 
🔗 GitHub: [github.com/mycode100](https://github.com/mycode100/)  
