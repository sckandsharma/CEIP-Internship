# Document Question Answering System using RAG
A Retrieval-Augmented Generation (RAG) pipeline that answers questions from custom PDF documents. Built as part of the Celebal Technologies Summer Internship - Week 7 Assignment.
## What is RAG?
Large Language Models (LLMs) are powerful but they only know what they were trained on. They cannot answer questions about your private documents, notes, or research papers.
RAG solves this by combining two steps:
1. **Retrieval** -- Find the most relevant pieces of text from your document
2. **Generation** -- Feed those pieces to an LLM and generate a grounded, accurate answer
This means the LLM does not hallucinate or guess. It answers strictly based on what is written in your document.
## How It Works
```
PDF Document
     |
     v
[Text Extraction] -- PyPDFLoader loads all pages
     |
     v
[Chunking] -- RecursiveCharacterTextSplitter splits into 500-char chunks
     |
     v
[Embedding] -- HuggingFace all-MiniLM-L6-v2 converts chunks to 384-dim vectors
     |
     v
[Vector Store] -- FAISS stores all vectors for fast similarity search
     |
     v
[User Query] -- User asks a question
     |
     v
[Retrieval] -- FAISS finds top 3 most similar chunks
     |
     v
[Generation] -- Groq LLama 3.3 70B generates answer using retrieved context
     |
     v
[Answer] -- Grounded, accurate response
```
## Tech Stack
|
 Component 
|
 Tool Used 
|
|
---
|
---
|
|
 Document Loader 
|
 LangChain PyPDFLoader 
|
|
 Text Splitter 
|
 RecursiveCharacterTextSplitter 
|
|
 Embedding Model 
|
 sentence-transformers/all-MiniLM-L6-v2 
|
|
 Vector Database 
|
 FAISS (Facebook AI Similarity Search) 
|
|
 LLM 
|
 LLama 3.3 70B via Groq API 
|
|
 Framework 
|
 LangChain 
|
|
 Language 
|
 Python 3.10 
|
## Project Structure
```
.
├── data/
│   └── AMSS_2026_MasterPrep.pdf    # Source document (26 pages)
├── RAG_Document_QA.ipynb            # Main notebook with outputs
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```
## Results
The notebook was run on a 26-page PDF document. Here are the key metrics:
|
 Metric 
|
 Value 
|
|
---
|
---
|
|
 Pages loaded 
|
 26 
|
|
 Chunks created (chunk_size=500) 
|
 135 
|
|
 Embedding dimensions 
|
 384 
|
|
 Retrieval method 
|
 Cosine Similarity 
|
|
 Top-k chunks retrieved 
|
 3 
|
|
 LLM temperature 
|
 0.4 
|
All outputs including retrieved chunks and generated answers are saved inside the notebook and are visible directly on GitHub without needing to run anything.
## Experiments Conducted
### Experiment 1 -- Chunk Size Comparison
|
 Chunk Size 
|
 Overlap 
|
 Chunks Created 
|
|
---
|
---
|
---
|
|
 200 
|
 20 
|
 359 
|
|
 500 (default) 
|
 20 
|
 135 
|
|
 1000 
|
 50 
|
 70 
|
Observation: Smaller chunks produce finer retrieval but may lose broader context. Larger chunks retain more context but retrieval becomes less precise.
### Experiment 2 -- Retrieval k Value
|
 k Value 
|
 Chunks Retrieved 
|
 Effect 
|
|
---
|
---
|
---
|
|
 3 (default) 
|
 3 
|
 Focused, less noise 
|
|
 5 
|
 5 
|
 More context, slightly more noise 
|
### Experiment 3 -- Temperature Comparison
|
 Temperature 
|
 Style 
|
 Observation 
|
|
---
|
---
|
---
|
|
 0.1 
|
 Factual 
|
 Consistent, focused answers 
|
|
 0.4 (default) 
|
 Balanced 
|
 Good mix of accuracy and readability 
|
|
 0.9 
|
 Creative 
|
 More varied phrasing, same core information 
|
## Key Learnings
- RAG enables LLMs to answer questions about documents they have never seen during training
- Chunk size directly impacts retrieval quality -- there is a tradeoff between precision and context
- FAISS provides fast local vector similarity search without needing a cloud database
- Temperature controls the creativity of the generated answer -- lower is better for factual QA
- The retrieval step is what makes RAG reliable -- the LLM only sees relevant context, not the entire document
## How to Run Locally
Clone the repository
```bash
git clone <repo-url>
cd Week_7_RAG
```
Install dependencies
```bash
pip install -r requirements.txt
```
Create a .env file with your Groq API key
```
GROQ_API_KEY = "your_key_here"
```
Get a free key from https://console.groq.com
Open the notebook and run all cells
```bash
jupyter notebook RAG_Document_QA.ipynb
```
## References
- LangChain Documentation -- https://python.langchain.com
- FAISS -- https://github.com/facebookresearch/faiss
- HuggingFace Sentence Transformers -- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Groq API -- https://console.groq.com
Built by Sckand Sharma | Celebal Technologies Summer Internship 2026 | Week 7 Assignment
