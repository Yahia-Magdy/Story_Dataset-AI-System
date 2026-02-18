# Story Dataset AI System

This project implements an AI-powered chatbot that can access and retrieve information from a collection of 1,000 stories. Users can ask questions about any aspect of the stories, including plot details, characters, settings, themes, events, dialogue, or descriptions. The chatbot provides accurate, relevant answers directly based on the story content.  

Additionally, the system includes a genre classification component that can categorize stories into 100 different genres, enabling both intelligent information retrieval and content organization within a single AI framework.
## Getting Started

Follow these steps to set up and run the system using **conda**.

## Setup & Run Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
```

### 2. Navigate to the Project Directory
```bash
cd src
```

### 3. Create a Conda Environment
```bash
conda create -n myenv python=3.12 -y
```

### 4. Activate the Conda Environment
```bash
conda activate myenv
```

### 5. Install Required Packages
```bash
pip install -r requirements.txt
```

### 6. Ingest Qdrant Vector Database
```bash
python -m RAG.controllers.ingest_qdrant
```

### 7. Run the Application
```bash
streamlit run main.py
```
# Project Structure

```
Story Retrieval and Classification System/
│
├── src/
│   ├── RAG/
│   ├── Genres_Classification/
│   ├── main.py
│   ├── deployment.md
│   ├── .gitignore
│   └── .env.examples
│
├── LICENSE
└── README.md
```
## Additional Documentation

For in-depth explanations of each system component, see the internal READMEs:

- **RAG Module** – Detailed description of the retrieval-augmented generation system, metadata extraction, vector search, and LLM prompt construction.  
  Location: `src/RAG/README.md`

- **Genres Classification Module** – In-depth overview of story genre classification, preprocessing, data augmentation, training methodology, and evaluation.  
  Location: `src/Genres_Classification/README.md`

- **Deployment** – CPU deployment instructions, optimizations, latency benchmarking, and production considerations.  
  Location: `src/deployment.md`


