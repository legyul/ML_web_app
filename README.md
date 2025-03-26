# 🤖 AutoML Web App with Clustering, Classification & AI Q&A (RAG + LoRA)
  
This is a powerful and easy-to-use **AI web application** that lets users:  
- Upload a dataset  
- Choose between **Classification** or **Clustering**  
- Train models automatically  
- Download reports and predictions  
- Ask questions about the data using **AI-powered Q&A (RAG + LoRA)**  
  
Built with custom machine learning models, optimized for deployment on AWS, and a clean UI.  
  
----
  
## 🔍 Key Features  
### 📊 1. Clustering  
- Automatically finds the best number of clusters (Elbow & Silhouette methods)  
- Supports **K-Means** and **Agglomerative Clustering**  
- PCA visualization for easy understanding  
- Creates and downloads a PDF report + CSV results  
  
### 🧠 2. Classification  
- Users can choose from four custom-built models:  
    - Naive Bayes  
    - Decision Tree  
    - Random Forest  
    - Logistic Regression (with automatic hyperparameter tuning when user selects best model)  
- Or, let the system automatically select the best model based on ROC-AUC  
- After training:  
    - Download the trained model  
    - View and download log records  
    - Download a full training report  
    - Ask the AI about prediction results via the Q&A system (CSV does not include direct predictions)
  
### 💬 3. AI-Powered Q&A (RAG + LoRA)
- Ask questions about your uploaded dataset and model results  
- Uses **Retrieval-Augmented Generation (RAG)** with **TinyLlama**  
- Fine-tuned on your data using **LoRA (Low-Rank Adaptation)**  
- Fast retrieval with **ChromaDB**  
  
---
  
## ⚙️ Tech Stack
|Area|Tools & Technologies|
|:---:|:------------------|
|Backend|Flask, PyTorch, LangChain, Transformers|
|Frontend|HTML, JavaScript|
|ML Models|Custom Naive Bayes, Decision Tree, etc.|
|LLM|TinyLlama + LoRA|
|Vector Store|ChromaDB|
|Deployment|Docker, AWS EC2, S3|
|CI/CD|Crontab (checks Github for updates hourly)|
  
---
  
## 🗂️ Project Structure
<pre>
<code>
project/
├── src/
│   ├── app.py                   # Main Flask application
│   ├── lora_train.py            # LoRA fine-tuning on TinyLlama
│   ├── rag_index.py             # Embedding & indexing for RAG
│   ├── rag_qa.py                # RAG-based QA interface
│   ├── utils/                   # Helper modules and shared functions
│   └── models/
│       ├── classification_main.py   # Full classification workflow
│       ├── classification_model.py  # All classification model implementations
│       ├── clustering_main.py       # Full clustering workflow
│       └── clustering_model.py      # All clustering model implementations
├── templates/
│   └── index.html               # Frontend UI (Form, Chat interface)
├── static/                      # CSS and JavaScript files
├── models/                      # Trained models (saved to S3)
├── logs/                        # Log files (viewable/downloadable)
├── requirements.txt             # Python dependencies
└── README.md
</code>
</pre>
  
---
  
## 👤 Author
**namdarine** - _No-Code AI Engineer_  
🚀 Live App: [https://automlplatform.tech/](https://automlplatform.tech/)  
🧑‍💻 Portfolio: [https://namdarine.github.io](https://namdarine.github.io)  
✍️ Blog (Medium): [https://medium.com/@namdarine](https://medium.com/@namdarine)  
_I'm currently building and sharing insights about no-code AI systems and automation._  
  
Passionate about making AI more accessible, and empowering users to build AI without writing code.  
  
---
  
## 📄 License
This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
