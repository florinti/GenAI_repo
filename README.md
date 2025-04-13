🧠 RAG System for UserGuides
A lightweight Retrieval-Augmented Generation (RAG) system using FastAPI, FAISS, Gemini Pro, and the CoreSim User Guide as its knowledge base.


🚀 Project Overview
This project allows users to ask natural language questions about the CoreSim User Guide. It leverages chunked PDF text, vector embeddings, and Google's Gemini Pro model to retrieve relevant information and generate precise answers.


🛠️ Tools and Libraries Used
🧠 Sentence Transformers (all-MiniLM-L6-v2) — for generating semantic embeddings

📚 Hugging Face Transformers — for tokenization and chunking

🔍 FAISS — for fast similarity search over dense embeddings

🧾 PyPDF2 — for extracting text from PDF files

⚡ FastAPI — for building the web API

🌍 Google Generative AI — Gemini Pro model for answer generation

🌱 python-dotenv — for managing secrets via .env file

🧩 Strategy: Chunking, Embedding, and Retrieval


1. 📄 Chunking Strategy
The CoreSim PDF is tokenized using a sliding window approach:

Max tokens per chunk: 200

Stride (overlap): 100 tokens

Tokenizer: sentence-transformers/all-MiniLM-L6-v2

This ensures sufficient context is preserved across chunks while maintaining size compatibility with the embedding model.

2. 🌐 Embedding Strategy
Each chunk is embedded using:

Model: all-MiniLM-L6-v2 from Sentence Transformers

Output: Normalized dense vectors

Embedding format: NumPy arrays

All vectors are L2-normalized to enable inner product search in FAISS.

3. 🔎 Retrieval Strategy
Index type: FAISS IndexFlatIP (Inner Product)

Search: Top-k chunks (default: 3) most similar to the question embedding

Prompt building: Retrieved chunks are injected into a structured prompt template passed to Gemini

4. 💬 Answer Generation
The prompt is crafted with context chunks + user question.

It’s passed to the Gemini Pro model (gemini-1.5-pro-latest) via Google Generative AI SDK.

The model responds with a concise, context-aware answer.

📬 API Usage
Endpoint: POST /ask

Request body:

{
  "question": "What options can we configure for the AMF Configuration Update procedure?",
  "top_k": 3
}
Response format:

{
  "answer": "...",
  "retrieved_chunks": ["...chunk1...", "...chunk2...", "..."],
  "prompt": "Full prompt with context...",
  "inference_time": 0.7432
}


📁 Folder Structure
.
├── main.py
├── UserDocs/
│   └── coresimUserGuide.pdf
├── .env
└── README.md

How To Run:
python .\main.py
http://127.0.0.1:8000/doc
{
  "question": "How do we configure AMF Configuration Update?",
  "top_k": 3
}
{
  "answer": "Click the \"Add AMF Configuration Update(s)\" button to configure a new AMF configuration update message.  You can also delete a configured message by clicking the \"Delete AMF Configuration Update\" button.  The available settings to modify are AMF name, AMF relative capacity, and the supplementary GUAMI and PLMN list. Note that after an AMF configuration update procedure, the newly advertised values are not applied further in the test.  There is also a \"Delay (ms)\" setting which controls the delay between NG setup and the first AMF configuration update, or between subsequent AMF configuration update procedures.",
  "retrieved_chunks": [
    "value setthevalueforthisparameter. theacceptedvaluesarebetween0 - 31. unit selecttheunitsizeforthisparameterfromthedrop - downlist. theavailable optionsare : 2s, 30s, 1m, 10m, 1h, 10handdeactivated. nssai thesesettings aredescribed below. tai thesesettings aredescribed below. publicwarning systemthesesettings aredescribed below. amf configuration updateamfconfiguration updates canmodifyamfname, amfrelative capacity, orthe supplementary guamiandplmnlist. afteranamfconfiguration update procedure, thenewlyadvertised valuesarenotapplied furtherinthetest.. thesesettings aredescribed below. emergency settingsthesesettings aredescribed below. overload configurationsele",
    "##ttings aredescribed below. amf configuration updateamfconfiguration updates canmodifyamfname, amfrelative capacity, orthe supplementary guamiandplmnlist. afteranamfconfiguration update procedure, thenewlyadvertised valuesarenotapplied furtherinthetest.. thesesettings aredescribed below. emergency settingsthesesettings aredescribed below. overload configurationselectthecheckboxtoenablethisoption. thisoptionallowsyoutoconfigure the4gand5goverload. delay thetimetowait ( inseconds ), tosendtheoverloadstartaftereachsuccessful s1setup. a0valuemeanstheprocedureisnotinitiated. duration thedurationoftheoverload, inseconds, afterwhichthe",
    "##for pws / etwsmessages. warningmessage contentsaddthecontentofthewarningmessagethatwillbebroadcasted tothe ues. amfconfiguration update ( s ) thefollowingtabledescribestheconfiguration settingsthatarerequiredforamfconfiguration update. setting description amfconfiguration update ( s ) : selecttheaddamfconfiguration update ( s ) buttontoconfigureanewamf configuration updatemessage. amfconfiguration update : selectthedeleteamfconfiguration updatebuttontodeletetheamfconfiguration updatemessageconfiguration. delay ( ms ) thedelaybetweenngsetupandthefirstamfconfiguration update, orbetween subsequentamfconfiguration updateproced"
  ],
  "prompt": "Use the following context from the CoreSim User Guide to answer the question.\n\nContext:\nvalue setthevalueforthisparameter. theacceptedvaluesarebetween0 - 31. unit selecttheunitsizeforthisparameterfromthedrop - downlist. theavailable optionsare : 2s, 30s, 1m, 10m, 1h, 10handdeactivated. nssai thesesettings aredescribed below. tai thesesettings aredescribed below. publicwarning systemthesesettings aredescribed below. amf configuration updateamfconfiguration updates canmodifyamfname, amfrelative capacity, orthe supplementary guamiandplmnlist. afteranamfconfiguration update procedure, thenewlyadvertised valuesarenotapplied furtherinthetest.. thesesettings aredescribed below. emergency settingsthesesettings aredescribed below. overload configurationsele\n\n##ttings aredescribed below. amf configuration updateamfconfiguration updates canmodifyamfname, amfrelative capacity, orthe supplementary guamiandplmnlist. afteranamfconfiguration update procedure, thenewlyadvertised valuesarenotapplied furtherinthetest.. thesesettings aredescribed below. emergency settingsthesesettings aredescribed below. overload configurationselectthecheckboxtoenablethisoption. thisoptionallowsyoutoconfigure the4gand5goverload. delay thetimetowait ( inseconds ), tosendtheoverloadstartaftereachsuccessful s1setup. a0valuemeanstheprocedureisnotinitiated. duration thedurationoftheoverload, inseconds, afterwhichthe\n\n##for pws / etwsmessages. warningmessage contentsaddthecontentofthewarningmessagethatwillbebroadcasted tothe ues. amfconfiguration update ( s ) thefollowingtabledescribestheconfiguration settingsthatarerequiredforamfconfiguration update. setting description amfconfiguration update ( s ) : selecttheaddamfconfiguration update ( s ) buttontoconfigureanewamf configuration updatemessage. amfconfiguration update : selectthedeleteamfconfiguration updatebuttontodeletetheamfconfiguration updatemessageconfiguration. delay ( ms ) thedelaybetweenngsetupandthefirstamfconfiguration update, orbetween subsequentamfconfiguration updateproced\n\nQuestion: How do we configure AMF Configuration Update?\n\nAnswer:",
  "inference_time": 3.5730865001678467
}
