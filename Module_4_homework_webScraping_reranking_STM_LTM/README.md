This script defines a news-aware question-answering API using FastAPI that:

🔍 Scrapes content from the BBC News website, indexes it using vector embeddings, and allows natural language querying via a /query endpoint.
🧠 Core Features:
1. Scraping + Indexing
Target: https://www.bbc.com/news

Depth-controlled crawler that recursively follows internal links (depth=2 by default).

Cleans and stores paragraph-level chunks using SentenceTransformer embeddings.

Chunks are stored in ChromaDB, a persistent vector database.

2. Query Pipeline
A query goes through:

Embedding-based retrieval (with all-MiniLM-L6-v2)

Re-ranking using CrossEncoder (ms-marco-TinyBERT)

Top results are formatted into a prompt.

3. LLM-Powered Response
A prompt is constructed using:

Retrieved chunks (short-term memory)

Prior relevant conversation history (long-term memory)

Immediate previous turn (short-term memory)

Sent to Ollama LLM (e.g., LLaMA2) running locally (http://localhost:11434)

Response is cleaned and returned as an answer.

4. Memory Management
A JSON file history.json stores all Q&A pairs.

Embedding similarity is used to retrieve the most relevant prior topic for contextual memory.

Both long-term and short-term memory are explicitly injected into the prompt.






**Example:**

{
  "query": "Is Australia's politics influenced by Trump?",
  "top_k": 10,
  "min_score": 0.3
}

{
  "answer": "Based on the provided context, it is possible to draw some conclusions about the influence of Donald Trump on Australian politics. The passage mentions that Dutton, who is the leader of the opposition in Australia, was seen by many as the country's version of Trump. This suggests that there are similarities between the two politicians, such as their populist and nationalist rhetoric. However, the passage also indicates that Australians do not appear to want a political leader like Trump, which suggests that there may be some resistance to this type of politics in Australia. Therefore, based on the context provided, it can be concluded that while there may be some similarities between Trump and Dutton, Australians do not seem to be embracing the same type of populist and nationalist politics as seen in the United States.",
  "relevant_chunks": [
    {
      "text": "Dutton, whether he liked it or not, was a man who many saw as Australia's Trump - but as it turns out Australians do not appear to want that.",
      "score": 0.8225639462471008,
      "source_url": "https://www.bbc.com/news/articles/cdxgwnj8v5eo",
      "depth": 1
    }
  ],
  "used_prompt": {
    "user_query": "Is Australia's politics influenced by Trump?",
    "retrieved_chunks": [
      "Dutton, whether he liked it or not, was a man who many saw as Australia's Trump - but as it turns out Australians do not appear to want that."
    ]
  },
  "short_term_memory": "Q: is nationalism present in Romania? A: Based on the provided context, it is not possible to determine what nationalist Simion did without additional information. The text only provides information about Simion's victory in the presidential election and his views on NATO and US troops in Romania. It does not provide any details about his actions or activities as a nationalist. Therefore, the answer to the user's question is \"based on the context provided, it is not possible to determine what nationalist Simion did without additional information.\"",
  "long_term_memory": {
    "query": "what is the name of Australia's opposition leader?",
    "answer": "Based on the provided context, the name of Australia's opposition leader is Peter Dutton.",
    "topic_label": "* Australian Politics"
  }
}

where history.json is (newest at the botton of the file):
{
  "topics": [
    {
      "query": "what did the nationalist Simion do?",
      "answer": "Based on the provided context, it is not possible to determine what nationalist Simion did without additional information. The text only provides information about Simion's victory in the presidential election and his views on NATO and US troops in Romania. It does not provide any details about his actions or activities as a nationalist. Therefore, the answer to the user's question is \"based on the context provided, it is not possible to determine what nationalist Simion did without additional information.\"",
      "topic_label": "Nationalism in Romania"
    },
    {
      "query": "what is the name of Australia's opposition leader?",
      "answer": "Based on the provided context, the name of Australia's opposition leader is Peter Dutton.",
      "topic_label": "* Australian Politics"
    },
    {
      "query": "is nationalism present in Romania?",
      "answer": "Based on the provided context, it is not possible to determine what nationalist Simion did without additional information. The text only provides information about Simion's victory in the presidential election and his views on NATO and US troops in Romania. It does not provide any details about his actions or activities as a nationalist. Therefore, the answer to the user's question is \"based on the context provided, it is not possible to determine what nationalist Simion did without additional information.\"",
      "topic_label": "Nationalism in Romania"
    },
    {
      "query": "Is Australia's politics influenced by Trump?",
      "answer": "Based on the provided context, it is possible to draw some conclusions about the influence of Donald Trump on Australian politics. The passage mentions that Dutton, who is the leader of the opposition in Australia, was seen by many as the country's version of Trump. This suggests that there are similarities between the two politicians, such as their populist and nationalist rhetoric. However, the passage also indicates that Australians do not appear to want a political leader like Trump, which suggests that there may be some resistance to this type of politics in Australia. Therefore, based on the context provided, it can be concluded that while there may be some similarities between Trump and Dutton, Australians do not seem to be embracing the same type of populist and nationalist politics as seen in the United States.",
      "topic_label": "Populism in Australia"
    }
  ]
}
