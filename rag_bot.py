"""
rag_bot.py  -  RAG backend for the Zyro Dynamics HR Assistant
=============================================================

This file is your working notebook code (code.ipynb) moved into a normal
Python file so that app.py can import it:

    from rag_bot import ask_bot

WHAT WAS CHANGED vs. THE NOTEBOOK?
  Only "housekeeping" things were removed. The RAG logic is identical:
    - removed: `!pip install ...`  (a notebook-only command; install once in terminal)
    - removed: the unused first LLM_PROVIDER / LLM_MODEL / Kaggle CORPUS_PATH cell
      (LLM_MODEL was overwritten later by the ChatGroq object anyway)
    - removed: the stray "10" line, the retriever test cell, the `test_llm`
      "Say hello" cell, and the test.csv / submission.csv cells
      (those are for testing in the notebook, not for the chatbot)
    - changed:  CORPUS_PATH now points to the PDF folder relative to THIS FILE,
      so the app works no matter which folder you launch Streamlit from.

  UNCHANGED: chunking (1200 / 400), embeddings (all-MiniLM-L6-v2), FAISS,
  retriever (similarity, k=5), LLM (Groq, gpt-oss-120b, temp 0.7, 500 tokens),
  RAG prompt, guardrail prompt, refusal message, rag_chain() and ask_bot().
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

# --------------------------------------------------------------------------
# 1. Environment (.env -> GROQ_API_KEY)
# --------------------------------------------------------------------------
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY not found. Check your .env file.")

# --------------------------------------------------------------------------
# 2. Document loading + chunking
# --------------------------------------------------------------------------
# Path(__file__).parent = the folder this file lives in, so the corpus folder
# is found even if you run `streamlit run` from somewhere else.
CORPUS_PATH = str(Path(__file__).parent / "zyro-dynamics-hr-corpus")

loader = PyPDFDirectoryLoader(CORPUS_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=400,
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# --------------------------------------------------------------------------
# 3. Embeddings + FAISS vector store + retriever
# --------------------------------------------------------------------------
model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embeddings model loaded")

vectorstore = FAISS.from_documents(chunks, model)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    },
)
print("Retriever ready")

# --------------------------------------------------------------------------
# 4. LLM
# --------------------------------------------------------------------------
LLM_MODEL = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=500,
)
print("LLM model: groq initialized")

# --------------------------------------------------------------------------
# 5. RAG chain
# --------------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_template(
"""
You are an HR assistant. Answer employee questions using only the information provided in the HR policy context.

If the context contains the information needed to answer the question, give a clear and natural answer based on the context.

Do not use information outside the provided context and do not make up an answer.

If the question is not related to HR policies, politely explain that you can only help with HR policy-related questions.

If the question is related to HR but the required information is not available in the context, politely say that the information is not available in the provided HR policies.

Context: {context}

Question: {question}

"""
)


# all the info required for LLM
def format_docs(docs):
    return ".\n\n".join(d.page_content for d in docs)


@traceable(name="rag_chain")
def rag_chain(question: str):
    docs = retriever.invoke(question)
    context = format_docs(docs)

    chain = RAG_PROMPT | LLM_MODEL | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": answer,
        "sources": docs
    }


# --------------------------------------------------------------------------
# 6. Guardrail (to ensure no context is used beyond the corpus)
# --------------------------------------------------------------------------
GUARDIAL_PROMPT = ChatPromptTemplate.from_template("""
You are helping employees of Zyro Dynamics with questions about their
company policies.

First, check whether the question is related to an internal HR policy such
as leave, salary, benefits, travel, performance, work from home, company
conduct, IT and data security, POSH, ESOP, or resignation.

If the question is related to the company's HR policies, return IN_SCOPE.
If it is about something unrelated to the company's HR policies, such as
general knowledge, coding, another company, or an unrelated product,
return OUT_OF_SCOPE.

Respond with exactly one word: IN_SCOPE or OUT_OF_SCOPE.

Question: {question}
""")

REFUSAL_MESSAGE = (
    "I'm an HR assistant and can only help with questions about company HR "
    "policies (leave, reimbursement, code of conduct, etc.). "
    "I don't have information to answer that question."
)


def ask_bot(question: str):

    guardial_chain = GUARDIAL_PROMPT | LLM_MODEL | StrOutputParser()

    verdict = guardial_chain.invoke({
        "question": question
    }).strip().upper()

    if "OUT_OF_SCOPE" in verdict:
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": []
        }

    return rag_chain(question)


print("Guardial Initialized!")
