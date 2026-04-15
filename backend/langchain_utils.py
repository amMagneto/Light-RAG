import os
import langchain
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder

# load environment variables from .env file
load_dotenv()

# Initialising the LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def get_simple_chain():
    prompt = ChatPromptTemplate.from_template("Explain {topic} in a concise way.")

    output_parser = StrOutputParser()

    # This is the LCEL
    chain = prompt | llm | output_parser
    return chain

def get_conversational_rag_chain(vectorstore):
    # 1. how to re-write the question to be standalone
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history,"
        "forumulate a standalone question which can be understood"
        "without the chat history"
    )

    contextualize_q_prompt= ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    #2 rewrite question then search vectorstore with the standalone question
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.35},
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 3. combine retrieved documents with question to answer
    system_prompt = (
        "You are a strict retrieval QA assistant. "
        "Answer using only the retrieved context below. "
        "Do not use outside knowledge, assumptions, or parametric memory. "
        "If the answer is not explicitly present in the context, reply exactly: "
        "I don't know based on the provided sources.\n\n"
        "Retrieved context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),

    ])

    # 4. orchestrate the chain(retrieval and generation)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

   