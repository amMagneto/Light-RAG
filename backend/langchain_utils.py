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

# initialise the LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def get_simple_chain():
    prompt = ChatPromptTemplate.from_template("Explain {topic} in a concise way.")

    output_parser = StrOutputParser()

    # This is the LCEL
    chain = prompt | llm | output_parser
    return chain

def get_retriever(vectorstore, k: int = 5):
    """Returns a standard similarity retriever."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

def get_conversational_rag_chain(vectorstore, k: int = 5):
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood "
        "without the chat history. ALWAYS prefix the standalone question with 'query: '. "
        "Preserve all specific terms, names, and keywords exactly as stated."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    retriever = get_retriever(vectorstore, k)

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are a helpful assistant that answers questions using the provided context. "
        "Use the retrieved context as your primary source. "
        "If the context contains relevant information, answer confidently and specifically. "
        "Only say 'I don't know based on the provided sources' if the context contains "
        "absolutely no information related to the question.\n\n"
        "Retrieved context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

   