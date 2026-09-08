import os
import time

import streamlit as st

from dotenv import load_dotenv

# Modern LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

# Retrieval-chain utilities are currently in langchain_classic
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

# -------------------------------------------------------
# 1. Load environment variables
# -------------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

# -------------------------------------------------------
# 2. Streamlit UI
# -------------------------------------------------------

st.title("RockyBot: News Research Tool 📈")

st.sidebar.title("News Article URLs")


# Collect URLs from the sidebar
urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")

    if url:
        urls.append(url)


# Button to process URLs
process_url_clicked = st.sidebar.button("Process URLs")


# Folder where FAISS will be stored
faiss_path = "faiss_store_openai"


# Placeholder for status messages
main_placeholder = st.empty()


# -------------------------------------------------------
# 3. Initialize LLM
# -------------------------------------------------------

llm = init_chat_model("qwen3.8:latest", model_provider="ollama", temperature=0)



# -------------------------------------------------------
# 4. Initialize embedding model
# -------------------------------------------------------

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", 
                              api_key=api_key, 
                              openai_api_base=base_url)



# -------------------------------------------------------
# 5. Process URLs
# -------------------------------------------------------

if process_url_clicked:

    if not urls:
        st.warning("Please enter at least one URL.")

    else:

        # -----------------------------------------------
        # Load web pages
        # -----------------------------------------------

        main_placeholder.text("Loading articles... ✅")

        loader = UnstructuredURLLoader(
            urls=urls
        )

        data = loader.load()

        st.write(
            f"Loaded {len(data)} documents."
        )


        # -----------------------------------------------
        # Split documents into chunks
        # -----------------------------------------------

        main_placeholder.text("Splitting text... ✅")

        text_splitter = RecursiveCharacterTextSplitter(

            # Try paragraph, newline, sentence, comma,
            # space, then character-level splitting
            separators=[
                "\n\n",
                "\n",
                ".",
                ",",
                " ",
                ""
            ],

            chunk_size=1000,

            # Some overlap helps preserve context
            chunk_overlap=100,

            length_function=len,
        )


        docs = text_splitter.split_documents(data)

        st.write(
            f"Created {len(docs)} chunks."
        )


        # -----------------------------------------------
        # Create vector embeddings + FAISS index
        # -----------------------------------------------

        main_placeholder.text(
            "Creating embeddings and FAISS index... ✅"
        )

        vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=embeddings
        )


        # -----------------------------------------------
        # Save FAISS index locally
        # -----------------------------------------------

        vectorstore.save_local(
            faiss_path
        )

        main_placeholder.text(
            "FAISS index saved successfully ✅"
        )

        time.sleep(1)


# -------------------------------------------------------
# 6. Ask questions
# -------------------------------------------------------

query = st.text_input(
    "Question:"
)


if query:

    # Make sure FAISS index exists
    if os.path.exists(faiss_path):

        # -----------------------------------------------
        # Load saved FAISS database
        # -----------------------------------------------

        vectorstore = FAISS.load_local(

            faiss_path,

            embeddings,

            # FAISS metadata uses pickle internally.
            # Only enable this for indexes YOU created/trust.
            allow_dangerous_deserialization=True,
        )


        # -----------------------------------------------
        # Convert vector store into a retriever
        # -----------------------------------------------

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 4
            }
        )


        # -----------------------------------------------
        # Prompt for RAG
        # -----------------------------------------------

        prompt = ChatPromptTemplate.from_template(
            """
            You are a news research assistant.

            Answer the user's question using only the
            information contained in the retrieved context.

            If the answer cannot be found in the context,
            say that you do not have enough information.

            Context:
            {context}

            Question:
            {input}

            Answer:
            """
        )


        # -----------------------------------------------
        # Chain that sends retrieved documents to LLM
        # -----------------------------------------------

        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt,
        )


        # -----------------------------------------------
        # Full RAG chain:
        #
        # Question
        #   ↓
        # Retriever
        #   ↓
        # Relevant documents
        #   ↓
        # Prompt
        #   ↓
        # LLM
        #   ↓
        # Answer
        # -----------------------------------------------

        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )


        # -----------------------------------------------
        # Execute RAG pipeline
        # -----------------------------------------------

        result = retrieval_chain.invoke(
            {
                "input": query
            }
        )


        # -----------------------------------------------
        # Display answer
        # -----------------------------------------------

        st.header("Answer")

        st.write(
            result["answer"]
        )


        # -----------------------------------------------
        # Display sources
        # -----------------------------------------------

        source_docs = result.get(
            "context",
            []
        )


        if source_docs:

            st.subheader("Sources:")

            # Avoid displaying duplicate URLs
            sources = []

            for doc in source_docs:

                source = doc.metadata.get(
                    "source"
                )

                if source and source not in sources:
                    sources.append(source)


            for source in sources:
                st.write(source)

    else:

        st.warning(
            "Please process the URLs first."
        )
