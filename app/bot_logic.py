# At the VERY TOP of bot_logic.py (first lines)
from dotenv import load_dotenv
load_dotenv()  # This should be before any other imports
import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI  # Updated import
from langchain.embeddings import HuggingFaceEmbeddings  # Corrected import
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

class FinalBot:
    def __init__(self):
        # Initialize components
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all RAG components"""
        # Configuration
        self.DB_FAISS_PATH = "vectorstore/db_faiss"
        
        # Initialize memory (last exchange only)
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=1,
            return_messages=True,
            output_key="answer"
        )
        
        # Initialize LLM (OpenRouter API with Mistral-7B)
        self.llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),  # Load from environment
            model="mistralai/mistral-7b-instruct",
            temperature=0.7,
            max_tokens=1024,
            streaming=False,  # Disable streaming for API
        )
        
        # Prompt template for HyDE
        self.prompt_template = PromptTemplate(
            input_variables=["question"],
            template="Please write a detailed answer to the following question:\n\n{question}"
        )
        
        # LLMChain for HyDE
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        
        # Embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # HyDE wrapper around base embeddings
        self.hyde_embeddings = HypotheticalDocumentEmbedder(
            llm_chain=self.llm_chain,
            base_embeddings=self.embedding_model
        )
        
        # Load FAISS vector store
        self.vectorstore = FAISS.load_local(
            self.DB_FAISS_PATH,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )
        
        # Retriever with HyDE
        self.retriever = self.vectorstore.as_retriever(embedding=self.hyde_embeddings)
        
        # QA Chain with memory and source documents
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            output_key="answer"
        )
    
    def get_response(self, user_query: str) -> Dict[str, Any]:
        """Get response from RAG system"""
        try:
            response = self.qa_chain.invoke({"question": user_query})
            
            # Extract unique sources only
            sources = list(set(
                doc.metadata.get("source", "Unknown") 
                for doc in response["source_documents"]
            ))
            
            return {
                "answer": response["answer"],
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"Error processing your request: {str(e)}",
                "sources": []
            }

# Initialize the bot instance
bot_instance = FinalBot()

def get_bot_response(question: str) -> Dict[str, Any]:
    """Interface function for FastAPI"""
    return bot_instance.get_response(question)