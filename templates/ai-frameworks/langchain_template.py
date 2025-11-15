"""LangChain Integration Template for AI Development Workspace.

This template provides a production-ready foundation for building AI applications
using LangChain framework with proper error handling, logging, and configuration.
"""

import logging
from pathlib import Path
from typing import Any

from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIWorkspaceAgent:
    """Production-ready LangChain agent for AI development workspace."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 1000):
        """Initialize the AI agent with configuration.

        Args:
            model_name: Name of the LLM model to use
            temperature: Creativity level (0-1)
            max_tokens: Maximum tokens per response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

        # Initialize memory
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

        # Initialize conversation chain
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory, verbose=True)

        logger.info(f"Initialized AIWorkspaceAgent with model: {model_name}")

    def generate_code(
        self, description: str, language: str = "python", style_guide: str | None = None
    ) -> dict[str, Any]:
        """Generate code based on description with quality standards.

        Args:
            description: What the code should do
            language: Programming language
            style_guide: Specific style requirements

        Returns:
            Dictionary with generated code and metadata
        """
        style_prompt = f" following {style_guide} style guide" if style_guide else ""

        prompt = f"""
        Generate high-quality {language} code{style_prompt} for the following requirement:
        
        Requirement: {description}
        
        Please provide:
        1. Clean, well-documented code
        2. Proper error handling
        3. Type hints (where applicable)
        4. Unit test examples
        5. Usage documentation
        
        Code:
        """

        try:
            with get_openai_callback() as cb:
                response = self.conversation.predict(input=prompt)

            return {
                "code": response,
                "language": language,
                "tokens_used": cb.total_tokens,
                "cost": cb.total_cost,
                "description": description,
            }

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {"error": str(e), "description": description}

    def review_code(self, code: str, language: str = "python") -> dict[str, Any]:
        """Review code for quality, security, and best practices.

        Args:
            code: Code to review
            language: Programming language

        Returns:
            Code review with suggestions
        """
        prompt = f"""
        Please review this {language} code for:
        1. Code quality and readability
        2. Security vulnerabilities
        3. Performance issues
        4. Best practices compliance
        5. Potential bugs
        
        Code to review:
        ```{language}
        {code}
        ```
        
        Provide detailed feedback with specific suggestions for improvement.
        """

        try:
            with get_openai_callback() as cb:
                review = self.conversation.predict(input=prompt)

            return {"review": review, "language": language, "tokens_used": cb.total_tokens, "cost": cb.total_cost}

        except Exception as e:
            logger.error(f"Code review failed: {e}")
            return {"error": str(e)}

    def setup_rag_pipeline(self, documents_path: Path) -> BaseRetriever:
        """Set up Retrieval-Augmented Generation pipeline.

        Args:
            documents_path: Path to documents for knowledge base

        Returns:
            Configured retriever for RAG
        """
        try:
            # Load documents
            loader = TextLoader(str(documents_path))
            documents = loader.load()

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            # Create embeddings
            embeddings = OpenAIEmbeddings()

            # Create vector store
            vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")

            # Return retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            logger.info(f"RAG pipeline setup complete with {len(texts)} chunks")
            return retriever

        except Exception as e:
            logger.error(f"RAG setup failed: {e}")
            raise

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics and costs.

        Returns:
            Usage statistics
        """
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "memory_length": len(self.memory.chat_memory.messages),
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize agent
    agent = AIWorkspaceAgent(model_name="gpt-3.5-turbo", temperature=0.3)

    # Example: Generate a Python function
    result = agent.generate_code(
        description="Create a function to validate email addresses using regex", language="python", style_guide="PEP 8"
    )

    print("Generated Code:")
    print(result.get("code", "No code generated"))

    # Example: Review code
    sample_code = r"""
    def validate_email(email):
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    """

    review = agent.review_code(sample_code, "python")
    print("\nCode Review:")
    print(review.get("review", "No review available"))

    # Print usage stats
    print("\nUsage Stats:")
    print(agent.get_usage_stats())
