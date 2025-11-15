"""Generative AI Application Starter Template.

Complete starter template for building production-ready generative AI applications
with multiple model support, RAG capabilities, and monitoring.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# AI and ML libraries
import structlog
import uvicorn
import yaml

# Web framework
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.document_loaders import PDFLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Monitoring and observability
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Metrics
request_counter = Counter("ai_requests_total", "Total AI requests", ["model", "operation", "status"])
response_time = Histogram("ai_response_time_seconds", "AI response time", ["model", "operation"])
active_conversations = Gauge("active_conversations", "Number of active conversations")


# Configuration
@dataclass
class AIConfig:
    """AI model configuration."""

    openai_api_key: str
    anthropic_api_key: str
    default_model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    enable_streaming: bool = True
    enable_rag: bool = True
    vector_store_path: str = "./vector_store"
    documents_path: str = "./documents"


def load_config() -> AIConfig:
    """Load configuration from file or environment."""
    config_file = Path("config/ai_config.yaml")

    if config_file.exists():
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
        return AIConfig(**config_data)

    # Fallback to environment variables
    import os

    return AIConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""), anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "")
    )


# Pydantic models for API
class ChatMessage(BaseModel):
    role: str = Field(..., enum=["user", "assistant", "system"])
    content: str = Field(..., max_length=10000)
    timestamp: datetime | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=4000)
    conversation_id: str | None = None
    model: str | None = None
    stream: bool = False
    include_sources: bool = True
    max_tokens: int | None = None
    temperature: float | None = None


class CodeGenerationRequest(BaseModel):
    description: str = Field(..., max_length=1000)
    language: str = Field(default="python")
    style: str = Field(default="clean")
    include_tests: bool = True
    include_docs: bool = True


class DocumentAnalysisRequest(BaseModel):
    document_path: str
    question: str = Field(..., max_length=500)
    context_size: int = Field(default=4, ge=1, le=10)


class AIModelManager:
    """Manages multiple AI models and provides unified interface."""

    def __init__(self, config: AIConfig):
        self.config = config
        self.models = {}
        self.embeddings = None
        self.vector_store = None
        self.memory_stores = {}  # conversation_id -> memory

        self._initialize_models()
        if config.enable_rag:
            self._initialize_rag()

    def _initialize_models(self):
        """Initialize AI models."""
        try:
            # OpenAI models
            if self.config.openai_api_key:
                self.models["gpt-3.5-turbo"] = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    openai_api_key=self.config.openai_api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

                self.models["gpt-4"] = ChatOpenAI(
                    model_name="gpt-4",
                    openai_api_key=self.config.openai_api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

            # Anthropic models
            if self.config.anthropic_api_key:
                self.models["claude-3-sonnet"] = ChatAnthropic(
                    model="claude-3-sonnet-20240229",
                    anthropic_api_key=self.config.anthropic_api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

            logger.info("AI models initialized", models=list(self.models.keys()))

        except Exception as e:
            logger.error("Failed to initialize models", error=str(e))
            raise

    def _initialize_rag(self):
        """Initialize Retrieval-Augmented Generation."""
        try:
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.config.openai_api_key)

            # Load or create vector store
            vector_store_path = Path(self.config.vector_store_path)

            if vector_store_path.exists():
                # Load existing vector store
                self.vector_store = Chroma(persist_directory=str(vector_store_path), embedding_function=self.embeddings)
                logger.info("Loaded existing vector store", path=str(vector_store_path))
            else:
                # Create new vector store from documents
                documents = self._load_documents()
                if documents:
                    self.vector_store = Chroma.from_documents(
                        documents, self.embeddings, persist_directory=str(vector_store_path)
                    )
                    logger.info("Created new vector store", documents=len(documents))
                else:
                    logger.warning("No documents found for RAG initialization")

        except Exception as e:
            logger.error("Failed to initialize RAG", error=str(e))

    def _load_documents(self) -> list[Any]:
        """Load documents for RAG."""
        documents = []
        docs_path = Path(self.config.documents_path)

        if not docs_path.exists():
            return documents

        # Load text files
        for txt_file in docs_path.glob("**/*.txt"):
            loader = TextLoader(str(txt_file))
            documents.extend(loader.load())

        # Load PDF files
        for pdf_file in docs_path.glob("**/*.pdf"):
            loader = PDFLoader(str(pdf_file))
            documents.extend(loader.load())

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        return text_splitter.split_documents(documents)

    def get_memory(self, conversation_id: str):
        """Get or create conversation memory."""
        if conversation_id not in self.memory_stores:
            self.memory_stores[conversation_id] = ConversationBufferMemory(
                return_messages=True, memory_key="chat_history"
            )
        return self.memory_stores[conversation_id]

    async def chat(self, request: ChatRequest) -> dict[str, Any]:
        """Handle chat request."""
        model_name = request.model or self.config.default_model

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")

        model = self.models[model_name]
        conversation_id = request.conversation_id or "default"

        try:
            with get_openai_callback() as cb:
                if self.config.enable_rag and self.vector_store:
                    # Use RAG for enhanced responses
                    memory = self.get_memory(conversation_id)

                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=model,
                        retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
                        memory=memory,
                        return_source_documents=request.include_sources,
                    )

                    result = await asyncio.to_thread(qa_chain, {"question": request.message})

                    response = {
                        "message": result["answer"],
                        "conversation_id": conversation_id,
                        "model": model_name,
                        "tokens_used": cb.total_tokens,
                        "cost": cb.total_cost,
                        "has_sources": bool(result.get("source_documents")),
                    }

                    if request.include_sources and "source_documents" in result:
                        response["sources"] = [
                            {"content": doc.page_content[:200] + "...", "metadata": doc.metadata}
                            for doc in result["source_documents"]
                        ]

                else:
                    # Simple chat without RAG
                    response_text = await asyncio.to_thread(model.predict, request.message)

                    response = {
                        "message": response_text,
                        "conversation_id": conversation_id,
                        "model": model_name,
                        "tokens_used": cb.total_tokens,
                        "cost": cb.total_cost,
                    }

            request_counter.labels(model=model_name, operation="chat", status="success").inc()

            logger.info(
                "Chat completed",
                model=model_name,
                tokens=cb.total_tokens,
                cost=cb.total_cost,
                conversation_id=conversation_id,
            )

            return response

        except Exception as e:
            request_counter.labels(model=model_name, operation="chat", status="error").inc()

            logger.error("Chat failed", error=str(e), model=model_name, conversation_id=conversation_id)
            raise

    async def generate_code(self, request: CodeGenerationRequest) -> dict[str, Any]:
        """Generate code based on description."""
        model = self.models[self.config.default_model]

        prompt = f"""
        Generate {request.language} code for the following description:
        
        Description: {request.description}
        Style: {request.style}
        
        Requirements:
        - Clean, readable code
        - Proper error handling
        - Type hints (where applicable)
        - Security best practices
        """

        if request.include_tests:
            prompt += "\n- Include unit tests"

        if request.include_docs:
            prompt += "\n- Include comprehensive documentation"

        try:
            with get_openai_callback() as cb:
                response_text = await asyncio.to_thread(model.predict, prompt)

            request_counter.labels(model=self.config.default_model, operation="code_generation", status="success").inc()

            return {
                "code": response_text,
                "language": request.language,
                "description": request.description,
                "tokens_used": cb.total_tokens,
                "cost": cb.total_cost,
            }

        except Exception as e:
            request_counter.labels(model=self.config.default_model, operation="code_generation", status="error").inc()

            logger.error("Code generation failed", error=str(e))
            raise


# Initialize FastAPI app
app = FastAPI(
    title="Generative AI Application",
    description="Production-ready generative AI application with multiple model support",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
config = load_config()
ai_manager = AIModelManager(config)


# API Endpoints
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat with AI models."""
    try:
        response = await ai_manager.chat(request)
        return response
    except Exception as e:
        logger.error("Chat endpoint failed", error=str(e))
        raise HTTPException(status_code=500, detail="Chat service unavailable")


@app.post("/generate-code")
async def generate_code_endpoint(request: CodeGenerationRequest):
    """Generate code with AI."""
    try:
        response = await ai_manager.generate_code(request)
        return response
    except Exception as e:
        logger.error("Code generation endpoint failed", error=str(e))
        raise HTTPException(status_code=500, detail="Code generation service unavailable")


@app.get("/models")
async def list_models():
    """List available AI models."""
    return {
        "models": list(ai_manager.models.keys()),
        "default_model": config.default_model,
        "rag_enabled": config.enable_rag,
        "streaming_enabled": config.enable_streaming,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_available": len(ai_manager.models),
        "rag_initialized": ai_manager.vector_store is not None,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics."""
    return generate_latest().decode("utf-8")


# WebSocket for streaming (optional)
from fastapi import WebSocket


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    conversation_id = f"ws_{datetime.now().timestamp()}"

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            request = ChatRequest(**data, conversation_id=conversation_id)

            # Process and send response
            response = await ai_manager.chat(request)
            await websocket.send_json(response)

    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run("ai_app_starter:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
