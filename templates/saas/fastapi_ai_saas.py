"""Production-Ready FastAPI SaaS Boilerplate for AI Applications.

This template provides a complete SaaS foundation with authentication,
payments, AI model integration, and monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta

import jwt
import openai
import redis
import stripe
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI SaaS Platform",
    description="Production-ready SaaS platform with AI capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Security
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted hosts middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com", "*.yourdomain.com"])

# Database setup
DATABASE_URL = "postgresql://user:password@localhost/saas_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup for caching and rate limiting
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Stripe setup
stripe.api_key = "sk_test_your_stripe_secret_key"

# OpenAI setup
openai.api_key = "your_openai_api_key"

# Metrics
request_count = Counter("requests_total", "Total requests", ["method", "endpoint"])
request_duration = Histogram("request_duration_seconds", "Request duration")
ai_requests = Counter("ai_requests_total", "Total AI requests", ["model", "status"])


# Database Models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    subscription_tier = Column(String, default="free")
    api_key = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class APIUsage(Base):
    __tablename__ = "api_usage"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    endpoint = Column(String)
    tokens_used = Column(Integer)
    cost = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)


# Pydantic Models
class UserCreate(BaseModel):
    email: str = Field(..., example="user@example.com")
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    email: str
    password: str


class AIRequest(BaseModel):
    prompt: str = Field(..., max_length=4000, example="Generate a Python function to sort a list")
    model: str = Field(default="gpt-3.5-turbo", example="gpt-3.5-turbo")
    max_tokens: int = Field(default=1000, le=4000)
    temperature: float = Field(default=0.7, ge=0, le=2)


class CodeGenerationRequest(BaseModel):
    description: str = Field(..., example="Create a REST API endpoint for user management")
    language: str = Field(default="python", example="python")
    framework: str = Field(default="fastapi", example="fastapi")
    include_tests: bool = Field(default=True)


class SubscriptionRequest(BaseModel):
    tier: str = Field(..., example="pro")
    payment_method: str = Field(..., example="pm_1234567890")


# Dependency functions
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)
):
    """Get current authenticated user."""
    try:
        payload = jwt.decode(credentials.credentials, "your_secret_key", algorithms=["HS256"])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")

        return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def check_rate_limit(user: User = Depends(get_current_user)):
    """Check API rate limits based on subscription tier."""
    rate_limits = {
        "free": {"requests": 100, "window": 3600},  # 100 requests per hour
        "pro": {"requests": 1000, "window": 3600},  # 1000 requests per hour
        "enterprise": {"requests": 10000, "window": 3600},  # 10000 requests per hour
    }

    limit = rate_limits.get(user.subscription_tier, rate_limits["free"])
    key = f"rate_limit:{user.id}"

    current_requests = redis_client.get(key)
    if current_requests is None:
        redis_client.setex(key, limit["window"], 1)
    elif int(current_requests) >= limit["requests"]:
        raise HTTPException(
            status_code=429, detail=f"Rate limit exceeded. Limit: {limit['requests']} requests per hour"
        )
    else:
        redis_client.incr(key)

    return True


# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    request_count.labels(method=request.method, endpoint=request.url.path).inc()
    request_duration.observe(duration)

    return response


# Authentication endpoints
@app.post("/auth/register", tags=["Authentication"])
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash password (use proper hashing in production)
    hashed_password = f"hashed_{user.password}"  # Replace with bcrypt

    # Generate API key
    api_key = f"sk_{user.email}_{int(time.time())}"

    # Create user
    db_user = User(email=user.email, hashed_password=hashed_password, api_key=api_key)

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Create JWT token
    token = jwt.encode(
        {"user_id": db_user.id, "exp": datetime.utcnow() + timedelta(days=30)}, "your_secret_key", algorithm="HS256"
    )

    return {"message": "User registered successfully", "user_id": db_user.id, "api_key": api_key, "token": token}


@app.post("/auth/login", tags=["Authentication"])
async def login(user_login: UserLogin, db: Session = Depends(get_db)):
    """User login."""
    user = db.query(User).filter(User.email == user_login.email).first()
    if not user or user.hashed_password != f"hashed_{user_login.password}":
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account deactivated")

    token = jwt.encode(
        {"user_id": user.id, "exp": datetime.utcnow() + timedelta(days=30)}, "your_secret_key", algorithm="HS256"
    )

    return {"token": token, "user_id": user.id, "subscription_tier": user.subscription_tier}


# AI endpoints
@app.post("/ai/generate", tags=["AI Services"])
async def generate_text(
    request: AIRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    rate_limit: bool = Depends(check_rate_limit),
    db: Session = Depends(get_db),
):
    """Generate text using AI models."""
    try:
        start_time = time.time()

        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        duration = time.time() - start_time
        tokens_used = response["usage"]["total_tokens"]
        cost = calculate_cost(request.model, tokens_used)

        # Log usage in background
        background_tasks.add_task(log_api_usage, user.id, "/ai/generate", tokens_used, cost, db)

        ai_requests.labels(model=request.model, status="success").inc()

        return {
            "response": response["choices"][0]["message"]["content"],
            "tokens_used": tokens_used,
            "cost": cost,
            "model": request.model,
            "processing_time": duration,
        }

    except Exception as e:
        ai_requests.labels(model=request.model, status="error").inc()
        logger.error(f"AI generation failed: {e}")
        raise HTTPException(status_code=500, detail="AI service temporarily unavailable")


@app.post("/ai/code-generation", tags=["AI Services"])
async def generate_code(
    request: CodeGenerationRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    rate_limit: bool = Depends(check_rate_limit),
    db: Session = Depends(get_db),
):
    """Generate code with AI assistance."""
    prompt = f"""
    Generate {request.language} code using {request.framework} framework:
    
    Description: {request.description}
    
    Requirements:
    - Clean, well-documented code
    - Proper error handling
    - Type hints (where applicable)
    - Security best practices
    """

    if request.include_tests:
        prompt += "\n- Include unit tests"

    try:
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert software engineer. Generate production-quality code."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.3,
        )

        tokens_used = response["usage"]["total_tokens"]
        cost = calculate_cost("gpt-4", tokens_used)

        background_tasks.add_task(log_api_usage, user.id, "/ai/code-generation", tokens_used, cost, db)

        return {
            "code": response["choices"][0]["message"]["content"],
            "language": request.language,
            "framework": request.framework,
            "tokens_used": tokens_used,
            "cost": cost,
        }

    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise HTTPException(status_code=500, detail="Code generation service unavailable")


# Subscription management
@app.post("/subscription/upgrade", tags=["Subscription"])
async def upgrade_subscription(
    request: SubscriptionRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Upgrade user subscription."""
    subscription_prices = {"pro": "price_pro_monthly", "enterprise": "price_enterprise_monthly"}

    if request.tier not in subscription_prices:
        raise HTTPException(status_code=400, detail="Invalid subscription tier")

    try:
        # Create Stripe subscription
        subscription = stripe.Subscription.create(
            customer=user.stripe_customer_id,  # Assuming this exists
            items=[
                {
                    "price": subscription_prices[request.tier],
                }
            ],
            payment_behavior="default_incomplete",
            expand=["latest_invoice.payment_intent"],
        )

        # Update user subscription
        user.subscription_tier = request.tier
        db.commit()

        return {
            "subscription_id": subscription.id,
            "client_secret": subscription.latest_invoice.payment_intent.client_secret,
            "status": subscription.status,
        }

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(status_code=400, detail="Payment processing failed")


# Analytics and usage
@app.get("/analytics/usage", tags=["Analytics"])
async def get_usage_analytics(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get user's API usage analytics."""
    # Get usage for last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)

    usage_data = db.query(APIUsage).filter(APIUsage.user_id == user.id, APIUsage.timestamp >= thirty_days_ago).all()

    total_requests = len(usage_data)
    total_tokens = sum(record.tokens_used for record in usage_data)
    total_cost = sum(float(record.cost) for record in usage_data)

    return {
        "period": "30_days",
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "total_cost": f"${total_cost:.4f}",
        "average_tokens_per_request": total_tokens / max(total_requests, 1),
        "subscription_tier": user.subscription_tier,
    }


# Health check and metrics
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.0"}


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest().decode("utf-8")


# Utility functions
def calculate_cost(model: str, tokens: int) -> str:
    """Calculate cost based on model and token usage."""
    pricing = {
        "gpt-3.5-turbo": 0.002 / 1000,  # $0.002 per 1K tokens
        "gpt-4": 0.03 / 1000,  # $0.03 per 1K tokens
    }

    rate = pricing.get(model, 0.002 / 1000)
    return f"{tokens * rate:.6f}"


async def log_api_usage(user_id: int, endpoint: str, tokens: int, cost: str, db: Session):
    """Log API usage to database."""
    usage_record = APIUsage(user_id=user_id, endpoint=endpoint, tokens_used=tokens, cost=cost)

    db.add(usage_record)
    db.commit()


# Create database tables
Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
