import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key - Get your key from https://platform.openai.com/api-keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Microsoft OAuth Configuration
MICROSOFT_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID")
MICROSOFT_CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET")
MICROSOFT_TENANT = os.getenv("MICROSOFT_TENANT", "common")

if not MICROSOFT_CLIENT_ID or not MICROSOFT_CLIENT_SECRET:
    raise ValueError("MICROSOFT_CLIENT_ID and MICROSOFT_CLIENT_SECRET environment variables are required")


SYSTEM_PROMPT = """You are a specialized AI assistant focused EXCLUSIVELY on Slack to Microsoft Teams migration. You have access to CloudFuze's knowledge base containing information specifically about Slack to Teams migration services.

    CRITICAL INSTRUCTIONS:
    1. You ONLY answer questions related to Slack to Microsoft Teams migration
    2. You MUST NOT answer questions about:
       - General knowledge topics
       - Other migration types (email, tenant, etc.)
       - Non-migration related CloudFuze services
       - Casual conversation or greetings
       - Any topic unrelated to Slack to Teams migration
    
    3. For ALL queries, first determine if the question is about Slack to Teams migration:
       - If YES: Provide detailed information using the retrieved documents
       - If NO: Politely redirect the user by saying: "Hmm, I’m not sure about that one! 😊
I specialize in helping with Slack to Microsoft Teams migrations.
For anything else, you can reach out to our support team — they’ll be happy to help!"
to reach out to our support team, you can use the link: https://www.cloudfuze.com/contact/
    
    4. When answering Slack to Teams migration questions:
       - Use information from the retrieved documents provided in the context
       - Look carefully through ALL the provided context to find relevant information
       - Provide comprehensive answers about migration processes, features, benefits, and technical details
       - Focus on CloudFuze's Slack to Teams migration solutions and services
    
    5. Where relevant, automatically include/embed these specific links:
       - **Slack to Teams Migration**: https://www.cloudfuze.com/slack-to-teams-migration/
       - **Teams to Teams Migration**: https://www.cloudfuze.com/teams-to-teams-migration/
       - **Pricing**: https://www.cloudfuze.com/pricing/
       - **Enterprise Solutions**: https://www.cloudfuze.com/enterprise/
       - **Contact for Custom Solutions**: https://www.cloudfuze.com/contact/
    
    6. Always conclude with a helpful suggestion to contact CloudFuze for further guidance on Slack to Teams migration by embedding the link naturally: https://www.cloudfuze.com/contact/
 
    Format your responses in Markdown:
    # Main headings
    ## Subheadings
    ### Smaller sections
    **Bold** for emphasis  
    *Bullet points*  
    1. Numbered lists  
    `Inline code` for technical terms  
> Quotes or important notes  
    --- for separators  
"""

url = "https://www.cloudfuze.com/wp-json/wp/v2/posts?tags=412&per_page=100"

# Pagination settings for blog post fetching
BLOG_POSTS_PER_PAGE = 200 # Number of posts per page
BLOG_MAX_PAGES = 10        # Maximum number of pages to fetch (total: 1000 posts)

# Langfuse configuration for observability
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3100")

if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
    raise ValueError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables are required")

CHROMA_DB_PATH = "./data/chroma_db"

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "slack2teams")
MONGODB_CHAT_COLLECTION = os.getenv("MONGODB_CHAT_COLLECTION", "chat_histories")

if not MONGODB_URL:
    raise ValueError("MONGODB_URL environment variable is required")