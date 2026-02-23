"""
AstraDB Population Script
Helper script to add sample documents to your AstraDB vector store
"""

from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()


def populate_astradb_with_samples():
    """
    Populates AstraDB with sample company/technical documents.
    Customize this with your own documents!
    """
    
    print("🔧 Initializing AstraDB connection...")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        # model="text-embedding-3-small"
        model="text-embedding-3-large",
        dimensions=1024
    )
    
    # Initialize vector store
    vector_store = AstraDBVectorStore(
        embedding=embeddings,
        collection_name=os.getenv("ASTRA_DB_COLLECTION", "default_collection"),
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        namespace=os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace"),
    )
    
    print("✅ Connected to AstraDB")
    
    # Sample documents (replace with your actual content)
    sample_documents = [
        Document(
            page_content="""
            Company Security Policy
            
            All employees must follow these security guidelines:
            1. Use strong passwords with at least 12 characters
            2. Enable two-factor authentication on all company accounts
            3. Never share credentials with anyone
            4. Report suspicious emails to security@company.com
            5. Lock your computer when stepping away
            6. Use VPN when working remotely
            7. Encrypt sensitive data before sharing
            
            Violations may result in disciplinary action.
            """,
            metadata={"source": "security_policy", "department": "IT"}
        ),
        Document(
            page_content="""
            Employee Onboarding Process
            
            Week 1:
            - Complete HR paperwork
            - Set up workstation and accounts
            - Meet with direct manager
            - Complete security training
            
            Week 2:
            - Shadow team members
            - Begin initial projects
            - Schedule 1:1 with manager
            
            Week 3-4:
            - Take on independent tasks
            - Participate in team meetings
            - Complete compliance training
            
            30-day check-in with HR scheduled automatically.
            """,
            metadata={"source": "hr_handbook", "department": "HR"}
        ),
        Document(
            page_content="""
            Vacation Request Policy
            
            To request time off:
            1. Submit request at least 2 weeks in advance
            2. Use the HR portal: portal.company.com/pto
            3. Get manager approval
            4. Update team calendar
            
            Annual allowance:
            - 0-2 years: 15 days
            - 3-5 years: 20 days
            - 6+ years: 25 days
            
            Unused days can roll over up to 5 days per year.
            """,
            metadata={"source": "hr_handbook", "department": "HR"}
        ),
        Document(
            page_content="""
            Product X Technical Specifications
            
            Hardware:
            - Processor: Quad-core ARM Cortex-A72
            - RAM: 8GB DDR4
            - Storage: 256GB NVMe SSD
            - Connectivity: Wi-Fi 6, Bluetooth 5.2, Ethernet
            
            Software:
            - Operating System: Linux Ubuntu 22.04 LTS
            - Runtime: Python 3.11, Node.js 20
            - Database: PostgreSQL 15
            
            Performance:
            - Max throughput: 10,000 requests/second
            - Latency: <50ms average
            - Uptime: 99.9% SLA
            """,
            metadata={"source": "product_specs", "department": "Engineering", "product": "Product X"}
        ),
        Document(
            page_content="""
            Remote Work Guidelines
            
            Eligibility:
            - Must be employed for at least 3 months
            - Manager approval required
            - Suitable role for remote work
            
            Requirements:
            - Dedicated workspace
            - Reliable internet (min 50 Mbps)
            - Available during core hours (10 AM - 3 PM)
            
            Equipment:
            - Company provides laptop and accessories
            - Monthly internet stipend: $50
            
            Communication:
            - Daily standup on Slack
            - Weekly video team meeting
            - Respond to messages within 2 hours during work hours
            """,
            metadata={"source": "remote_work_policy", "department": "HR"}
        ),
        Document(
            page_content="""
            API Documentation - Authentication
            
            Our API uses OAuth 2.0 for authentication.
            
            Getting Started:
            1. Register your application at developer.company.com
            2. Obtain client_id and client_secret
            3. Request access token
            
            Example Request:
            POST /oauth/token
            {
                "grant_type": "client_credentials",
                "client_id": "your_client_id",
                "client_secret": "your_client_secret"
            }
            
            Response:
            {
                "access_token": "eyJhbGc...",
                "token_type": "Bearer",
                "expires_in": 3600
            }
            
            Use the token in subsequent requests:
            Authorization: Bearer eyJhbGc...
            """,
            metadata={"source": "api_docs", "department": "Engineering", "version": "v2"}
        )
    ]
    
    print(f"📄 Preparing {len(sample_documents)} documents...")
    
    # Optional: Split long documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split documents
    all_splits = []
    for doc in sample_documents:
        splits = text_splitter.split_documents([doc])
        all_splits.extend(splits)
    
    print(f"✂️  Split into {len(all_splits)} chunks")
    
    # Add to vector store
    print("⬆️  Uploading to AstraDB...")
    vector_store.add_documents(all_splits)
    
    print("✅ Successfully populated AstraDB with sample documents!")
    print(f"   Total chunks added: {len(all_splits)}")
    print("\nYou can now run queries against these documents.")
    
    return vector_store


def test_retrieval(vector_store):
    """Test retrieval with a sample query"""
    print("\n" + "="*70)
    print("🧪 Testing retrieval...")
    print("="*70)
    
    test_query = "What are the security policies?"
    print(f"\nQuery: {test_query}")
    
    results = vector_store.similarity_search(test_query, k=2)
    
    print(f"\nFound {len(results)} results:\n")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
        print()


def clear_collection(vector_store):
    """Clear all documents from the collection (use with caution!)"""
    print("⚠️  WARNING: This will delete all documents from the collection!")
    confirmation = input("Type 'DELETE' to confirm: ")
    
    if confirmation == "DELETE":
        vector_store.clear()
        print("✅ Collection cleared")
    else:
        print("❌ Cancelled")


if __name__ == "__main__":
    print("="*70)
    print("AstraDB Population Script")
    print("="*70)
    print("\nOptions:")
    print("1. Populate with sample documents")
    print("2. Test retrieval")
    print("3. Clear collection (WARNING: deletes all data)")
    print("4. Exit")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        vector_store = populate_astradb_with_samples()
        
        # Ask if user wants to test
        test = input("\nWould you like to test retrieval? (y/n): ").strip().lower()
        if test == 'y':
            test_retrieval(vector_store)
    
    elif choice == "2":
        print("\n🔧 Connecting to AstraDB...")
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
        vector_store = AstraDBVectorStore(
            embedding=embeddings,
            collection_name=os.getenv("ASTRA_DB_COLLECTION", "default_collection"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            namespace=os.getenv("ASTRA_DB_KEYSPACE"),
        )
        test_retrieval(vector_store)
    
    elif choice == "3":
        print("\n🔧 Connecting to AstraDB...")
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
        vector_store = AstraDBVectorStore(
            embedding=embeddings,
            collection_name=os.getenv("ASTRA_DB_COLLECTION", "default_collection"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            namespace=os.getenv("ASTRA_DB_KEYSPACE"),
        )
        clear_collection(vector_store)
    
    elif choice == "4":
        print("Goodbye!")
    
    else:
        print("❌ Invalid choice")
