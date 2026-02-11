#!/usr/bin/env python3
"""
Neo4j Knowledge Graph Builder for Financial Planning Documents
Uses Groq API (Llama 3.3 70B) to extract structured data.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# specific imports
from docx import Document
from neo4j import GraphDatabase
from groq import Groq, RateLimitError

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# Constants for Rate Limiting (Groq Free Tier)
# Groq free tier often limits Total Tokens Per Minute (TPM). 
# A 30s delay is conservative to avoid 429 errors on large docs.
DELAY_BETWEEN_DOCS_SECONDS = 30 

class Neo4jGraphBuilder:
    """Builds Neo4j knowledge graph from financial documents using Groq"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
            
        self.client = Groq(api_key=GROQ_API_KEY)
        
    def close(self):
        self.driver.close()
    
    def read_docx(self, filepath: str) -> str:
        """Extract text from Word document"""
        try:
            doc = Document(filepath)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return ""
    
    def extract_entities_with_groq(self, document_text: str) -> Dict[str, Any]:
        """Use Groq (Llama 3.3) to extract structured entities"""
        
        # Llama 3 prompt optimized for JSON extraction
        system_prompt = "You are a specialized financial data extraction AI. You output ONLY valid JSON."
        
        user_prompt = f"""
Analyze this financial planning document and extract structured information into the specific JSON format below.

Output ONLY valid JSON. Do not include markdown formatting like ```json ... ```.

REQUIRED JSON STRUCTURE:
{{
  "clients": [
    {{
      "name": "Full Name",
      "dob": "DD/MM/YYYY or null",
      "age": number or null,
      "occupation": "string or null",
      "employer": "string or null",
      "income": number or null,
      "health_notes": "string or null",
      "marital_status": "string or null"
    }}
  ],
  "dependants": [
    {{
      "name": "string",
      "age": number or null,
      "school_type": "string or null",
      "notes": "string or null"
    }}
  ],
  "assets": {{
    "properties": [
      {{
        "type": "string",
        "value": number or null,
        "address": "string or null",
        "mortgage_amount": number or null,
        "mortgage_lender": "string or null",
        "mortgage_rate": number or null,
        "mortgage_end_date": "string or null"
      }}
    ],
    "pensions": [
      {{
        "type": "string",
        "provider": "string or null",
        "value": number or null,
        "contribution_amount": number or null,
        "contribution_frequency": "string or null",
        "owner": "string"
      }}
    ],
    "investments": [
      {{
        "type": "string",
        "value": number or null,
        "contribution_amount": number or null,
        "allocation": "string or null",
        "owner": "string"
      }}
    ]
  }},
  "liabilities": [
    {{
      "type": "string",
      "amount": number or null,
      "lender": "string or null",
      "rate": number or null
    }}
  ],
  "protection": [
    {{
      "type": "string",
      "provider": "string or null",
      "cover_amount": number or null,
      "monthly_premium": number or null,
      "status": "string"
    }}
  ],
  "goals": {{
    "retirement": {{
      "target_age": number or null,
      "target_income": number or null,
      "lifestyle_notes": "string or null"
    }},
    "education": {{
      "target_amount_per_child": number or null,
      "notes": "string or null"
    }},
    "other_goals": [
      {{
        "description": "string",
        "target_date": "string or null",
        "estimated_cost": number or null
      }}
    ]
  }},
  "tax_info": {{
    "total_household_income": number or null,
    "estimated_iht_liability": number or null,
    "tax_bracket": "string or null"
  }},
  "recommendations": [
    {{
      "category": "string",
      "priority": "string",
      "description": "string"
    }}
  ],
  "adviser": "string or null",
  "document_type": "string"
}}

DOCUMENT TEXT:
{document_text}
"""
        
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                # JSON mode ensures valid structure
                response_format={"type": "json_object"} 
            )
            
            json_text = completion.choices[0].message.content
            return json.loads(json_text)

        except RateLimitError as e:
            print(f"⚠️ Groq Rate Limit Hit: {e}")
            print("   (You may need to increase the DELAY_BETWEEN_DOCS_SECONDS constant)")
            return {}
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {}
    
    def create_constraints(self):
        """Create uniqueness constraints in Neo4j"""
        constraints = [
            "CREATE CONSTRAINT client_name IF NOT EXISTS FOR (c:Client) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT dependant_id IF NOT EXISTS FOR (d:Dependant) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT property_id IF NOT EXISTS FOR (p:Property) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT pension_id IF NOT EXISTS FOR (p:Pension) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT investment_id IF NOT EXISTS FOR (i:Investment) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT adviser_name IF NOT EXISTS FOR (a:Adviser) REQUIRE a.name IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception:
                    pass
    
    def build_graph_from_entities(self, entities: Dict[str, Any], document_name: str):
        """Build Neo4j graph from extracted entities"""
        # (This logic remains standard, condensed for brevity but fully functional)
        if not entities: return

        with self.driver.session() as session:
            # Adviser
            if entities.get('adviser'):
                session.run("MERGE (a:Adviser {name: $name})", name=entities['adviser'])
            
            # Clients
            clients = entities.get('clients', [])
            if not clients: return

            primary_client_name = clients[0].get('name', f"Unknown_{document_name}")

            for client in clients:
                props = {k: v for k, v in client.items() if v is not None}
                props['source_document'] = document_name
                if 'name' not in props: props['name'] = primary_client_name
                
                session.run("""
                    MERGE (c:Client {name: $name})
                    SET c += $props
                    """, name=props['name'], props=props)
                
                if entities.get('adviser'):
                    session.run("""
                        MATCH (c:Client {name: $c_name}), (a:Adviser {name: $a_name})
                        MERGE (a)-[:ADVISES]->(c)
                    """, c_name=props['name'], a_name=entities['adviser'])

            # Dependants
            for idx, dep in enumerate(entities.get('dependants', [])):
                dep_id = f"{primary_client_name}_dep_{idx}"
                session.run("""
                    MATCH (c:Client {name: $p_name})
                    MERGE (d:Dependant {id: $id})
                    SET d += $props
                    MERGE (c)-[:PARENT_OF]->(d)
                """, p_name=primary_client_name, id=dep_id, props=dep)

            # Assets (Properties, Pensions, Investments)
            assets = entities.get('assets', {})
            
            for idx, prop in enumerate(assets.get('properties', [])):
                prop_id = f"{primary_client_name}_prop_{idx}"
                session.run("""
                    MATCH (c:Client {name: $c_name})
                    MERGE (p:Property {id: $id})
                    SET p += $props
                    MERGE (c)-[:OWNS]->(p)
                """, c_name=primary_client_name, id=prop_id, props=prop)

            for idx, pen in enumerate(assets.get('pensions', [])):
                owner = pen.get('owner', primary_client_name)
                pen_id = f"{owner}_pen_{idx}"
                session.run("""
                    MERGE (c:Client {name: $owner})
                    MERGE (p:Pension {id: $id})
                    SET p += $props
                    MERGE (c)-[:HAS_ACCOUNT]->(p)
                """, owner=owner, id=pen_id, props=pen)

            for idx, inv in enumerate(assets.get('investments', [])):
                owner = inv.get('owner', primary_client_name)
                inv_id = f"{owner}_inv_{idx}"
                session.run("""
                    MERGE (c:Client {name: $owner})
                    MERGE (i:Investment {id: $id})
                    SET i += $props
                    MERGE (c)-[:HAS_ACCOUNT]->(i)
                """, owner=owner, id=inv_id, props=inv)

            # Goals
            goals = entities.get('goals', {})
            if goals.get('retirement'):
                session.run("""
                    MATCH (c:Client {name: $name})
                    MERGE (g:Goal {id: $id})
                    SET g.type='Retirement', g += $props
                    MERGE (c)-[:HAS_GOAL]->(g)
                """, name=primary_client_name, id=f"{primary_client_name}_ret_goal", props=goals['retirement'])
    
    def process_all_documents(self, documents_dir: str):
        """Process all Word documents in directory with rate limiting"""
        documents_path = Path(documents_dir)
        docx_files = list(documents_path.glob("*.docx"))
        
        print(f"\nFound {len(docx_files)} documents to process")
        print(f"Using Groq Model: llama-3.3-70b-versatile")
        print(f"Rate Limiting: Sleeping {DELAY_BETWEEN_DOCS_SECONDS}s between docs")
        
        self.create_constraints()
        
        successful = 0
        failed = 0
        
        for i, filepath in enumerate(docx_files, 1):
            print(f"\n[{i}/{len(docx_files)}] Processing: {filepath.name}")
            
            try:
                # 1. Read
                text = self.read_docx(str(filepath))
                if not text:
                    print("   ⚠️ Empty document or read error")
                    failed += 1
                    continue
                print(f"   ↳ Length: {len(text)} chars")

                # 2. Extract
                print("   ↳ Extracting with Groq...")
                entities = self.extract_entities_with_groq(text)
                
                if entities:
                    # 3. Build Graph
                    self.build_graph_from_entities(entities, filepath.stem)
                    print("   ✓ Graph updated")
                    successful += 1
                else:
                    print("   ❌ Extraction failed (empty response)")
                    failed += 1

            except Exception as e:
                print(f"   ❌ Critical error: {e}")
                failed += 1
            
            # Rate Limit Sleep (Skip on last item)
            if i < len(docx_files):
                print(f"   zzz Sleeping {DELAY_BETWEEN_DOCS_SECONDS}s to respect Groq limits...")
                time.sleep(DELAY_BETWEEN_DOCS_SECONDS)
        
        print(f"\n{'='*60}")
        print(f"Complete! Success: {successful} | Failed: {failed}")
        print(f"{'='*60}")

def main():
    # Check env vars
    if not GROQ_API_KEY:
        print("❌ Error: GROQ_API_KEY not found in .env")
        return
    
    # Locate docs
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == 'utils' else script_dir
    documents_dir = project_root / 'sources' / 'proactive_agent'
    
    if not documents_dir.exists():
        print(f"❌ Error: Documents directory not found: {documents_dir}")
        return
    
    # Run
    builder = Neo4jGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        builder.process_all_documents(str(documents_dir))
    finally:
        builder.close()

if __name__ == "__main__":
    main()