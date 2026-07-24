"""
Product Intelligence and Customer Service Platform — ShopMart Retail
======================================================================
Build this file in 6 phases, one day's skills at a time. It is the Retail
counterpart to loan_origination_assistant.py (Finance) — pick ONE of the two
to implement; both are scaffolded the same way.

Phase 1  (Day 1)   — Secure client · enrichment system prompt · Q&A system prompt
Phase 2  (Day 2)   — ProductRecord schema · parse() with retry · enrichment pipeline
Phase 3  (Day 2)   — ProductConversationManager · multi-turn Q&A session
Phase 4  (Day 3)   — Tool definitions (inventory/price/vendor spec) · manual agentic loop
Phase 5  (Day 3-4) — Chroma-backed RAG over catalogue + vendor specs · Voyage embeddings
Phase 6  (Day 4)   — Enrichment accuracy + Q&A faithfulness evaluation

Run:
    python product_platform.py
"""

# ── Imports (provided) ─────────────────────────────────────────────────────────
import json
import os
import re
import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, PositiveFloat, ValidationError, field_validator

import anthropic

# ── Constants (provided) ───────────────────────────────────────────────────────
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
TOKEN_WARN_THRESHOLD = 30_000       # print a warning once the Q&A session crosses this
TOKEN_COMPACT_THRESHOLD = 60_000    # summarise-and-reset once the Q&A session crosses this
MAX_PARSE_RETRIES = 2
TOP_K_CHUNKS = 3
RAW_CATALOGUE_PATH = Path("data/retail_products.txt")
VENDOR_SPEC_DIR = Path("data/vendor_specs")
EVAL_LOG_PATH = Path("eval_logs/product_platform_v1.jsonl")
CHROMA_COLLECTION_NAME = "shopmart_catalogue"
FALLBACK_RESPONSE = (
    "I don't have that specification on file — I recommend checking with the "
    "vendor or contacting our support team."
)


# ── Mock data (provided — do not modify) ──────────────────────────────────────
# Simulates the three real-time systems the Q&A assistant calls via tool use.

INVENTORY_DB = {
    "SKU-E001": {"available": True,  "quantity": 12, "warehouse": "Whitefield WH"},
    "SKU-A002": {"available": False, "quantity": 0,  "warehouse": "N/A"},
    "SKU-H003": {"available": True,  "quantity": 47, "warehouse": "Hosur Plant"},
}

PRICE_DB = {
    "SKU-E001": {"price_inr": 124_990.0, "discount_pct": 8, "offer_ends": "2026-07-26"},
    "SKU-A002": {"price_inr": 3_495.0,   "discount_pct": 0, "offer_ends": "N/A"},
    "SKU-H003": {"price_inr": 2_199.0,   "discount_pct": 0, "offer_ends": "N/A"},
}

# Keyed by (sku, normalised spec_field) — mirrors what a vendor spec-sheet
# lookup service would return for a single named field.
VENDOR_SPEC_DB = {
    ("SKU-E001", "ram"):               {"field": "RAM", "value": "32GB LPDDR5, soldered — not user-upgradeable", "source": "vendor_sheet"},
    ("SKU-E001", "thunderbolt"):       {"field": "Ports", "value": "2x Thunderbolt 4 (USB-C), 1x USB-C 3.2 Gen 2", "source": "vendor_sheet"},
    ("SKU-E001", "ports"):             {"field": "Ports", "value": "2x Thunderbolt 4 (USB-C), 1x USB-C 3.2 Gen 2", "source": "vendor_sheet"},
    ("SKU-E001", "warranty"):          {"field": "Warranty", "value": "12 months international warranty, India-serviceable via Dell ExpressService", "source": "vendor_sheet"},
    ("SKU-A002", "water_resistance"):  {"field": "Water Resistance", "value": "30 metres (3 ATM) — splash resistant only, not for swimming", "source": "vendor_sheet"},
    ("SKU-A002", "strap_material"):    {"field": "Strap Material", "value": "Genuine leather, brown", "source": "vendor_sheet"},
    ("SKU-H003", "isi_certification"): {"field": "ISI Certification", "value": "IS 2347:2017 certified", "source": "vendor_sheet"},
    ("SKU-H003", "warranty"):          {"field": "Warranty", "value": "24 months on the cooker body, 12 months on gasket/safety valve", "source": "vendor_sheet"},
}

# 6-turn test conversation about the Dell XPS 15 (SKU-E001). The final turn
# asks about a spec that is NOT in the vendor sheet, to exercise the fallback.
TEST_CONVERSATION = [
    "Hi, I'm looking at the Dell XPS 15, SKU-E001. Can I upgrade the RAM myself later?",
    "Does it support Thunderbolt?",
    "What's the warranty like in India?",
    "Is it in stock right now?",
    "What's the current price, with any active discount?",
    "One last thing — does it come in a silver colour option?",
]

# Ground-truth ProductRecord field values for 5 products (Phase 6 enrichment eval)
ENRICHMENT_GOLDEN_SET = [
    {"sku": "SKU-E001", "ground_truth": {
        "brand": "Dell", "category": "electronics", "price_inr": 124_990.0, "in_stock": True}},
    {"sku": "SKU-E002", "ground_truth": {
        "brand": "boAt", "category": "electronics", "price_inr": 1_499.0, "in_stock": True}},
    {"sku": "SKU-A002", "ground_truth": {
        "brand": "Titan", "category": "apparel", "price_inr": 3_495.0, "in_stock": False}},
    {"sku": "SKU-H003", "ground_truth": {
        "brand": "Prestige", "category": "homeware", "price_inr": 2_199.0, "in_stock": True}},
    {"sku": "SKU-B001", "ground_truth": {
        "brand": "Himalaya Herbals", "category": "beauty", "price_inr": 165.0, "in_stock": True}},
]

# 5 customer queries with the expected grounding source (Phase 6 Q&A eval)
QA_GOLDEN_SET = [
    {"query": "What is the RAM capacity of the Dell XPS 15?", "sku": "SKU-E001", "expected_source": "catalogue"},
    {"query": "Does the Dell XPS 15 support Thunderbolt?", "sku": "SKU-E001", "expected_source": "vendor_spec"},
    {"query": "Is the Titan Kairos watch safe to wear while swimming?", "sku": "SKU-A002", "expected_source": "vendor_spec"},
    {"query": "Is the Prestige Svachh cooker ISI certified?", "sku": "SKU-H003", "expected_source": "vendor_spec"},
    {"query": "Does the Dell XPS 15 come in a silver colour option?", "sku": "SKU-E001", "expected_source": "fallback"},
]


# ── Raw catalogue loader (provided) ────────────────────────────────────────────

def load_raw_products(path: Path = RAW_CATALOGUE_PATH) -> dict[str, str]:
    """Parse data/retail_products.txt into {sku: raw_description_text}.

    The file uses '### SKU-XXX' headers to delimit each vendor's raw
    submission. This is plain file parsing (not a taught skill) — provided
    so Phase 2 can focus on the extraction prompt and parse/retry loop.
    """
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"^### (SKU-[A-Z0-9-]+)\s*$", text, flags=re.MULTILINE)[1:]
    return {
        sku: desc.strip()
        for sku, desc in zip(blocks[0::2], blocks[1::2])
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Secure Foundation (Day 1 skills)
# ═══════════════════════════════════════════════════════════════════════════════

def make_client() -> anthropic.Anthropic:
    """Initialise the shared Anthropic client used by both the enrichment
    pipeline (batch) and the Q&A assistant (real-time).

    TODO:
    - Call load_dotenv() to pick up .env
    - Read ANTHROPIC_API_KEY with os.environ.get()
    - Raise EnvironmentError with a descriptive message if it is absent
    - Return anthropic.Anthropic() — no api_key= argument; SDK reads env automatically
    """
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key is None:
        raise EnvironmentError("ANTHROPIC_API_KEY not found!")
    return anthropic.Anthropic(api_key=api_key)

# Write the enrichment prompt here. Applied per-product in run_enrichment_pipeline().
ENRICHMENT_SYSTEM = """
You are a ShopMart Product Data Specialist responsible for extracting and enriching structured catalogue data from product descriptions, listings, marketing copy, and other source text.

Rules:

1. Role
- Extract product attributes and populate a ProductRecord object.
- Use only information present in the source text and reasonable contextual inference permitted by these instructions.

2. Category Inference
- If category or subcategory is not explicitly stated, infer the most likely category and subcategory from the product name, description, specifications, brand context, and other available clues.
- Only infer when there is sufficient evidence in the source text.
- If multiple categories are equally plausible and no reliable determination can be made, return null.

3. Price Normalization
- Normalize all prices to a plain INR floating-point number.
- Support formats including but not limited to:
  - "₹1,299"
  - "Rs 1299"
  - "INR 1,299.00"
  - "1299 rupees"
  - Prices written in words, e.g.:
    - "one thousand two hundred ninety-nine rupees" → 1299.0
    - "one lakh ten thousand rupees" → 110000.0
- Remove currency symbols, commas, and formatting.
- Return the numeric INR value only.

4. No Fabrication
- Never invent product specifications, features, dimensions, materials, technical details, ratings, brand information, or other attributes that are not present in the source text.
- If a field cannot be determined from the source text or reliable contextual inference, return null.
- Prefer null over guessing.
- Do not create values solely because they are common for similar products.

5. Output Contract
- Return only valid JSON.
- The JSON must conform exactly to the ProductRecord schema.
- Do not include explanations, markdown, comments, code fences, reasoning, confidence scores, or additional keys.
- Ensure all field values are schema-compliant and JSON-serializable.

Your response must be a single valid JSON object matching the ProductRecord schema and nothing else.
"""

# Write the Q&A system prompt here. {product_context} is filled in per-turn in
# Phase 4/5 with retrieved catalogue + vendor-spec chunks.
QA_SYSTEM = """
You are ShopMart's knowledgeable product advisor. Your job is to help retail customers by answering product questions using only the information provided in the retrieved catalogue data and vendor specifications.

Rules:
1. Grounding
   - Answer exclusively from the product information contained in the provided product context.
   - Do not use outside knowledge, assumptions, or general product expertise.
   - If the requested specification, feature, or detail is not explicitly present in the product context, respond with exactly:
     "{fallback}"

2. Tone
   - Be warm, friendly, professional, and customer-focused.
   - Provide clear and concise answers appropriate for retail shoppers.

3. Accuracy and Safety
   - Never invent, infer, or guess facts.
   - Never fabricate warranty periods, compatibility claims, certification statuses, performance metrics, or product capabilities.
   - Only state information that is directly supported by the product context.

4. Response Style
   - Answer the customer's question directly.
   - When appropriate, summarize the relevant product details from the context.
   - If multiple products are present in the context, clearly identify which product the information comes from.

Relevant product context:
{{product_context}}
""".format(fallback=FALLBACK_RESPONSE)



# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Structured Catalogue Enrichment (Day 2 skills)
# ═══════════════════════════════════════════════════════════════════════════════

class ProductRecord(BaseModel):
    """Validated catalogue record produced by the enrichment pipeline.

    TODO (Phase 2):
    - Replace each `Any` placeholder below with the correct type — `Any` is
      just a Pydantic-safe stand-in so this class can be imported before
      Phase 2 is implemented
    - Add a @field_validator for price_inr checking it is > 0
    - Remember: in Pydantic v2, @classmethod must appear ABOVE @field_validator
    """

    sku:               str   # str
    name:               str   # str
    brand:              Optional[str]   # Optional[str]
    category:           Literal["electronics","apparel","homeware","beauty","grocery","sports","other"]   # Literal["electronics","apparel","homeware","beauty","grocery","sports","other"]
    subcategory:        str   # str
    price_inr:          float   # float — must be > 0
    mrp_inr:            Optional[float]   # Optional[float] — original price if discounted
    key_features:       list[str]   # list[str] — 3-6 bullet points
    specifications:     dict[str,str]   # dict[str, str] — e.g. {"RAM": "16GB"}
    in_stock:           bool   # bool
    warranty_months:    Optional[int]   # Optional[int]
    care_instructions:  Optional[str]   # Optional[str] — relevant for apparel/homeware

    @field_validator("price_inr")
    @classmethod
    def validate_price_inr(cls, value):
        if value <= 0:
            raise ValueError("price_inr must be positive")
        return value  # Pydantic v2: a validator MUST return the value, or the field becomes None


def extract_product_record(
    client: anthropic.Anthropic,
    sku: str,
    raw_description: str,
) -> ProductRecord:
    """Extract and validate a ProductRecord from one raw vendor description.

    Uses client.messages.parse() and retries on ValidationError.

    TODO (Phase 2):
    - Build a messages list: a single user turn containing the sku and
      raw_description, asking Claude to extract all ProductRecord fields
    - Call client.messages.parse(model, max_tokens, system=ENRICHMENT_SYSTEM,
      messages=messages, output_format=ProductRecord)
    - Return response.parsed_output on success
    - On ValidationError: append the assistant response and error details, then retry
    - After MAX_PARSE_RETRIES attempts, re-raise the last ValidationError
    """
    messages = [{"role":"user","content":f"sku: {sku} {raw_description} extract all ProductRecord fields"}]
    for i in range(MAX_PARSE_RETRIES + 1):
        try:
            response = client.messages.parse(model=MODEL, max_tokens=1500, system=ENRICHMENT_SYSTEM, messages=messages, output_format=ProductRecord)
            return response.parsed_output, i  # return the validated ProductRecord, not the raw response
        except ValidationError as e:
            if i == MAX_PARSE_RETRIES:
                raise
            messages.append({"role":"user","content":f"ERROR: {str(e)}"})
            
    
def run_enrichment_pipeline(
    client: anthropic.Anthropic,
    raw_products: Optional[dict[str, str]] = None,
) -> tuple[list[ProductRecord], dict]:
    """Run extract_product_record() over every raw vendor description.
    TODO (Phase 2):
    - Default raw_products to load_raw_products() when not provided
    - For each (sku, raw_description):
        * call extract_product_record(); on success append to a results list
        * on repeated ValidationError, log the failure (sku + error) instead
          of raising — this pipeline must not crash on one bad record
        * track a retry count per product (extract_product_record can return
          it, or you can count attempts here)
    - Print a final summary table: total processed, succeeded, failed, retried
    - Return (list[ProductRecord], summary_dict) where summary_dict has
      keys: "succeeded", "failed" (list of skus), "retried" (dict sku->count)
    """
    if raw_products is None:
        raw_products = load_raw_products()
    results = []
    succeeded = 0
    failed = 0
    retried = 0
    list_failed = []
    retry_dict = {}
    for sku, raw_description in raw_products.items():
        try:
            record, retries = extract_product_record(client,sku,raw_description)
            results.append(record)
            succeeded += 1
            if retries > 0:
                retry_dict[sku]=retries
                retried +=1
            print("Enriched SKU " + sku)
        except ValidationError as e:
            print(f"ERROR: {sku} - {e}")
            failed += 1
            list_failed.append(sku)
    print("----- SUMMARY -----")
    print(f"processed: {len(raw_products):>8}")
    print(f"succeeded: {succeeded:>8}")
    print(f"failed:    {failed:>8}")
    for bad_sku in list_failed:
        print(f"  - {bad_sku}")
    print(f"retried:   {retried:>8}")
    summary_dict = {
        "succeeded": succeeded,
        "failed": list_failed,
        "retried": retry_dict,
    }
    return results, summary_dict
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Multi-Turn Q&A Conversation (Day 2 skills)
# ═══════════════════════════════════════════════════════════════════════════════

class ProductConversationManager:
    """Maintains full message history for a multi-turn Q&A session, and
    tracks which product(s) have been discussed so far.

    TODO (Phase 3):
    - __init__(self, client, system): store client and system; initialise
      self.messages = [] and self.products_discussed: set[str] = set()
    - send(self, user_message, skus_mentioned=()) -> str:
        * update self.products_discussed with any skus_mentioned
        * append {"role": "user", "content": user_message}
        * call client.messages.create(model, max_tokens, system, messages)
        * append {"role": "assistant", "content": reply}   ← full list, not just .text
        * return the text reply
    - token_count(self) -> int:
        * use client.messages.count_tokens(model, system, messages)
        * return result.input_tokens
        * print a warning if this exceeds TOKEN_WARN_THRESHOLD
    - summarise_and_reset(self) -> str:
        * build a history_text string from self.messages
        * ask Claude to summarise in <=150 words, preserving which SKUs were
          discussed and any open questions
        * reset self.messages to [{"role":"user","content":"[Summary]\\n{summary}"}]
        * return the summary string
        * triggered by the caller once token_count() exceeds TOKEN_COMPACT_THRESHOLD
    """

    def __init__(self, client: anthropic.Anthropic, system: str) -> None:
        self.client = client
        self.system = system
        self.messages = []
        self.products_discussed: set[str] = set()
    def send(self, user_message: str, skus_mentioned: tuple[str, ...] = ()) -> str:
        for sku in skus_mentioned:
            self.products_discussed.add(sku)
        self.messages.append({"role":"user", "content":user_message})
        reply = self.client.messages.create(model="claude-sonnet-4-6", max_tokens=1500, system=self.system,messages=self.messages)
        self.messages.append({"role":"assistant","content":reply.content})
        return reply.content[0].text
    
    def token_count(self) -> int:
        result = self.client.messages.count_tokens(model="claude-sonnet-4-6", system=self.system, messages=self.messages)
        if result.input_tokens > TOKEN_WARN_THRESHOLD:
            print("WARNING: Token threshold exceeded")
        return result.input_tokens
     
    def summarise_and_reset(self) -> str:
        self.messages.append({"role":"user", "content":"Summarize this conversation in <=150 words, preserving which SKUs were discussed and any open questions"})
        response = self.client.messages.create(model="claude-sonnet-4-6",max_tokens=1500, system=self.system,messages=self.messages)
        self.messages = [{"role":"user","content":f"[Summary]\\n {response.content[0].text}"}]

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Real-Time Tool Integration (Day 3 skills)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Mock tool implementations (provided — matches the mock DBs above) ──────────

def _check_inventory(sku: str) -> dict:
    """Look up live stock for a SKU. Returns an error dict for unknown SKUs."""
    result = INVENTORY_DB.get(sku)
    if not result:
        return {"error": f"SKU {sku} not found in inventory system."}
    return result


def _get_current_price(sku: str) -> dict:
    """Look up the live (possibly discounted) price for a SKU."""
    result = PRICE_DB.get(sku)
    if not result:
        return {"error": f"SKU {sku} not found in pricing system."}
    return result


def _fetch_vendor_spec(sku: str, spec_field: str) -> dict:
    """Look up one named spec field from the vendor spec sheet for a SKU."""
    key = (sku, spec_field.strip().lower().replace(" ", "_"))
    result = VENDOR_SPEC_DB.get(key)
    if not result:
        return {"field": spec_field, "value": None, "source": "not_found"}
    return result


TOOL_FN_MAP = {
    "check_inventory":    _check_inventory,
    "get_current_price":  _get_current_price,
    "fetch_vendor_spec":  _fetch_vendor_spec,
}


def build_qa_tools() -> list[dict]:
    """Return the list of tool definitions passed to client.messages.create().

    TODO (Phase 4):
    Define three tools as dicts with "name", "description", and "input_schema".

    Tool 1 — check_inventory
      input:  sku (string)
      output: {"available": bool, "quantity": int, "warehouse": str}

    Tool 2 — get_current_price
      input:  sku (string)
      output: {"price_inr": float, "discount_pct": int, "offer_ends": str}

    Tool 3 — fetch_vendor_spec
      input:  sku (string), spec_field (string) — e.g. "ram", "water_resistance"
      output: {"field": str, "value": str|null, "source": str}

    Remember:
    - Each input_schema needs "type": "object", "properties": {...}, "required": [...],
      "additionalProperties": False
    - Descriptions should explain WHEN to call each tool: check_inventory/
      get_current_price are for live, daily-changing data; fetch_vendor_spec
      is for a spec missing from the retrieved catalogue context
    """
    return [
        {
            "name": "check_inventory",
            "description": (
                "Look up LIVE stock availability for a product by its SKU. Call this "
                "whenever the customer asks whether an item is in stock, how many units "
                "are available, or which warehouse it ships from. Inventory changes "
                "daily, so always call this tool rather than trusting the catalogue."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "The product SKU, e.g. 'SKU-E001'."},
                },
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_current_price",
            "description": (
                "Look up the LIVE selling price and any active discount for a product by "
                "its SKU. Call this whenever the customer asks about price, cost, "
                "discounts, or current offers. Pricing and promotions change daily, so "
                "always call this tool rather than quoting a price from the catalogue."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "The product SKU, e.g. 'SKU-E001'."},
                },
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
        {
            "name": "fetch_vendor_spec",
            "description": (
                "Fetch a single named specification field from the vendor's spec sheet "
                "for a product. Call this ONLY when a specific spec the customer asked "
                "about is missing from the retrieved product context — for example "
                "'ram', 'ports', 'thunderbolt', 'water_resistance', 'warranty', or "
                "'isi_certification'. Do not use it for live stock or pricing."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "The product SKU, e.g. 'SKU-E001'."},
                    "spec_field": {
                        "type": "string",
                        "description": (
                            "The spec field to look up, e.g. 'ram', 'ports', "
                            "'water_resistance', 'warranty'."
                        ),
                    },
                },
                "required": ["sku", "spec_field"],
                "additionalProperties": False,
            },
        },
    ]


def run_qa_agentic_loop(
    client: anthropic.Anthropic,
    conversation_history: list[dict],
    tools: list[dict],
) -> list[dict]:
    """Run the manual agentic loop for the Q&A assistant: Claude calls tools,
    you execute them, loop until end_turn.

    TODO (Phase 4):
    - Start with messages = conversation_history (make a copy to be safe)
    - while True:
        * call client.messages.create(model, max_tokens, system=QA_SYSTEM,
          tools=tools, messages=messages)
        * if response.stop_reason == "end_turn": break
        * append {"role": "assistant", "content": response.content}   ← full list
        * for each tool_use block in response.content:
            - print the tool name and input
            - look up the function in TOOL_FN_MAP, call it with **block.input
            - set is_error = "error" in result
            - append tool_result to tool_results list
        * append {"role": "user", "content": tool_results}
    - Return the final messages list

    Key mistakes to avoid (same as the Finance case study):
    - Break on "end_turn", NOT on "tool_use"
    - Append response.content (the list), NOT response.content[0].text
    - Tool result content must be json.dumps(result) — a string, not a dict
    """
    messages = list(conversation_history)  # copy so we don't mutate the caller's list
    # QA_SYSTEM still carries the {product_context} slot; in the agentic-loop path the
    # retrieved context is supplied via the conversation messages, so blank the slot.
    system = QA_SYSTEM.format(product_context="") if "{product_context}" in QA_SYSTEM else QA_SYSTEM

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=system,
            tools=tools,
            messages=messages,
        )
        # Append the full content list (text + any tool_use blocks), never just .text
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  [tool] {block.name}({json.dumps(block.input)})")
            result = TOOL_FN_MAP[block.name](**block.input)
            print(f"         source -> {json.dumps(result)}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),          # must be a string, not a dict
                "is_error": "error" in result,
            })

        if not tool_results:
            # Non-end_turn stop with no tool calls (e.g. max_tokens/refusal) — stop looping
            break

        messages.append({"role": "user", "content": tool_results})

    return messages


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — RAG over Product Catalogue and Vendor Specs (Day 3-4 skills)
# ═══════════════════════════════════════════════════════════════════════════════

def _embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed texts with OpenAI (text-embedding-3-small), read from OPENAI_API_KEY.

    `input_type` ('document' for indexing, 'query' for retrieval) is kept for
    call-site symmetry with providers that distinguish the two roles; OpenAI's
    embeddings do not, so it is accepted and ignored here."""
    import openai  # lazy so Phases 1-4 run without OPENAI_API_KEY / the package

    client = openai.OpenAI()  # reads OPENAI_API_KEY from the environment
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [item.embedding for item in response.data]


def _product_to_document(rec: ProductRecord) -> str:
    """Flatten a ProductRecord into a single embeddable text document."""
    lines = [
        f"{rec.name} ({rec.brand or 'unbranded'})",
        f"Category: {rec.category} / {rec.subcategory}",
        f"Price: INR {rec.price_inr:.0f}",
    ]
    if rec.key_features:
        lines.append("Key features: " + "; ".join(rec.key_features))
    if rec.specifications:
        lines.append("Specifications: " + "; ".join(f"{k}: {v}" for k, v in rec.specifications.items()))
    if rec.warranty_months:
        lines.append(f"Warranty: {rec.warranty_months} months")
    if rec.care_instructions:
        lines.append(f"Care: {rec.care_instructions}")
    return "\n".join(lines)


def _sku_from_spec(text: str) -> Optional[str]:
    """Pull the SKU out of a vendor spec-sheet header, e.g. '... (SKU-E001)'."""
    match = re.search(r"\((SKU-[A-Z0-9-]+)\)", text)
    return match.group(1) if match else None


def build_product_index(records: list[ProductRecord]) -> object:
    """Build a Chroma collection over enriched products + vendor spec sheets.

    TODO (Phase 5):
    - import chromadb and voyageai lazily inside this function (so Phases 1-4
      run without those keys/packages configured)
    - Start a Chroma client: chromadb.PersistentClient(path="./chroma_data")
      (or EphemeralClient() for an in-memory index during development)
    - get_or_create_collection(CHROMA_COLLECTION_NAME) — pass an embedding_function
      that wraps voyageai.Client().embed(texts, model="voyage-3", input_type=...)
      or precompute embeddings yourself and pass them via collection.add(embeddings=...)
    - For each ProductRecord: build one text document (name + brand + category +
      key_features + specifications), and collection.add(
          ids=[sku], documents=[text], metadatas=[{"sku":.., "category":.., "brand":..}]
      )
    - Also read every *.md file under VENDOR_SPEC_DIR, chunk if needed, and
      collection.add(...) each with metadata {"sku": <sku>, "source": "vendor_spec"}
      (map filename -> sku, e.g. via a small dict or filename convention)
    - Return the populated collection
    """
    import chromadb  # lazy import — Phases 1-4 run without chromadb installed

    collection = chromadb.EphemeralClient().get_or_create_collection(CHROMA_COLLECTION_NAME)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    # One document per enriched catalogue record.
    category_by_sku = {rec.sku: rec.category for rec in records}
    for rec in records:
        ids.append(rec.sku)
        documents.append(_product_to_document(rec))
        metadatas.append({
            "sku": rec.sku,
            "category": rec.category,
            "brand": rec.brand or "unknown",   # Chroma metadata can't be None
            "source": "catalogue",
        })

    # One document per vendor spec sheet; SKU parsed from the header, category
    # borrowed from the matching record so category filtering covers specs too.
    for spec_path in sorted(VENDOR_SPEC_DIR.glob("*.md")):
        spec_text = spec_path.read_text(encoding="utf-8")
        sku = _sku_from_spec(spec_text)
        ids.append(f"{sku}-vendor-spec" if sku else spec_path.stem)
        documents.append(spec_text)
        metadatas.append({
            "sku": sku or "unknown",
            "category": category_by_sku.get(sku, "unknown"),
            "source": "vendor_spec",
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=_embed_texts(documents, input_type="document"),
        metadatas=metadatas,
    )
    print(f"  Indexed {len(records)} products + {len(ids) - len(records)} "
          f"vendor spec sheets into '{CHROMA_COLLECTION_NAME}'.")
    return collection


def retrieve_product_context(
    query: str,
    collection: object,
    category_filter: Optional[str] = None,
    top_k: int = TOP_K_CHUNKS,
) -> str:
    """Retrieve the top-k catalogue/vendor-spec chunks most relevant to query.

    TODO (Phase 5):
    - Embed the query with the same Voyage model used in build_product_index()
    - Call collection.query(query_embeddings=[...], n_results=top_k,
      where={"category": category_filter} if category_filter else None)
      — this is the metadata filter: apply it only when the customer has
      already specified a category (e.g. "I'm looking at laptops")
    - Format each result as a `[Product Context: SKU-XXX]` block followed by
      its text, joined by double newlines
    - Return the formatted string

    This string replaces {product_context} in QA_SYSTEM and is also where
    `"citations": {"enabled": true}` should be wired in on the message block
    that carries this context, so answers cite verifiable sources.
    """
    query_embedding = _embed_texts([query], input_type="query")[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        # Apply the category filter only when the customer has specified one.
        where={"category": category_filter} if category_filter else None,
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    if not documents:
        return ""

    blocks = []
    for doc, meta in zip(documents, metadatas):
        sku = (meta or {}).get("sku", "unknown")
        blocks.append(f"[Product Context: {sku}]\n{doc}")
    return "\n\n".join(blocks)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Evaluation (Day 4 skills)
# ═══════════════════════════════════════════════════════════════════════════════

FAITHFULNESS_JUDGE_SYSTEM = """
You are a strict evaluation judge for ShopMart's product Q&A assistant. You are given
the PRODUCT CONTEXT that was available to the assistant and the ASSISTANT ANSWER it
produced. Judge only whether the answer is grounded in the provided context.

Scoring (integer 1-5):
- 5 = every factual claim in the answer is directly supported by the product context.
- 4 = fully supported, with minor harmless phrasing not drawn from the context.
- 3 = mostly supported but includes at least one claim that cannot be verified from the context.
- 2 = several claims are unsupported by the context.
- 1 = the answer is largely fabricated / contradicts or ignores the context.

Rules:
- An answer that correctly declines because the requested detail is not in the context
  (e.g. the defined fallback response) is FULLY faithful — score it 5.
- In your reasoning, explicitly flag any warranty period, compatibility claim, or
  certification status that the answer states but that is absent from the context.
- Do not reward or penalise tone, helpfulness, or completeness — only factual grounding.

Return ONLY a single JSON object and nothing else:
{"score": <int 1-5>, "reasoning": "<one or two sentences>"}
"""


def judge_faithfulness(client: anthropic.Anthropic, context: str, answer: str) -> dict:
    """Score a Q&A answer for faithfulness to the retrieved product context.

    TODO (Phase 6):
    - Build a user prompt combining context and answer
    - Call client.messages.create() with FAITHFULNESS_JUDGE_SYSTEM
    - Strip markdown fences from the response text before json.loads()
      Hint: re.search(r"```(?:json)?\\s*(\\{[\\s\\S]*?\\})\\s*```", text)
    - Return the parsed dict {"score": int, "reasoning": str}
    - On any parse error, return {"score": 0, "reasoning": "parse error: <raw text>"}
    """
    user_prompt = (
        "PRODUCT CONTEXT:\n"
        f"{context or '(no product context was retrieved)'}\n\n"
        "ASSISTANT ANSWER:\n"
        f"{answer}\n\n"
        "Score how well the assistant answer is supported by the product context above."
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=FAITHFULNESS_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    try:
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        payload = match.group(1) if match else text
        data = json.loads(payload)
        return {"score": int(data["score"]), "reasoning": str(data.get("reasoning", ""))}
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return {"score": 0, "reasoning": f"parse error: {text}"}


_SPEC_STOPWORDS = {
    "and", "the", "with", "for", "not", "only", "inch", "inches", "size",
    "colour", "color", "built", "into", "your", "per", "via", "plus",
}


def _values_match(field: str, actual: Any, expected: Any) -> bool:
    """Field-aware comparison of an extracted value against ground truth."""
    if actual is None:
        return False
    if field == "price_inr":
        try:
            return abs(float(actual) - float(expected)) < 1.0
        except (TypeError, ValueError):
            return False
    if field == "brand":
        a, e = str(actual).strip().lower(), str(expected).strip().lower()
        return a == e or e in a or a in e   # "Himalaya" vs "Himalaya Herbals"
    if isinstance(expected, bool):
        return actual == expected
    if isinstance(expected, str):
        return str(actual).strip().lower() == expected.strip().lower()
    return actual == expected


def _is_hallucinated(value: str, raw_lower: str) -> bool:
    """Conservative hallucination flag: a spec/feature counts as hallucinated only
    if NONE of its meaningful tokens appear anywhere in the raw vendor description."""
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    meaningful = [t for t in tokens if len(t) >= 3 and t not in _SPEC_STOPWORDS]
    if not meaningful:
        return False
    return not any(t in raw_lower for t in meaningful)


def evaluate_enrichment_accuracy(
    records: list[ProductRecord],
    golden_set: list[dict] = ENRICHMENT_GOLDEN_SET,
) -> dict:
    """Compare enriched records against ENRICHMENT_GOLDEN_SET ground truth.

    TODO (Phase 6):
    - Index records by sku
    - For each golden entry: compare each ground_truth field against the
      matching ProductRecord field; count correct / total across all entries
    - Flag any record where a specification appears in `specifications` or
      `key_features` but cannot be found anywhere in the raw vendor
      description (hallucinated spec)
    - Return {"field_accuracy": float, "hallucinated_specs": int,
      "flagged_skus": list[str]}
    """
    by_sku = {rec.sku: rec for rec in records}
    raw_products = load_raw_products()

    correct = total = hallucinated = 0
    flagged: set[str] = set()

    for entry in golden_set:
        sku = entry["sku"]
        ground_truth = entry["ground_truth"]
        rec = by_sku.get(sku)
        if rec is None:
            total += len(ground_truth)   # missing record → every field counts as wrong
            continue

        for field, expected in ground_truth.items():
            total += 1
            if _values_match(field, getattr(rec, field, None), expected):
                correct += 1

        raw_lower = raw_products.get(sku, "").lower()
        for text in list(rec.specifications.values()) + rec.key_features:
            if _is_hallucinated(text, raw_lower):
                hallucinated += 1
                flagged.add(sku)

    return {
        "field_accuracy": correct / total if total else 0.0,
        "hallucinated_specs": hallucinated,
        "flagged_skus": sorted(flagged),
    }


def log_eval_result(record: dict) -> None:
    """Append one evaluation result as a JSON line to EVAL_LOG_PATH.

    TODO (Phase 6):
    - Add "timestamp": datetime.datetime.utcnow().isoformat() to the record
    - EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    - Open EVAL_LOG_PATH in append mode and write json.dumps(record) + "\\n"
    """
    entry = {**record, "timestamp": datetime.datetime.utcnow().isoformat()}
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _final_answer(messages: list[dict]) -> tuple[str, list[str]]:
    """Return (answer_text, sources) from a finished agentic-loop history.

    `sources` are the exact cited_text snippets Claude drew on, extracted from the
    citations attached to its answer (citations were enabled on the retrieved
    product-context document block). Facts that came from a live tool are surfaced
    separately by the `[tool] ... source ->` lines printed during the loop.
    """
    for msg in reversed(messages):
        if msg["role"] != "assistant":
            continue
        texts: list[str] = []
        sources: list[str] = []
        for block in msg["content"]:
            if getattr(block, "type", None) != "text":
                continue
            texts.append(block.text)
            for citation in (getattr(block, "citations", None) or []):
                cited = (getattr(citation, "cited_text", "") or "").strip()
                if not cited:
                    continue
                snippet = cited if len(cited) <= 200 else cited[:200] + "…"
                if snippet not in sources:
                    sources.append(snippet)
        if texts:
            return " ".join(t.strip() for t in texts).strip(), sources
    return "", []


def _answer_query(
    client: anthropic.Anthropic,
    query: str,
    context: str,
    tools: list[dict],
) -> tuple[str, list[str]]:
    """Drive one grounded Q&A turn through the agentic loop and return the answer.

    When context is present it is passed as a citations-enabled document block so the
    answer can cite the retrieved source; the loop can still call tools for any spec
    that is missing from the context. Returns (answer_text, cited_sources).
    """
    if context:
        user_content = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": context},
                "title": "Product Context",
                "citations": {"enabled": True},
            },
            {"type": "text", "text": query},
        ]
    else:
        user_content = query
    messages = run_qa_agentic_loop(client, [{"role": "user", "content": user_content}], tools)
    return _final_answer(messages)


def run_evaluation(
    client: anthropic.Anthropic,
    records: list[ProductRecord],
    collection: object,
    tools: list[dict],
) -> None:
    """Run both evaluation tracks and print the combined report.

    TODO (Phase 6):
    ENRICHMENT ACCURACY
    - result = evaluate_enrichment_accuracy(records)
    - log_eval_result({"track": "enrichment", **result})
    - Print "ENRICHMENT ACCURACY (5 golden products)" + the scores

    Q&A FAITHFULNESS
    - For each entry in QA_GOLDEN_SET:
        1. category_filter = None (or infer one from the query)
        2. context = retrieve_product_context(entry["query"], collection, category_filter)
        3. Drive a short Q&A turn through run_qa_agentic_loop() (or a direct
           client.messages.create() call with QA_SYSTEM.format(product_context=context))
           to get the assistant's answer
        4. faith = judge_faithfulness(client, context, answer)
        5. log_eval_result({"track": "qa", "query": entry["query"], **faith})
    - Print "Q&A FAITHFULNESS (5 golden queries)" + average faithfulness score

    - Print an overall PASS/FAIL summary (choose and document your own thresholds)
    """
    # ── ENRICHMENT ACCURACY ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ENRICHMENT ACCURACY (5 golden products)")
    print("=" * 60)
    enrichment = evaluate_enrichment_accuracy(records)
    log_eval_result({"track": "enrichment", **enrichment})
    print(f"  Field accuracy    : {enrichment['field_accuracy']:.0%}")
    print(f"  Hallucinated specs: {enrichment['hallucinated_specs']}")
    print(f"  Flagged SKUs      : {', '.join(enrichment['flagged_skus']) or 'none'}")

    # ── Q&A FAITHFULNESS ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Q&A FAITHFULNESS (5 golden queries)")
    print("=" * 60)
    scores: list[int] = []
    for entry in QA_GOLDEN_SET:
        query = entry["query"]
        context = ""
        if collection is not None:
            context = retrieve_product_context(query, collection, category_filter=None)
        answer, sources = _answer_query(client, query, context, tools)
        faith = judge_faithfulness(client, context, answer)
        scores.append(faith["score"])
        log_eval_result({
            "track": "qa",
            "query": query,
            "expected_source": entry["expected_source"],
            "answer": answer,
            "sources": sources,
            **faith,
        })
        print(f"\n  Q: {query}")
        print(f"  A: {answer}")
        if sources:
            print("  Sources cited:")
            for s in sources:
                print(f"    • {s}")
        print(f"  Faithfulness: {faith['score']}/5 — {faith['reasoning']}")

    avg_faith = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  Average faithfulness: {avg_faith:.1f} / 5")

    # ── OVERALL PASS/FAIL (thresholds chosen for this case study) ─────────────
    enrichment_pass = enrichment["field_accuracy"] >= 0.80 and enrichment["hallucinated_specs"] == 0
    qa_pass = avg_faith >= 4.0
    print("\n" + "=" * 60)
    print(f"  ENRICHMENT : {'PASS' if enrichment_pass else 'FAIL'}  "
          f"(threshold: field accuracy >= 80% and 0 hallucinated specs)")
    print(f"  Q&A        : {'PASS' if qa_pass else 'FAIL'}  "
          f"(threshold: avg faithfulness >= 4.0 / 5)")
    print(f"  OVERALL    : {'PASS' if enrichment_pass and qa_pass else 'FAIL'}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION — wire all phases together
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Phase 1: Initialise ────────────────────────────────────────────────────
    client = make_client()
    tools = build_qa_tools()        # Phase 4 — safe to call empty list until then

    # ── Phase 2: Enrich the full catalogue ────────────────────────────────────
    records, summary = run_enrichment_pipeline(client)
    print(f"\n  Enrichment summary: {summary}")

    # ── Phase 5: Build the RAG index ──────────────────────────────────────────
    # Degrades gracefully: if VOYAGE_API_KEY / chromadb aren't configured, the rest
    # of the run (enrichment + conversation) still works — retrieval is just skipped.
    collection = None
    try:
        collection = build_product_index(records)
    except Exception as e:  # noqa: BLE001 — keep Phases 1-4 runnable without RAG deps
        print(f"\n  [RAG index unavailable — continuing without retrieval] {e}")

    # ── Run the test Q&A conversation end-to-end ──────────────────────────────
    print(f"\n{'='*60}")
    print("Running: Dell XPS 15 multi-turn Q&A test conversation")
    print("=" * 60)

    # Drive the conversation through the Phase 4 agentic loop so the assistant can
    # ground answers in retrieved context (Phase 5) AND call the live tools (Phase 4).
    # The bare ProductConversationManager path always returns the fallback here: it
    # is built with an empty product_context and passes no tools, so the grounding
    # rule leaves it nothing to answer from. We thread `messages` across turns to
    # keep the multi-turn history, injecting each turn's retrieved context as a
    # citations-enabled document block.
    messages: list[dict] = []
    for turn in TEST_CONVERSATION:
        context = ""
        if collection is not None:
            context = retrieve_product_context(turn, collection, category_filter="electronics")
        if context:
            user_content: Any = [
                {
                    "type": "document",
                    "source": {"type": "text", "media_type": "text/plain", "data": context},
                    "title": "Product Context",
                    "citations": {"enabled": True},
                },
                {"type": "text", "text": turn},
            ]
        else:
            user_content = turn  # no RAG context — the tools still cover specs/stock/price
        messages.append({"role": "user", "content": user_content})
        messages = run_qa_agentic_loop(client, messages, tools)
        answer, sources = _final_answer(messages)
        print(f"\n  Customer: {turn}\n  Assistant: {answer}")
        if sources:
            print("  Sources cited:")
            for s in sources:
                print(f"    • {s}")

    # ── Phase 6: Run full evaluation ──────────────────────────────────────────
    run_evaluation(client, records, collection, tools)


if __name__ == "__main__":
    main()
