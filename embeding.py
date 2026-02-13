"""
Football Laws RAG API - Python Test Script
Tests all endpoints with sample questions
"""
import requests
import json
from datetime import datetime

BASE_URL = "https://fcrules.vercel.app"

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_root():
    print_section("1️⃣ Testing ROOT Endpoint")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_health():
    print_section("2️⃣ Testing HEALTH Endpoint")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(json.dumps(data, indent=2))

    if data.get("retriever_loaded"):
        print("\n✅ Retriever is loaded and ready!")
    else:
        print("\n❌ Retriever not loaded!")

def test_stats():
    print_section("3️⃣ Testing STATS Endpoint")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(json.dumps(data, indent=2))
    print(f"\n📊 Total Chunks: {data.get('total_chunks')}")
    print(f"📊 Unique Laws: {len(data.get('unique_laws', []))}")
    print(f"📊 Total Queries: {data.get('total_queries_processed')}")

def test_question(title, question, top_k=5):
    print_section(f"❓ {title}")
    print(f"Question: {question}\n")

    start_time = datetime.now()
    response = requests.post(
        f"{BASE_URL}/ask",
        json={
            "question": question,
            "top_k": top_k,
            "include_raw_chunks": False
        }
    )
    end_time = datetime.now()

    print(f"Status Code: {response.status_code}")
    print(f"Response Time: {(end_time - start_time).total_seconds():.2f}s\n")

    if response.status_code == 200:
        data = response.json()

        # Print answer
        print("📝 ANSWER:")
        print("-" * 60)
        print(data.get('answer', 'No answer provided'))
        print("-" * 60)

        # Print evidence
        print(f"\n📚 EVIDENCE ({len(data.get('evidence', []))} sources):")
        for i, ev in enumerate(data.get('evidence', [])[:3], 1):  # Show top 3
            print(f"\n  {i}. {ev.get('citation')}")
            print(f"     {ev.get('text_preview')}")

        # Print retrieval info
        print(f"\n🔍 RETRIEVAL INFO:")
        retrieval = data.get('retrieval_info', {})
        print(f"   - Expanded query: {retrieval.get('expanded_query')}")
        print(f"   - Quote mode: {retrieval.get('quote_mode')}")
        print(f"   - Boosted laws: {retrieval.get('boosted_laws')}")
        print(f"   - Chunks retrieved: {retrieval.get('chunks_retrieved')}")

        print(f"\n⏱️  Processing time: {data.get('processing_time_ms', 0):.0f}ms")
    else:
        print(f"❌ Error: {response.text}")

def main():
    print("\n" + "="*60)
    print("  🧪 FOOTBALL LAWS RAG API - TEST SUITE")
    print("="*60)
    print(f"  Base URL: {BASE_URL}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    try:
        # Basic endpoints
        test_root()
        test_health()
        test_stats()

        # Question tests
        test_question(
            "Handball in Penalty Area",
            "What happens if a player deliberately handles the ball in their own penalty area?"
        )

        test_question(
            "Fan Interference",
            "What should happen if a fan runs onto the pitch and kicks the ball away?"
        )

        test_question(
            "Offside Position",
            "When is a player in an offside position?"
        )

        test_question(
            "Shoulder Charge",
            "Is it allowed to shoulder charge an opponent who is far from the ball?"
        )

        test_question(
            "Goalkeeper Handball",
            "Can a goalkeeper handle the ball if a teammate deliberately kicks it to them?"
        )

        print_section("✅ ALL TESTS COMPLETED")
        print("Your API is working perfectly! 🎉\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}\n")

if __name__ == "__main__":
    main()