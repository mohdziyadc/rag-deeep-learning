
import requests
from test_data import SAMPLE_DOCS

def test_ingest():

    print("\n" + "="*60)
    print("TESTING DOCUMENT INGESTION")
    print("="*60)

    doc_ids = []

    for i, doc in enumerate(SAMPLE_DOCS):
        response = requests.post(
            'http://localhost:8000/api/documents/ingest',
            json={
                "title": doc['title'],
                "content": doc['content']
            }
        )

        if response.status_code == 200:
            result = response.json()
            doc_ids.append(result["doc_id"])
            print(f"✅ Ingested: {doc['title']}")
            print(f"   - Doc ID: {result['doc_id']}")
            print(f"   - Chunks: {result['chunks_count']}")
            print(f"   - Is Indexed: {result['is_indexed']}")
        else:
            print(f"❌ Failed: {doc['title']}")
            print(f"   - Error: {response.text}")
    return doc_ids


if __name__ == '__main__':
    test_ingest()