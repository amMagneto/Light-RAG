import chromadb

client = chromadb.PersistentClient(path="./backend/chroma_db")
collections = client.list_collections()
print("Collections:", collections)

for col in collections:
    c = client.get_collection(col.name)
    print(f"{col.name}: {c.count()} chunks")
    print(c.peek(3))  # Show first 3 chunks
