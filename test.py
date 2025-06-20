from weathergraph import graph

while True:
    query = input("You: ")
    if query.lower() in ("exit", "quit"):
        break
    result = graph.invoke({"input": query})
    print("🤖:", result["output"])
