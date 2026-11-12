from tavily import TavilyClient

from app.config import TAVILY_API_KEY


client = None


def web_search(query: str) -> dict:
    global client
    if client is None:
        if not TAVILY_API_KEY:
            raise ValueError("Missing TAVILY_API_KEY")
        client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(query=query, search_depth="basic", max_results=5)
    return {
        "query": query,
        "results": [{"title": item.get("title", ""), "url": item.get("url", ""),
                     "content": item.get("content", "")}
                    for item in response.get("results", [])],
    }
