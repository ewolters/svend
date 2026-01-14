"""
External API Tools - Wolfram, PubChem, Web Search

Tools that connect to external services for verified data.
These require API keys or network access.
"""

from typing import Optional, Dict, Any, List
import json
import os

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


# ==================== WOLFRAM ALPHA ====================

class WolframAlphaTool:
    """
    WolframAlpha API integration.

    Provides verified factual data, computations, and knowledge queries.
    Requires WOLFRAM_APP_ID environment variable.
    """

    def __init__(self):
        self._client = None
        self.app_id = os.environ.get("WOLFRAM_APP_ID")

    def query(
        self,
        query: str,
        format: str = "short",
    ) -> Dict[str, Any]:
        """
        Query WolframAlpha.

        Args:
            query: Natural language query
            format: "short" (just answer), "full" (detailed), "steps" (with steps)
        """
        if not self.app_id:
            return {
                "success": False,
                "error": "WOLFRAM_APP_ID not set. Get one at https://developer.wolframalpha.com/",
            }

        try:
            import requests

            # Use Short Answers API for quick results
            if format == "short":
                url = "https://api.wolframalpha.com/v1/result"
                params = {
                    "appid": self.app_id,
                    "i": query,
                }
                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    return {
                        "success": True,
                        "query": query,
                        "result": response.text,
                        "format": "short",
                    }
                else:
                    return {
                        "success": False,
                        "error": f"API error: {response.status_code}",
                        "message": response.text,
                    }

            else:
                # Use Full Results API
                url = "https://api.wolframalpha.com/v2/query"
                params = {
                    "appid": self.app_id,
                    "input": query,
                    "format": "plaintext",
                    "output": "json",
                }

                if format == "steps":
                    params["podstate"] = "Step-by-step solution"

                response = requests.get(url, params=params, timeout=15)
                data = response.json()

                if data.get("queryresult", {}).get("success"):
                    pods = data["queryresult"].get("pods", [])
                    results = []

                    for pod in pods:
                        pod_data = {
                            "title": pod.get("title"),
                            "content": [],
                        }
                        for subpod in pod.get("subpods", []):
                            if subpod.get("plaintext"):
                                pod_data["content"].append(subpod["plaintext"])

                        if pod_data["content"]:
                            results.append(pod_data)

                    return {
                        "success": True,
                        "query": query,
                        "results": results,
                        "format": format,
                    }
                else:
                    return {
                        "success": False,
                        "error": "Query failed",
                        "tips": data.get("queryresult", {}).get("tips"),
                    }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}


def wolfram_tool(
    query: str,
    format: Optional[str] = None,
) -> ToolResult:
    """Tool function for WolframAlpha."""
    wolfram = WolframAlphaTool()
    result = wolfram.query(query, format or "short")

    if result.get("success"):
        if result.get("format") == "short":
            output = result["result"]
        else:
            # Format full results
            output_parts = []
            for pod in result.get("results", []):
                output_parts.append(f"**{pod['title']}**")
                output_parts.extend(pod["content"])
                output_parts.append("")
            output = "\n".join(output_parts)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=output,
            metadata=result,
        )
    else:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=result.get("error"),
        )


def create_wolfram_tool() -> Tool:
    """Create the WolframAlpha tool."""
    return Tool(
        name="wolfram",
        description="Query WolframAlpha for verified factual data, computations, and knowledge. Use for: distances, populations, historical dates, scientific data, unit conversions with context, mathematical computations.",
        parameters=[
            ToolParameter(
                name="query",
                description="Natural language query (e.g., 'distance from Earth to Mars', 'population of France')",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="format",
                description="Response format: 'short' (just answer), 'full' (detailed), 'steps' (with steps)",
                type="string",
                required=False,
                enum=["short", "full", "steps"],
            ),
        ],
        execute_fn=wolfram_tool,
        timeout_ms=20000,
    )


# ==================== PUBCHEM ====================

class PubChemTool:
    """
    PubChem database integration.

    Query chemical compounds for properties, structures, and safety data.
    No API key required.
    """

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def search_compound(
        self,
        identifier: str,
        id_type: str = "name",
    ) -> Dict[str, Any]:
        """
        Search for a compound.

        Args:
            identifier: Compound name, formula, CID, or SMILES
            id_type: Type of identifier (name, formula, cid, smiles)
        """
        try:
            import requests

            # Map identifier type to PubChem namespace
            namespace_map = {
                "name": "name",
                "formula": "formula",
                "cid": "cid",
                "smiles": "smiles",
                "inchi": "inchi",
            }
            namespace = namespace_map.get(id_type, "name")

            # Get CID first
            url = f"{self.BASE_URL}/compound/{namespace}/{identifier}/cids/JSON"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return {"success": False, "error": f"Compound not found: {identifier}"}

            data = response.json()
            cid = data["IdentifierList"]["CID"][0]

            return {
                "success": True,
                "cid": cid,
                "identifier": identifier,
            }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_properties(
        self,
        identifier: str,
        properties: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get compound properties.

        Args:
            identifier: Compound name or CID
            properties: List of properties to fetch
        """
        try:
            import requests

            # Default properties
            if properties is None:
                properties = [
                    "MolecularFormula",
                    "MolecularWeight",
                    "CanonicalSMILES",
                    "IUPACName",
                    "XLogP",
                    "TPSA",
                    "HBondDonorCount",
                    "HBondAcceptorCount",
                ]

            props_str = ",".join(properties)

            # Try as name first, then as CID
            for namespace in ["name", "cid"]:
                url = f"{self.BASE_URL}/compound/{namespace}/{identifier}/property/{props_str}/JSON"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    props = data["PropertyTable"]["Properties"][0]

                    return {
                        "success": True,
                        "compound": identifier,
                        "cid": props.get("CID"),
                        "properties": props,
                    }

            return {"success": False, "error": f"Compound not found: {identifier}"}

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_synonyms(
        self,
        identifier: str,
    ) -> Dict[str, Any]:
        """Get compound synonyms/alternative names."""
        try:
            import requests

            url = f"{self.BASE_URL}/compound/name/{identifier}/synonyms/JSON"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return {"success": False, "error": f"Compound not found: {identifier}"}

            data = response.json()
            synonyms = data["InformationList"]["Information"][0]["Synonym"]

            return {
                "success": True,
                "compound": identifier,
                "synonyms": synonyms[:20],  # Limit to 20
                "total": len(synonyms),
            }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_safety(
        self,
        identifier: str,
    ) -> Dict[str, Any]:
        """Get GHS safety information."""
        try:
            import requests

            # First get CID
            search_result = self.search_compound(identifier)
            if not search_result.get("success"):
                return search_result

            cid = search_result["cid"]

            # Get GHS classification
            url = f"{self.BASE_URL}/compound/cid/{cid}/property/GHSClassification/JSON"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                # Parse GHS data
                return {
                    "success": True,
                    "compound": identifier,
                    "cid": cid,
                    "ghs_data": data,
                }

            # If no GHS data, return basic info
            return {
                "success": True,
                "compound": identifier,
                "cid": cid,
                "ghs_data": None,
                "note": "No GHS classification data available",
            }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}


def pubchem_tool(
    operation: str,
    identifier: str,
    properties: Optional[str] = None,
) -> ToolResult:
    """Tool function for PubChem."""
    pubchem = PubChemTool()

    try:
        props_list = json.loads(properties) if properties else None
    except:
        props_list = None

    if operation == "properties":
        result = pubchem.get_properties(identifier, props_list)
    elif operation == "synonyms":
        result = pubchem.get_synonyms(identifier)
    elif operation == "safety":
        result = pubchem.get_safety(identifier)
    elif operation == "search":
        result = pubchem.search_compound(identifier)
    else:
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            output=None,
            error=f"Unknown operation: {operation}",
        )

    if result.get("success"):
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=json.dumps(result, indent=2),
            metadata=result,
        )
    else:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=result.get("error"),
        )


def create_pubchem_tool() -> Tool:
    """Create the PubChem tool."""
    return Tool(
        name="pubchem",
        description="Query PubChem database for chemical compound information. Get molecular properties, SMILES, synonyms, and safety data. No API key required.",
        parameters=[
            ToolParameter(
                name="operation",
                description="Operation: 'properties', 'synonyms', 'safety', 'search'",
                type="string",
                required=True,
                enum=["properties", "synonyms", "safety", "search"],
            ),
            ToolParameter(
                name="identifier",
                description="Compound identifier (name like 'aspirin', or CID like '2244')",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="properties",
                description="JSON array of specific properties to fetch (optional)",
                type="string",
                required=False,
            ),
        ],
        execute_fn=pubchem_tool,
        timeout_ms=15000,
    )


# ==================== WEB SEARCH ====================

class WebSearchTool:
    """
    Web search integration.

    Supports multiple backends (configurable).
    Default: DuckDuckGo (no API key required).
    """

    def search(
        self,
        query: str,
        max_results: int = 5,
        backend: str = "duckduckgo",
    ) -> Dict[str, Any]:
        """
        Search the web.

        Args:
            query: Search query
            max_results: Maximum number of results
            backend: Search backend to use
        """
        if backend == "duckduckgo":
            return self._search_ddg(query, max_results)
        else:
            return {"success": False, "error": f"Unknown backend: {backend}"}

    def _search_ddg(
        self,
        query: str,
        max_results: int,
    ) -> Dict[str, Any]:
        """Search using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })

            return {
                "success": True,
                "query": query,
                "results": formatted,
                "count": len(formatted),
            }

        except ImportError:
            return {
                "success": False,
                "error": "duckduckgo-search library required. Install with: pip install duckduckgo-search",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


def web_search_tool(
    query: str,
    max_results: Optional[int] = None,
) -> ToolResult:
    """Tool function for web search."""
    search = WebSearchTool()
    result = search.search(query, max_results or 5)

    if result.get("success"):
        # Format results nicely
        output_parts = [f"Search results for: {query}\n"]
        for i, r in enumerate(result["results"], 1):
            output_parts.append(f"{i}. {r['title']}")
            output_parts.append(f"   {r['url']}")
            output_parts.append(f"   {r['snippet'][:200]}...")
            output_parts.append("")

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output="\n".join(output_parts),
            metadata=result,
        )
    else:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=result.get("error"),
        )


def create_web_search_tool() -> Tool:
    """Create the web search tool."""
    return Tool(
        name="web_search",
        description="Search the web for current information. Use for recent events, current data, or fact-checking beyond training knowledge cutoff.",
        parameters=[
            ToolParameter(
                name="query",
                description="Search query",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                description="Maximum number of results (default: 5)",
                type="number",
                required=False,
            ),
        ],
        execute_fn=web_search_tool,
        timeout_ms=15000,
    )


# ==================== REGISTRATION ====================

def register_external_tools(registry: ToolRegistry) -> None:
    """Register all external API tools."""
    registry.register(create_wolfram_tool())
    registry.register(create_pubchem_tool())
    registry.register(create_web_search_tool())
