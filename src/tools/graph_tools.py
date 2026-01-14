"""
Graph Tools - Combinatorics and Graph Algorithms

Provides verified solutions for:
- Combinatorics (permutations, combinations, partitions)
- Graph algorithms (shortest path, connectivity, cycles)
- Tree operations (traversals, LCA, diameters)
"""

from typing import Optional, Dict, Any, List, Tuple, Set
from collections import defaultdict, deque
from functools import lru_cache
import heapq
import math

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class Combinatorics:
    """Combinatorial calculations with exact integer arithmetic."""

    @staticmethod
    @lru_cache(maxsize=1000)
    def factorial(n: int) -> int:
        """Cached factorial."""
        if n < 0:
            raise ValueError("Factorial undefined for negative numbers")
        if n <= 1:
            return 1
        return n * Combinatorics.factorial(n - 1)

    @staticmethod
    def permutations(n: int, r: Optional[int] = None) -> int:
        """P(n, r) = n! / (n-r)!"""
        if r is None:
            r = n
        if r > n or r < 0 or n < 0:
            raise ValueError(f"Invalid P({n}, {r})")
        if r == 0:
            return 1
        # Compute directly to avoid large factorial
        result = 1
        for i in range(n, n - r, -1):
            result *= i
        return result

    @staticmethod
    def combinations(n: int, r: int) -> int:
        """C(n, r) = n! / (r! * (n-r)!)"""
        if r > n or r < 0 or n < 0:
            raise ValueError(f"Invalid C({n}, {r})")
        if r == 0 or r == n:
            return 1
        # Use smaller r for efficiency
        r = min(r, n - r)
        result = 1
        for i in range(r):
            result = result * (n - i) // (i + 1)
        return result

    @staticmethod
    def combinations_with_replacement(n: int, r: int) -> int:
        """C(n+r-1, r) - combinations allowing repetition."""
        return Combinatorics.combinations(n + r - 1, r)

    @staticmethod
    def multinomial(n: int, groups: List[int]) -> int:
        """Multinomial coefficient: n! / (k1! * k2! * ... * km!)"""
        if sum(groups) != n:
            raise ValueError("Groups must sum to n")
        result = Combinatorics.factorial(n)
        for k in groups:
            result //= Combinatorics.factorial(k)
        return result

    @staticmethod
    def stirling_second(n: int, k: int) -> int:
        """Stirling number of the second kind: ways to partition n items into k non-empty subsets."""
        if k > n or k < 0:
            return 0
        if k == 0:
            return 1 if n == 0 else 0
        if k == n or k == 1:
            return 1

        # Use recurrence: S(n,k) = k*S(n-1,k) + S(n-1,k-1)
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                dp[i][j] = j * dp[i-1][j] + dp[i-1][j-1]
        return dp[n][k]

    @staticmethod
    def bell_number(n: int) -> int:
        """Bell number: total number of partitions of n items."""
        if n < 0:
            raise ValueError("Bell number undefined for negative n")
        if n == 0:
            return 1

        # Use Bell triangle
        bell = [[0] * (n + 1) for _ in range(n + 1)]
        bell[0][0] = 1
        for i in range(1, n + 1):
            bell[i][0] = bell[i-1][i-1]
            for j in range(1, i + 1):
                bell[i][j] = bell[i-1][j-1] + bell[i][j-1]
        return bell[n][0]

    @staticmethod
    def catalan(n: int) -> int:
        """Catalan number: C_n = C(2n, n) / (n+1)"""
        if n < 0:
            raise ValueError("Catalan number undefined for negative n")
        return Combinatorics.combinations(2 * n, n) // (n + 1)

    @staticmethod
    def derangements(n: int) -> int:
        """Number of permutations with no fixed points."""
        if n < 0:
            raise ValueError("Derangements undefined for negative n")
        if n == 0:
            return 1
        if n == 1:
            return 0

        # D(n) = (n-1) * (D(n-1) + D(n-2))
        d_prev, d_curr = 1, 0
        for i in range(2, n + 1):
            d_next = (i - 1) * (d_curr + d_prev)
            d_prev, d_curr = d_curr, d_next
        return d_curr

    @staticmethod
    def integer_partitions(n: int, max_parts: Optional[int] = None) -> List[List[int]]:
        """Generate all integer partitions of n."""
        if n < 0:
            return []
        if n == 0:
            return [[]]

        partitions = []
        max_parts = max_parts or n

        def generate(remaining: int, max_val: int, current: List[int], parts_left: int):
            if remaining == 0:
                partitions.append(current[:])
                return
            if parts_left == 0:
                return

            for i in range(min(remaining, max_val), 0, -1):
                current.append(i)
                generate(remaining - i, i, current, parts_left - 1)
                current.pop()

        generate(n, n, [], max_parts)
        return partitions

    @staticmethod
    def partition_count(n: int) -> int:
        """Count integer partitions of n (partition function p(n))."""
        if n < 0:
            return 0
        if n == 0:
            return 1

        # Dynamic programming
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                dp[j] += dp[j - i]
        return dp[n]


class Graph:
    """Graph data structure with common algorithms."""

    def __init__(self, directed: bool = False):
        self.directed = directed
        self.adj: Dict[Any, List[Tuple[Any, float]]] = defaultdict(list)
        self.nodes: Set[Any] = set()

    def add_edge(self, u: Any, v: Any, weight: float = 1.0):
        """Add an edge."""
        self.nodes.add(u)
        self.nodes.add(v)
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))

    def add_node(self, u: Any):
        """Add a node without edges."""
        self.nodes.add(u)

    @classmethod
    def from_edges(cls, edges: List[Tuple], directed: bool = False) -> 'Graph':
        """Create graph from edge list."""
        g = cls(directed)
        for edge in edges:
            if len(edge) == 2:
                g.add_edge(edge[0], edge[1])
            else:
                g.add_edge(edge[0], edge[1], edge[2])
        return g

    @classmethod
    def from_adjacency(cls, adj: Dict[Any, List[Any]], directed: bool = False) -> 'Graph':
        """Create graph from adjacency list."""
        g = cls(directed)
        for u, neighbors in adj.items():
            g.add_node(u)
            for v in neighbors:
                if isinstance(v, tuple):
                    g.add_edge(u, v[0], v[1])
                else:
                    g.add_edge(u, v)
        return g

    def bfs(self, start: Any) -> Dict[Any, int]:
        """BFS returning distances from start."""
        if start not in self.nodes:
            return {}

        distances = {start: 0}
        queue = deque([start])

        while queue:
            u = queue.popleft()
            for v, _ in self.adj[u]:
                if v not in distances:
                    distances[v] = distances[u] + 1
                    queue.append(v)

        return distances

    def dfs(self, start: Any) -> List[Any]:
        """DFS returning visit order."""
        if start not in self.nodes:
            return []

        visited = []
        seen = set()

        def visit(u):
            if u in seen:
                return
            seen.add(u)
            visited.append(u)
            for v, _ in self.adj[u]:
                visit(v)

        visit(start)
        return visited

    def dijkstra(self, start: Any) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
        """Dijkstra's algorithm returning distances and predecessors."""
        if start not in self.nodes:
            return {}, {}

        dist = {u: float('inf') for u in self.nodes}
        pred = {u: None for u in self.nodes}
        dist[start] = 0

        pq = [(0, start)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            for v, w in self.adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    pred[v] = u
                    heapq.heappush(pq, (dist[v], v))

        return dist, pred

    def shortest_path(self, start: Any, end: Any) -> Tuple[float, List[Any]]:
        """Find shortest path between two nodes."""
        dist, pred = self.dijkstra(start)

        if end not in dist or dist[end] == float('inf'):
            return float('inf'), []

        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = pred[current]
        path.reverse()

        return dist[end], path

    def connected_components(self) -> List[Set[Any]]:
        """Find all connected components (undirected graph)."""
        visited = set()
        components = []

        for start in self.nodes:
            if start not in visited:
                component = set()
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    if u in visited:
                        continue
                    visited.add(u)
                    component.add(u)
                    for v, _ in self.adj[u]:
                        if v not in visited:
                            queue.append(v)
                components.append(component)

        return components

    def is_connected(self) -> bool:
        """Check if graph is connected."""
        if not self.nodes:
            return True
        return len(self.connected_components()) == 1

    def has_cycle(self) -> bool:
        """Detect if graph has a cycle."""
        visited = set()
        rec_stack = set()

        def dfs_cycle(u, parent=None):
            visited.add(u)
            rec_stack.add(u)

            for v, _ in self.adj[u]:
                if v not in visited:
                    if dfs_cycle(v, u):
                        return True
                elif self.directed:
                    if v in rec_stack:
                        return True
                else:
                    if v != parent:
                        return True

            rec_stack.remove(u)
            return False

        for node in self.nodes:
            if node not in visited:
                if dfs_cycle(node):
                    return True
        return False

    def topological_sort(self) -> List[Any]:
        """Topological sort (directed acyclic graph only)."""
        if not self.directed:
            raise ValueError("Topological sort only for directed graphs")

        in_degree = {u: 0 for u in self.nodes}
        for u in self.nodes:
            for v, _ in self.adj[u]:
                in_degree[v] += 1

        queue = deque([u for u in self.nodes if in_degree[u] == 0])
        result = []

        while queue:
            u = queue.popleft()
            result.append(u)
            for v, _ in self.adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(result) != len(self.nodes):
            raise ValueError("Graph has a cycle - topological sort not possible")

        return result

    def minimum_spanning_tree(self) -> Tuple[float, List[Tuple]]:
        """Prim's algorithm for MST."""
        if not self.nodes:
            return 0, []

        start = next(iter(self.nodes))
        visited = {start}
        edges = []
        total_weight = 0

        # Priority queue: (weight, u, v)
        pq = [(w, start, v) for v, w in self.adj[start]]
        heapq.heapify(pq)

        while pq and len(visited) < len(self.nodes):
            w, u, v = heapq.heappop(pq)
            if v in visited:
                continue

            visited.add(v)
            edges.append((u, v, w))
            total_weight += w

            for next_v, next_w in self.adj[v]:
                if next_v not in visited:
                    heapq.heappush(pq, (next_w, v, next_v))

        return total_weight, edges


# Tool functions

def combinatorics_tool(
    operation: str,
    n: int,
    r: Optional[int] = None,
    groups: Optional[List[int]] = None,
    max_parts: Optional[int] = None,
) -> ToolResult:
    """Execute combinatorics operation."""
    try:
        comb = Combinatorics()

        if operation == "permutations":
            result = comb.permutations(n, r)
            expr = f"P({n}, {r})" if r is not None else f"P({n})"
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"{expr} = {result}",
                metadata={"expression": expr, "result": result}
            )

        elif operation == "combinations":
            if r is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="r is required for combinations")
            result = comb.combinations(n, r)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"C({n}, {r}) = {result}",
                metadata={"expression": f"C({n}, {r})", "result": result}
            )

        elif operation == "combinations_with_replacement":
            if r is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="r is required")
            result = comb.combinations_with_replacement(n, r)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"C({n}+{r}-1, {r}) = {result}",
                metadata={"result": result}
            )

        elif operation == "multinomial":
            if groups is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="groups is required for multinomial")
            result = comb.multinomial(n, groups)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Multinomial({n}; {groups}) = {result}",
                metadata={"result": result}
            )

        elif operation == "stirling":
            if r is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="r (k) is required for Stirling number")
            result = comb.stirling_second(n, r)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"S({n}, {r}) = {result}",
                metadata={"result": result}
            )

        elif operation == "bell":
            result = comb.bell_number(n)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"B({n}) = {result}",
                metadata={"result": result}
            )

        elif operation == "catalan":
            result = comb.catalan(n)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"C_{n} = {result}",
                metadata={"result": result}
            )

        elif operation == "derangements":
            result = comb.derangements(n)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"D({n}) = {result}",
                metadata={"result": result}
            )

        elif operation == "partitions":
            result = comb.integer_partitions(n, max_parts)
            count = len(result)
            # Limit output for large n
            if count <= 20:
                output = f"Partitions of {n}: {result}\nCount: {count}"
            else:
                output = f"Partitions of {n}: {result[:10]}... ({count} total)"
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata={"partitions": result if count <= 100 else result[:100], "count": count}
            )

        elif operation == "partition_count":
            result = comb.partition_count(n)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"p({n}) = {result}",
                metadata={"result": result}
            )

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=f"Unknown operation: {operation}. Valid: permutations, combinations, combinations_with_replacement, multinomial, stirling, bell, catalan, derangements, partitions, partition_count"
            )

    except Exception as e:
        return ToolResult(status=ToolStatus.ERROR, output=None, error=str(e))


def graph_tool(
    operation: str,
    edges: Optional[List[List]] = None,
    adjacency: Optional[Dict[str, List]] = None,
    directed: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> ToolResult:
    """Execute graph algorithm."""
    try:
        # Build graph
        if edges is not None:
            g = Graph.from_edges([tuple(e) for e in edges], directed)
        elif adjacency is not None:
            g = Graph.from_adjacency(adjacency, directed)
        else:
            return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Provide edges or adjacency list")

        if operation == "bfs":
            if start is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="start is required for BFS")
            distances = g.bfs(start)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"BFS distances from {start}: {dict(distances)}",
                metadata={"distances": distances}
            )

        elif operation == "dfs":
            if start is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="start is required for DFS")
            order = g.dfs(start)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"DFS order from {start}: {order}",
                metadata={"order": order}
            )

        elif operation == "shortest_path":
            if start is None or end is None:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="start and end required")
            dist, path = g.shortest_path(start, end)
            if path:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"Shortest path from {start} to {end}: {' -> '.join(map(str, path))} (distance: {dist})",
                    metadata={"distance": dist, "path": path}
                )
            else:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"No path exists from {start} to {end}",
                    metadata={"distance": float('inf'), "path": []}
                )

        elif operation == "connected_components":
            components = g.connected_components()
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Connected components ({len(components)}): {[list(c) for c in components]}",
                metadata={"components": [list(c) for c in components], "count": len(components)}
            )

        elif operation == "is_connected":
            connected = g.is_connected()
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Graph is {'connected' if connected else 'not connected'}",
                metadata={"connected": connected}
            )

        elif operation == "has_cycle":
            has_cycle = g.has_cycle()
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Graph {'has' if has_cycle else 'does not have'} a cycle",
                metadata={"has_cycle": has_cycle}
            )

        elif operation == "topological_sort":
            try:
                order = g.topological_sort()
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"Topological order: {order}",
                    metadata={"order": order}
                )
            except ValueError as e:
                return ToolResult(status=ToolStatus.ERROR, output=None, error=str(e))

        elif operation == "mst":
            weight, edges_list = g.minimum_spanning_tree()
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"MST weight: {weight}, edges: {edges_list}",
                metadata={"total_weight": weight, "edges": edges_list}
            )

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=f"Unknown operation: {operation}. Valid: bfs, dfs, shortest_path, connected_components, is_connected, has_cycle, topological_sort, mst"
            )

    except Exception as e:
        return ToolResult(status=ToolStatus.ERROR, output=None, error=str(e))


def create_combinatorics_tool() -> Tool:
    """Create combinatorics tool."""
    return Tool(
        name="combinatorics",
        description="Calculate combinatorial values: permutations P(n,r), combinations C(n,r), multinomials, Stirling numbers, Bell numbers, Catalan numbers, derangements, and integer partitions.",
        parameters=[
            ToolParameter(
                name="operation",
                description="Operation: permutations, combinations, combinations_with_replacement, multinomial, stirling, bell, catalan, derangements, partitions, partition_count",
                type="string",
                required=True,
                enum=["permutations", "combinations", "combinations_with_replacement", "multinomial", "stirling", "bell", "catalan", "derangements", "partitions", "partition_count"]
            ),
            ToolParameter(
                name="n",
                description="Primary integer (total items, or number to partition)",
                type="number",
                required=True,
            ),
            ToolParameter(
                name="r",
                description="Secondary integer (items to choose, for P/C/Stirling)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="groups",
                description="List of group sizes for multinomial coefficient",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="max_parts",
                description="Maximum number of parts for integer partitions",
                type="number",
                required=False,
            ),
        ],
        execute_fn=combinatorics_tool,
        timeout_ms=10000,
    )


def create_graph_tool() -> Tool:
    """Create graph algorithms tool."""
    return Tool(
        name="graph",
        description="Graph algorithms: BFS, DFS, shortest path (Dijkstra), connected components, cycle detection, topological sort, minimum spanning tree.",
        parameters=[
            ToolParameter(
                name="operation",
                description="Algorithm: bfs, dfs, shortest_path, connected_components, is_connected, has_cycle, topological_sort, mst",
                type="string",
                required=True,
                enum=["bfs", "dfs", "shortest_path", "connected_components", "is_connected", "has_cycle", "topological_sort", "mst"]
            ),
            ToolParameter(
                name="edges",
                description="Edge list: [[u, v], [u, v, weight], ...]. Use this OR adjacency.",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="adjacency",
                description="Adjacency list: {node: [neighbor, ...], ...}. Use this OR edges.",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="directed",
                description="Is graph directed? (default: false)",
                type="boolean",
                required=False,
            ),
            ToolParameter(
                name="start",
                description="Start node for BFS/DFS/shortest_path",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="end",
                description="End node for shortest_path",
                type="string",
                required=False,
            ),
        ],
        execute_fn=graph_tool,
        timeout_ms=15000,
    )


def register_graph_tools(registry: ToolRegistry) -> None:
    """Register graph and combinatorics tools."""
    registry.register(create_combinatorics_tool())
    registry.register(create_graph_tool())
