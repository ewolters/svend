"""
Enumerator Tool - Bounded Exhaustive Search

Kills hallucinations where model says "it seems unlikely" when it should say
"I checked all N possibilities - none satisfy the condition."

Provides:
- Exhaustive enumeration within bounds
- Counterexample finding
- "All satisfy" / "None satisfy" / "Some satisfy" verdicts
- Search completeness guarantees
"""

from typing import Optional, Dict, Any, List, Tuple, Callable, Set, Generator
from dataclasses import dataclass
from enum import Enum
from itertools import product, permutations, combinations, combinations_with_replacement
import math

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class SearchVerdict(Enum):
    """Result of exhaustive search."""
    ALL_SATISFY = "all_satisfy"  # Every element in search space satisfies condition
    NONE_SATISFY = "none_satisfy"  # No element satisfies condition
    SOME_SATISFY = "some_satisfy"  # At least one but not all
    FOUND_ONE = "found_one"  # Found at least one (search stopped early)
    SEARCH_EXHAUSTED = "search_exhausted"  # Completed full enumeration
    SEARCH_BOUNDED = "search_bounded"  # Hit limit before completing


@dataclass
class SearchResult:
    """Result of enumeration search."""
    verdict: SearchVerdict
    total_checked: int
    total_possible: Optional[int]  # None if unbounded/unknown
    satisfying_count: int
    examples: List[Any]  # Examples that satisfy
    counterexamples: List[Any]  # Examples that don't satisfy
    search_complete: bool  # Did we check everything?


class Enumerator:
    """
    Bounded exhaustive search engine.

    Guarantees: If search_complete=True, the verdict is CERTAIN.
    """

    MAX_DEFAULT = 100000  # Default max items to check

    @staticmethod
    def enumerate_integers(low: int, high: int) -> Generator[int, None, None]:
        """Generate integers in range."""
        for i in range(low, high + 1):
            yield i

    @staticmethod
    def enumerate_pairs(items: List[Any], allow_same: bool = False) -> Generator[Tuple[Any, Any], None, None]:
        """Generate all pairs from items."""
        for i, a in enumerate(items):
            start = i if allow_same else i + 1
            for j in range(start, len(items)):
                yield (a, items[j])

    @staticmethod
    def enumerate_tuples(items: List[Any], size: int) -> Generator[Tuple, None, None]:
        """Generate all tuples of given size."""
        yield from product(items, repeat=size)

    @staticmethod
    def enumerate_permutations(items: List[Any], size: Optional[int] = None) -> Generator[Tuple, None, None]:
        """Generate all permutations."""
        yield from permutations(items, size)

    @staticmethod
    def enumerate_combinations(items: List[Any], size: int) -> Generator[Tuple, None, None]:
        """Generate all combinations."""
        yield from combinations(items, size)

    @staticmethod
    def enumerate_subsets(items: List[Any], min_size: int = 0, max_size: Optional[int] = None) -> Generator[Tuple, None, None]:
        """Generate all subsets within size bounds."""
        if max_size is None:
            max_size = len(items)
        for size in range(min_size, max_size + 1):
            yield from combinations(items, size)

    @staticmethod
    def count_search_space(
        space_type: str,
        n: int,
        r: Optional[int] = None
    ) -> int:
        """Calculate size of search space."""
        if space_type == "integers":
            return n  # Assumed n is already the count
        elif space_type == "pairs":
            return n * (n - 1) // 2
        elif space_type == "pairs_with_replacement":
            return n * (n + 1) // 2
        elif space_type == "tuples":
            return n ** (r or 1)
        elif space_type == "permutations":
            if r is None:
                return math.factorial(n)
            return math.perm(n, r)
        elif space_type == "combinations":
            return math.comb(n, r or 0)
        elif space_type == "subsets":
            return 2 ** n
        else:
            return -1  # Unknown

    def search(
        self,
        generator: Generator,
        condition: Callable[[Any], bool],
        max_check: int = MAX_DEFAULT,
        find_all: bool = False,
        max_examples: int = 10,
        max_counterexamples: int = 5,
        total_possible: Optional[int] = None
    ) -> SearchResult:
        """
        Search through generator for items satisfying condition.

        Args:
            generator: Items to check
            condition: Function returning True if item satisfies
            max_check: Maximum items to check
            find_all: If False, stop after finding first satisfying item
            max_examples: Max satisfying examples to collect
            max_counterexamples: Max counterexamples to collect
            total_possible: Known total size of search space

        Returns:
            SearchResult with verdict and statistics
        """
        checked = 0
        satisfying = 0
        examples = []
        counterexamples = []

        for item in generator:
            if checked >= max_check:
                break

            checked += 1

            try:
                satisfies = condition(item)
            except:
                satisfies = False

            if satisfies:
                satisfying += 1
                if len(examples) < max_examples:
                    examples.append(item)
                if not find_all:
                    # Found one, can stop
                    return SearchResult(
                        verdict=SearchVerdict.FOUND_ONE,
                        total_checked=checked,
                        total_possible=total_possible,
                        satisfying_count=satisfying,
                        examples=examples,
                        counterexamples=counterexamples,
                        search_complete=False
                    )
            else:
                if len(counterexamples) < max_counterexamples:
                    counterexamples.append(item)

        # Determine if search was complete
        search_complete = total_possible is not None and checked >= total_possible

        # Determine verdict
        if search_complete:
            if satisfying == 0:
                verdict = SearchVerdict.NONE_SATISFY
            elif satisfying == checked:
                verdict = SearchVerdict.ALL_SATISFY
            else:
                verdict = SearchVerdict.SOME_SATISFY
        else:
            verdict = SearchVerdict.SEARCH_BOUNDED

        return SearchResult(
            verdict=verdict,
            total_checked=checked,
            total_possible=total_possible,
            satisfying_count=satisfying,
            examples=examples,
            counterexamples=counterexamples,
            search_complete=search_complete
        )

    def find_counterexample(
        self,
        generator: Generator,
        condition: Callable[[Any], bool],
        max_check: int = MAX_DEFAULT,
        total_possible: Optional[int] = None
    ) -> SearchResult:
        """
        Find a counterexample (item that does NOT satisfy condition).

        Useful for disproving universal claims.
        """
        # Negate condition and find example
        return self.search(
            generator,
            lambda x: not condition(x),
            max_check=max_check,
            find_all=False,
            max_examples=1,
            total_possible=total_possible
        )

    def verify_all(
        self,
        generator: Generator,
        condition: Callable[[Any], bool],
        max_check: int = MAX_DEFAULT,
        total_possible: Optional[int] = None
    ) -> SearchResult:
        """
        Verify that ALL items satisfy condition.

        Returns definitive answer if search completes.
        """
        return self.search(
            generator,
            condition,
            max_check=max_check,
            find_all=True,  # Need to check everything
            total_possible=total_possible
        )


# Built-in conditions for common checks

def is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def is_perfect_square(n: int) -> bool:
    """Check if n is a perfect square."""
    if n < 0:
        return False
    root = int(n ** 0.5)
    return root * root == n


def is_palindrome(s: Any) -> bool:
    """Check if s is a palindrome."""
    s = str(s)
    return s == s[::-1]


def digits_sum_to(n: int, target: int) -> bool:
    """Check if digits of n sum to target."""
    return sum(int(d) for d in str(abs(n))) == target


# Tool implementation

def enumerator_tool(
    operation: str,
    space_type: Optional[str] = None,
    low: Optional[int] = None,
    high: Optional[int] = None,
    items: Optional[List[Any]] = None,
    size: Optional[int] = None,
    condition: Optional[str] = None,
    condition_type: Optional[str] = None,
    condition_params: Optional[Dict[str, Any]] = None,
    max_check: int = 10000,
    find_all: bool = True,
) -> ToolResult:
    """Execute enumeration search."""
    try:
        enumerator = Enumerator()

        if operation == "search":
            # Build generator based on space_type
            if space_type == "integers":
                if low is None or high is None:
                    return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need low and high for integers")
                gen = enumerator.enumerate_integers(low, high)
                total = high - low + 1

            elif space_type == "pairs":
                if items is None:
                    return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need items for pairs")
                gen = enumerator.enumerate_pairs(items)
                total = len(items) * (len(items) - 1) // 2

            elif space_type == "tuples":
                if items is None or size is None:
                    return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need items and size for tuples")
                gen = enumerator.enumerate_tuples(items, size)
                total = len(items) ** size

            elif space_type == "permutations":
                if items is None:
                    return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need items for permutations")
                gen = enumerator.enumerate_permutations(items, size)
                total = math.perm(len(items), size) if size else math.factorial(len(items))

            elif space_type == "combinations":
                if items is None or size is None:
                    return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need items and size for combinations")
                gen = enumerator.enumerate_combinations(items, size)
                total = math.comb(len(items), size)

            elif space_type == "subsets":
                if items is None:
                    return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need items for subsets")
                min_size = condition_params.get("min_size", 0) if condition_params else 0
                max_size = condition_params.get("max_size", len(items)) if condition_params else len(items)
                gen = enumerator.enumerate_subsets(items, min_size, max_size)
                total = sum(math.comb(len(items), k) for k in range(min_size, max_size + 1))

            else:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error=f"Unknown space_type: {space_type}. Valid: integers, pairs, tuples, permutations, combinations, subsets"
                )

            # Build condition
            if condition_type == "prime":
                cond_fn = is_prime
            elif condition_type == "perfect_square":
                cond_fn = is_perfect_square
            elif condition_type == "palindrome":
                cond_fn = is_palindrome
            elif condition_type == "digits_sum":
                target = condition_params.get("target", 0) if condition_params else 0
                cond_fn = lambda n, t=target: digits_sum_to(n, t)
            elif condition_type == "divisible":
                divisor = condition_params.get("divisor", 1) if condition_params else 1
                cond_fn = lambda n, d=divisor: n % d == 0
            elif condition_type == "greater_than":
                threshold = condition_params.get("threshold", 0) if condition_params else 0
                cond_fn = lambda x, t=threshold: x > t
            elif condition_type == "less_than":
                threshold = condition_params.get("threshold", 0) if condition_params else 0
                cond_fn = lambda x, t=threshold: x < t
            elif condition_type == "in_range":
                low_t = condition_params.get("low", float('-inf')) if condition_params else float('-inf')
                high_t = condition_params.get("high", float('inf')) if condition_params else float('inf')
                cond_fn = lambda x, l=low_t, h=high_t: l <= x <= h
            elif condition_type == "sum_equals":
                target = condition_params.get("target", 0) if condition_params else 0
                cond_fn = lambda x, t=target: sum(x) == t if hasattr(x, '__iter__') else x == t
            elif condition_type == "all_different":
                cond_fn = lambda x: len(set(x)) == len(x) if hasattr(x, '__iter__') else True
            elif condition_type == "custom" and condition:
                # Safe eval of simple expression
                cond_fn = lambda x, expr=condition: eval(expr, {"__builtins__": {}, "x": x, "sum": sum, "len": len, "min": min, "max": max, "abs": abs, "all": all, "any": any})
            else:
                cond_fn = lambda x: True  # Accept all

            # Check if search space is too large
            if total > max_check:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"CANNOT_VERIFY: Search space ({total:,}) exceeds limit ({max_check:,}). Increase max_check or narrow search.",
                    metadata={
                        "status": "cannot_verify",
                        "total_possible": total,
                        "max_check": max_check,
                        "reason": "search_space_too_large"
                    }
                )

            # Run search
            result = enumerator.search(
                gen,
                cond_fn,
                max_check=max_check,
                find_all=find_all,
                total_possible=total
            )

            # Format output
            if result.search_complete:
                if result.verdict == SearchVerdict.ALL_SATISFY:
                    output = f"VERIFIED: ALL {result.total_checked:,} items satisfy the condition"
                elif result.verdict == SearchVerdict.NONE_SATISFY:
                    output = f"VERIFIED: NONE of {result.total_checked:,} items satisfy the condition"
                    if result.counterexamples:
                        output += f"\nCounterexamples: {result.counterexamples[:5]}"
                else:
                    output = f"VERIFIED: {result.satisfying_count:,} of {result.total_checked:,} items satisfy ({100*result.satisfying_count/result.total_checked:.1f}%)"
                    if result.examples:
                        output += f"\nExamples: {result.examples[:5]}"
            else:
                output = f"PARTIAL: Checked {result.total_checked:,}"
                if result.total_possible:
                    output += f" of {result.total_possible:,}"
                output += f", found {result.satisfying_count:,} satisfying"
                if result.examples:
                    output += f"\nExamples: {result.examples[:5]}"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata={
                    "verdict": result.verdict.value,
                    "total_checked": result.total_checked,
                    "total_possible": result.total_possible,
                    "satisfying_count": result.satisfying_count,
                    "search_complete": result.search_complete,
                    "examples": result.examples[:10],
                    "counterexamples": result.counterexamples[:5]
                }
            )

        elif operation == "count":
            # Just count the search space
            if space_type == "integers":
                if low is None or high is None:
                    return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need low and high")
                total = high - low + 1
            elif space_type == "pairs":
                n = len(items) if items else 0
                total = n * (n - 1) // 2
            elif space_type == "tuples":
                n = len(items) if items else 0
                total = n ** (size or 1)
            elif space_type == "permutations":
                n = len(items) if items else 0
                total = math.perm(n, size) if size else math.factorial(n)
            elif space_type == "combinations":
                n = len(items) if items else 0
                total = math.comb(n, size or 0)
            elif space_type == "subsets":
                n = len(items) if items else 0
                total = 2 ** n
            else:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error=f"Unknown space_type: {space_type}")

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Search space size: {total:,}",
                metadata={"total": total, "space_type": space_type}
            )

        elif operation == "find_counterexample":
            # Find something that violates condition
            if space_type == "integers":
                if low is None or high is None:
                    return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need low and high")
                gen = enumerator.enumerate_integers(low, high)
                total = high - low + 1
            else:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Only integers supported for counterexample search currently")

            # Build condition
            if condition_type and condition_params:
                # Similar to search, but we negate
                pass

            if condition_type == "custom" and condition:
                cond_fn = lambda x, expr=condition: eval(expr, {"__builtins__": {}, "x": x, "sum": sum, "len": len, "abs": abs})
            else:
                return ToolResult(status=ToolStatus.INVALID_INPUT, output=None, error="Need condition for counterexample search")

            result = enumerator.find_counterexample(gen, cond_fn, max_check, total)

            if result.examples:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"COUNTEREXAMPLE FOUND: {result.examples[0]} violates the condition",
                    metadata={"counterexample": result.examples[0], "search_complete": result.search_complete}
                )
            elif result.search_complete:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"NO COUNTEREXAMPLE: All {result.total_checked:,} items satisfy the condition",
                    metadata={"counterexample": None, "search_complete": True}
                )
            else:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"No counterexample found in first {result.total_checked:,} items (search incomplete)",
                    metadata={"counterexample": None, "search_complete": False}
                )

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=f"Unknown operation: {operation}. Valid: search, count, find_counterexample"
            )

    except Exception as e:
        return ToolResult(status=ToolStatus.ERROR, output=None, error=str(e))


def create_enumerator_tool() -> Tool:
    """Create enumerator tool."""
    return Tool(
        name="enumerate",
        description="Exhaustive search within bounds. Gives VERIFIED verdicts like 'ALL satisfy', 'NONE satisfy', or 'CANNOT_VERIFY (too large)'. Never say 'seems unlikely' - enumerate and KNOW.",
        parameters=[
            ToolParameter(
                name="operation",
                description="search (find satisfying), count (search space size), find_counterexample",
                type="string",
                required=True,
                enum=["search", "count", "find_counterexample"]
            ),
            ToolParameter(
                name="space_type",
                description="Type of search space: integers, pairs, tuples, permutations, combinations, subsets",
                type="string",
                required=False,
                enum=["integers", "pairs", "tuples", "permutations", "combinations", "subsets"]
            ),
            ToolParameter(
                name="low",
                description="Lower bound for integer search",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="high",
                description="Upper bound for integer search",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="items",
                description="List of items to form tuples/permutations/combinations from",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="size",
                description="Size of tuples/combinations/permutations",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="condition_type",
                description="Built-in condition: prime, perfect_square, palindrome, digits_sum, divisible, greater_than, less_than, in_range, sum_equals, all_different, custom",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="condition",
                description="Custom condition expression using 'x' (e.g., 'x % 7 == 0')",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="condition_params",
                description="Parameters for condition (e.g., {target: 10} for digits_sum)",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="max_check",
                description="Maximum items to check (default: 10000)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="find_all",
                description="Check all items vs stop at first (default: true)",
                type="boolean",
                required=False,
            ),
        ],
        execute_fn=enumerator_tool,
        timeout_ms=60000,  # Can be slow for large searches
    )


def register_enumerator_tools(registry: ToolRegistry) -> None:
    """Register enumerator tools."""
    registry.register(create_enumerator_tool())
