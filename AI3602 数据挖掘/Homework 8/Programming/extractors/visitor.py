from abc import ABC, abstractmethod
from tree_sitter import Parser, Node
from typing import List


class Visitor(ABC):
    """Base class for AST feature extractors

    XXX: Do NOT modify this class.
    """

    def __init__(self, parser: Parser):
        self.parser = parser
        self.results: List[str] = []

    def get_results(self) -> List[str]:
        return self.results

    @abstractmethod
    def visit(self, root: Node):
        pass
