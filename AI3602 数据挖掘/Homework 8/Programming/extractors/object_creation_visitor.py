from tree_sitter import Node, Parser
from .visitor import Visitor


class ObjectCreationVisitor(Visitor):
    def __init__(self, parser: Parser):
        super().__init__(parser)

    def _dfs(self, node: Node):
        # TODO (Task 1)
        # Complete the _dfs method to extract object creations from the AST.
        #
        # Hints:
        # 1. Object creations are in "object_creation_expression" nodes.
        #    They (typically) have the form `new ClassName()`.
        #    Node types can be accessed using `node.type` (a string).
        # 2. Store the method names in `self.results`.

        if node.type == "object_creation_expression":
            name_node = node.child_by_field_name("type")
            self.results.append(name_node.text.decode())

        # End of TODO

        for child in node.children:
            self._dfs(child)

    def visit(self, root: Node):
        self.results = []
        self._dfs(root)
