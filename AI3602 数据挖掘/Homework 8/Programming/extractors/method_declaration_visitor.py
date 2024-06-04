from tree_sitter import Parser, Node
from .visitor import Visitor


class MethodDeclarationVisitor(Visitor):
    def __init__(self, parser: Parser):
        super().__init__(parser)

    def _dfs(self, node: Node):
        # TODO (Task 1)
        # Complete the _dfs method to extract method names from the AST.
        #
        # Hints:
        # 1. Method names can be found in "method_declaration" nodes.
        #    Node types can be accessed using `node.type` (a string).
        # 2. Store the method names in `self.results`.

        if node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            self.results.append(name_node.text.decode())

        # End of TODO

        for child in node.children:
            self._dfs(child)

    def visit(self, root: Node):
        self.results = []
        self._dfs(root)
