from tree_sitter import Node, Parser
from .visitor import Visitor


class ClassDeclarationVisitor(Visitor):
    def __init__(self, parser: Parser):
        super().__init__(parser)

    def _dfs(self, node: Node):

        # class names are available in "class_declaration" nodes,
        # we check if the current node is a class_declaration
        if node.type == "class_declaration":
            # class_declaration nodes have a child node with the field name "name"
            # this "name" child is an identifier node with the class name
            name_node = node.child_by_field_name("name")

            # Get the text content of the node using the `text` attribute
            # The text content is a byte sequence, we decode it to a string
            # with the `decode()` method
            self.results.append(name_node.text.decode())

        # we continue the depth-first traverse for all children of the current node
        for child in node.children:
            self._dfs(child)

    def visit(self, root: Node):
        self.results = []
        self._dfs(root)
