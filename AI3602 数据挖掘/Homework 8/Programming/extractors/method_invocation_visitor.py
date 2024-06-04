from tree_sitter import Node, Parser
from .visitor import Visitor
from typing import Dict


class MethodInvocationVisitor(Visitor):
    def __init__(self, parser: Parser):
        super().__init__(parser)
        self.identifier2type: Dict[str, str] = {}

    def _dfs_collect_identifiers(self, node: Node):
        """
        This method traverses the AST, visits all local variable declarations,
        collects the variable identifiers and their types, and
        stores them in self.identifier2type.
        """

        # locate local variable declarations
        if node.type == "local_variable_declaration":
            # get types and variable names, names are stored in "declarator" nodes
            type_node = node.child_by_field_name("type")
            declarator_nodes = node.children_by_field_name("declarator")

            # one local variable declaration can have multiple declarators
            # iterate over all declarators and store the variable names and types
            for decl_node in declarator_nodes:
                name_node = decl_node.child_by_field_name("name")
                self.identifier2type[name_node.text.decode()] = type_node.text.decode()

        for child in node.children:
            self._dfs_collect_identifiers(child)

    def _dfs_method_invocation(self, node: Node):

        # TODO (Task 1)
        # Complete the _dfs_method_invocation method to
        # extract method invocations from the AST.
        #
        # Hints:
        # 1. Method invocations are in "method_invocation" nodes.
        #    They (typically) have the form `object.methodName()`.
        # 2. If the object of a method invocation is a variable,
        #    you should map the variable to its type.
        #    Use `self.identifier2type` to get the type of a variable.
        # 3. Store the method invocations in `self.results`.

        if node.type == "method_invocation":
            name_node = node.child_by_field_name("name")
            method_name = name_node.text.decode()
            object_node = node.child_by_field_name("object")
            object_name = object_node.text.decode()
            if object_name in self.identifier2type:
                    self.results.append(f"{self.identifier2type[object_name]}.{method_name}")
            else:
                self.results.append(f"{object_name}.{method_name}")

        # End of TODO

        for child in node.children:
            self._dfs_method_invocation(child)

    def visit(self, root: Node):
        self.results = []
        self.identifier2type = {}

        # We traverse the AST twice
        # 1. In the first pass, we collect all the identifiers and their types
        #    and store them in self.identifier2type
        # 2. In the second pass, we extract method invocations
        self._dfs_collect_identifiers(root)
        self._dfs_method_invocation(root)
