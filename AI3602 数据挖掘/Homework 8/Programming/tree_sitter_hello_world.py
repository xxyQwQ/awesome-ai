from tree_sitter import Language, Parser
from utils import pprint_tree


def main():
    parser = Parser()
    parser.set_language(Language("./parser/my-languages.so", "java"))
    code = """
    public class MyClass {
        public static void MyMethod() {
            System.out.println("Hello world!");
        }
    }
    """

    tree = parser.parse(code.encode())
    root = tree.root_node

    pprint_tree(root)

    if root.has_error:
        raise ValueError("original code is invalid")
    else:
        print("You are all set!")


if __name__ == "__main__":
    main()
