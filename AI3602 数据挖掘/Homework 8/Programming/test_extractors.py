# XXX: Do NOT modify this file!

import unittest
from tree_sitter import Node, Parser, Language
from extractors import (
    ClassDeclarationVisitor,
    MethodDeclarationVisitor,
    ObjectCreationVisitor,
    MethodInvocationVisitor,
    Visitor,
)
from typing import List


class ExtractorTestBase(unittest.TestCase):
    def setUp(self):
        self.parser = Parser()
        self.parser.set_language(Language("./parser/my-languages.so", "java"))

    def parse_code(self, code: str) -> Node:
        tree = self.parser.parse(code.encode())

        return tree.root_node

    def run_visitor(self, visitor: Visitor, code: str) -> List[str]:
        root = self.parse_code(code)

        if root.has_error:
            raise RuntimeError("Invalid AST!")

        visitor.visit(root)
        return visitor.get_results()


class TestClassDeclarationVisitor(ExtractorTestBase):
    def test_class_name_simple(self):
        code = """
        public class MyClass {
            public void readText(String file) {
                System.out.println('Hello World.');
            }
        }
        """

        visitor = ClassDeclarationVisitor(self.parser)
        self.assertEqual(self.run_visitor(visitor, code), ["MyClass"])

    def test_class_name_variable_child_num(self):
        code = """
        @override
        class AlsoMyClass {
            public void readText(String file) {
                System.out.println('Hello World.');
            }
        }
        """

        visitor = ClassDeclarationVisitor(self.parser)
        self.assertEqual(self.run_visitor(visitor, code), ["AlsoMyClass"])

    def test_class_name_with_inheritance(self):
        code = """
        public class NotYourClass extends Parent {
            public void readText(String file) {
                System.out.println('Hello World.');
            }
        }
        """

        visitor = ClassDeclarationVisitor(self.parser)
        self.assertEqual(self.run_visitor(visitor, code), ["NotYourClass"])

    def test_class_name_multiple(self):
        code = """
        class MyClass {
            public void readText(String file) {
                System.out.println('Hello World.');
            }
        }
        class AlsoMyClass {
            public void readText(String file) {
                System.out.println('Hello World.');
            }
        }
        """

        visitor = ClassDeclarationVisitor(self.parser)
        self.assertEqual(self.run_visitor(visitor, code), ["MyClass", "AlsoMyClass"])


class TestMethodDeclarationVisitor(ExtractorTestBase):
    def test_get_method_name_single(self):
        code = """
        public class MyClass {
            public void readText(String file) {
                System.out.println('Hello World.');
            }
        }
        """

        visitor = MethodDeclarationVisitor(self.parser)
        self.assertEqual(self.run_visitor(visitor, code), ["readText"])

    def test_method_name_with_field(self):
        code = """
        public class MyClass {
            public int thingy;
            public void createThingy() {
                thingy = 0;
            }
        }
        """

        visitor = MethodDeclarationVisitor(self.parser)
        self.assertEqual(self.run_visitor(visitor, code), ["createThingy"])

    def test_method_name_multi(self):
        code = """
        public class MyClass {
            public void readText(String file) {
                System.out.println('Hello World.');
            }
            public void writeText(String file) {
                System.out.println('Hello World.');
            }
        }
        """

        visitor = MethodDeclarationVisitor(self.parser)
        self.assertEqual(self.run_visitor(visitor, code), ["readText", "writeText"])

    def test_method_name_throw(self):
        code = """
        public class MyClass {
            public void readTextButThrow(String file) throws IOException {
                System.out.println('Hello World.');
            }
        }
        """

        visitor = MethodDeclarationVisitor(self.parser)
        self.assertEqual(self.run_visitor(visitor, code), ["readTextButThrow"])


class TestObjectCreationVisitor(ExtractorTestBase):
    def test_get_object_creation_simple(self):
        code = """
        public class MyClass {
            public void readText(String file) {
                BufferedReader br = new BufferedReader();
            }
        }
        """
        visitor = ObjectCreationVisitor(self.parser)
        expected = ["BufferedReader"]
        self.assertEqual(self.run_visitor(visitor, code), expected)

    def test_object_creation_multi(self):
        code = """
        public class MyClass {
            public void readText(String file) {
                BufferedReader br = new BufferedReader();
                File file = new File();
            }
        }
        """
        visitor = ObjectCreationVisitor(self.parser)
        expected = ["BufferedReader", "File"]
        self.assertEqual(self.run_visitor(visitor, code), expected)

    def test_object_creation_nested(self):
        code = """
        public class MyClass {
            public void readText(String file) {
                BufferedReader br = new BufferedReader(new FileInputStream(file));
            }
        }
        """
        visitor = ObjectCreationVisitor(self.parser)
        expected = ["BufferedReader", "FileInputStream"]
        self.assertEqual(self.run_visitor(visitor, code), expected)

    def test_object_creation_template(self):
        code = """
        public class MyClass {
            public void readText(String file) {
                List<String> list = new ArrayList<String>();
            }
        }
        """
        visitor = ObjectCreationVisitor(self.parser)
        expected = ["ArrayList<String>"]
        self.assertEqual(self.run_visitor(visitor, code), expected)


class TestMethodInvocationVisitor(ExtractorTestBase):
    def test_method_invocation_simple(self):
        code = """
        public class MyClass {
            public void readText(String file) {
                System.out.println('Hello World.');
            }
        }
        """
        visitor = MethodInvocationVisitor(self.parser)
        expected = ["System.out.println"]
        self.assertEqual(self.run_visitor(visitor, code), expected)

    def test_method_invocation_multi(self):
        code = """
        public class MyClass {
            public void readText(String file) {
                System.out.println('Hello World.');
                System.out.println('Hello World.');
                System.exit(0);
            }
        }
        """
        visitor = MethodInvocationVisitor(self.parser)
        expected = ["System.out.println", "System.out.println", "System.exit"]
        self.assertEqual(self.run_visitor(visitor, code), expected)

    def test_method_invocation_var_to_type(self):
        code = """
        public class MyClass {
            public void readText(String file) {
                BufferedReader br = new BufferedReader();
                br.readLine();
            }
        }
        """
        visitor = MethodInvocationVisitor(self.parser)
        expected = ["BufferedReader.readLine"]
        self.assertEqual(self.run_visitor(visitor, code), expected)

    def test_method_invocation_mixed(self):
        code = """
        public class MyClass {
            public void readText(String file) {
            BufferedReader br = new BufferedReader(new FileInputStream(file));
            String line = null;
            while ((line = br.readLine())!= null) {
                System.out.println(line);
            }
            br.close();
            }
        }
        """
        visitor = MethodInvocationVisitor(self.parser)
        expected = [
            "BufferedReader.readLine",
            "System.out.println",
            "BufferedReader.close",
        ]
        self.assertEqual(self.run_visitor(visitor, code), expected)


if __name__ == "__main__":
    unittest.main()
