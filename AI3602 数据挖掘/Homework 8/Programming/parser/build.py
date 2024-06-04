from tree_sitter import Language

Language.build_library(
    # Store the library in the `build` directory
    "my-languages.so",
    # Include one or more languages
    [
        "tree-sitter-java",
        # 'tree-sitter-python',
    ],
)
