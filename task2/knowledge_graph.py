"""Knowledge Graph Builder - Static analysis using AST"""

import os
import ast
import warnings
from typing import Dict, List
from collections import defaultdict

warnings.filterwarnings("ignore", category=SyntaxWarning)


class KnowledgeGraphBuilder:
    """Builds a deterministic knowledge graph from Python source files using AST."""

    def __init__(self, root_directory: str):
        self.root = root_directory
        self.nodes: Dict[str, Dict] = {}  # symbol -> metadata
        self.edges: Dict[str, List[tuple]] = defaultdict(list)  # from_sym -> [(to_sym, type)]
        self.file_contents: Dict[str, str] = {}  # filepath -> content
        self.imports_map: Dict[str, List[str]] = {}  # filepath -> [imported_symbols]
        self.py_files: List[str] = []

    def extract_python_files(self) -> List[str]:
        """Get all Python files in the repository."""
        py_files = []
        for root, dirs, files in os.walk(self.root):
            # Skip common unneeded directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'node_modules'}]
            for file in files:
                if file.endswith('.py'):
                    py_files.append(os.path.join(root, file))
        return sorted(py_files)

    def get_relative_path(self, filepath: str) -> str:
        """Get path relative to repository root."""
        return os.path.relpath(filepath, self.root)

    def extract_nodes_from_file(self, filepath: str) -> None:
        """Parse a Python file and extract ClassDef, FunctionDef, AsyncFunctionDef nodes."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            self.file_contents[filepath] = content

            tree = ast.parse(content)
            rel_path = self.get_relative_path(filepath)

            # Extract top-level definitions
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbol_id = f"{rel_path}::{node.name}"
                    self.nodes[symbol_id] = {
                        'type': node.__class__.__name__,
                        'name': node.name,
                        'file': rel_path,
                        'lineno': node.lineno,
                        'col_offset': node.col_offset,
                    }

                    # Extract docstring
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # Get only first line
                        first_line = docstring.split('\n')[0]
                        self.nodes[symbol_id]['docstring'] = first_line

                    # For functions/methods, extract type hints
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.nodes[symbol_id]['signature'] = self._extract_signature(node)

        except Exception:
            pass  # Silently skip files with parse errors

    def _extract_signature(self, func_node: ast.FunctionDef) -> str:
        """Extract function signature with type hints."""
        args = func_node.args
        params = []

        # Regular arguments
        for arg in args.args:
            param = arg.arg
            if arg.annotation:
                annotation_str = self._unparse_annotation(arg.annotation)
                param += f": {annotation_str}"
            params.append(param)

        # Keyword-only arguments
        for arg in args.kwonlyargs:
            param = arg.arg
            if arg.annotation:
                annotation_str = self._unparse_annotation(arg.annotation)
                param += f": {annotation_str}"
            params.append(param)

        return_type = ""
        if func_node.returns:
            return_type = f" -> {self._unparse_annotation(func_node.returns)}"

        return f"def {func_node.name}({', '.join(params)}){return_type}:"

    def _unparse_annotation(self, node: ast.expr) -> str:
        """Convert AST annotation node to string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._unparse_annotation(node.value)
            return f"{value}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            value = self._unparse_annotation(node.value)
            slice_str = self._unparse_annotation(node.slice)
            return f"{value}[{slice_str}]"
        elif isinstance(node, ast.Tuple):
            elts = [self._unparse_annotation(e) for e in node.elts]
            return f"({', '.join(elts)})"
        else:
            return "Any"

    def extract_imports(self, filepath: str) -> None:
        """Extract imports from a file."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            tree = ast.parse(content)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)

            self.imports_map[filepath] = imports
        except Exception:
            pass

    def build_graph(self) -> None:
        """Build the complete knowledge graph."""
        print("[KG] Extracting Python files...")
        self.py_files = self.extract_python_files()

        print(f"[KG] Found {len(self.py_files)} Python files. Extracting nodes...")
        for filepath in self.py_files:
            self.extract_nodes_from_file(filepath)

        print(f"[KG] Extracted {len(self.nodes)} symbols. Extracting imports...")
        for filepath in self.py_files:
            self.extract_imports(filepath)

        print(f"[KG] Knowledge graph built: {len(self.nodes)} nodes")
