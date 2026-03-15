"""Context Architect - Intelligent context assembly"""

import os
import ast
import re
from typing import Dict, Optional, Set
from knowledge_graph import KnowledgeGraphBuilder

FILE_SEP_SYMBOL = "<|file_sep|>"
FILE_COMPOSE_FORMAT = "{file_sep}{file_name}\n{file_content}"


def clean_code(content: str, preserve_types: bool = True) -> str:
    """
    Strip noise: debug output, trailing whitespace, but PRESERVE type hints and docstrings.

    Args:
        content: Code to clean
        preserve_types: If True, keep type hint comments and docstrings
    """
    lines = []
    in_docstring = False
    docstring_quote = None

    for line in content.split('\n'):
        # Remove trailing whitespace
        line = line.rstrip()
        stripped = line.lstrip()

        # Track docstrings (triple quotes)
        if '"""' in line or "'''" in line:
            quote = '"""' if '"""' in line else "'''"
            if in_docstring and quote == docstring_quote:
                in_docstring = False
                docstring_quote = None
                lines.append(line)
                continue
            elif not in_docstring:
                in_docstring = True
                docstring_quote = quote
                lines.append(line)
                continue

        # Keep docstring content
        if in_docstring:
            lines.append(line)
            continue

        # Skip debug-specific comments (but keep type hints)
        if stripped.startswith('#'):
            # Keep type hint comments
            if preserve_types and ('type:' in line or 'noqa' in line or 'pylint' in line):
                lines.append(line)
                continue
            if any(x in stripped.upper() for x in ['TODO', 'FIXME', 'DEBUG', 'HACK', 'XXX']):
                continue
            # Skip other comments (but be cautious)
            if not stripped.startswith('#!'):
                continue

        lines.append(line)

    return '\n'.join(lines)


def get_signature_only(symbol_id: str, kg: KnowledgeGraphBuilder) -> Optional[str]:
    """Get only signature and first docstring line for distant dependencies."""
    if symbol_id not in kg.nodes:
        return None

    node = kg.nodes[symbol_id]
    sig = node.get('signature', f"def {node['name']}():")
    doc = node.get('docstring', '')

    result = sig
    if doc:
        result += f"\n    \"\"\"{doc}\"\"\""

    return result


class ContextArchitect:
    """Intelligently ranks and assembles context using heuristic rules."""

    def __init__(self, root_directory: str, max_tokens: int = 16000, mellum_budget: int = 8000):
        self.root = root_directory
        self.max_tokens = max_tokens
        self.mellum_budget = mellum_budget
        self.kg = KnowledgeGraphBuilder(root_directory)
        self.kg.build_graph()

    def count_tokens(self, text: str) -> int:
        """
        More accurate token count based on Python code patterns.

        Uses regex to split on:
        - Words (identifiers, keywords, numbers)
        - Operators and punctuation
        - String delimiters
        """
        # Pattern: word characters OR operators/symbols
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return len(tokens)

    def get_local_context(self, current_file: str, prefix: str, context_lines: int = 50) -> str:
        """
        Extract immediate context with intelligent boundaries.

        Strategy:
        1. Find the function/class containing the cursor
        2. Include the complete function/class definition
        3. Add context above (parent functions/classes)
        4. Add more context below for completions
        """
        try:
            with open(current_file, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()

            prefix_lines = prefix.split('\n')
            cursor_line = len(prefix_lines)
            lines = file_content.split('\n')

            # Try to find function/class boundaries using AST
            try:
                tree = ast.parse(file_content)
                # Find the node containing cursor_line
                containing_node = None
                containing_start = 0
                containing_end = len(lines)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                            if node.lineno <= cursor_line <= (node.end_lineno or node.lineno + 50):
                                if node.lineno > containing_start:  # Take innermost
                                    containing_node = node
                                    containing_start = node.lineno - 1
                                    containing_end = (node.end_lineno or len(lines))

                # If we found a containing node, use it as boundary
                if containing_node:
                    # Expand to include parent context
                    start = max(0, containing_start - 10)
                    end = min(len(lines), containing_end + 30)  # More lines after for completions
                else:
                    # Fallback: use fixed lines
                    start = max(0, cursor_line - context_lines)
                    end = min(len(lines), cursor_line + 50)  # Increased from 20 to 50
            except:
                # If AST parse fails, use fixed lines
                start = max(0, cursor_line - context_lines)
                end = min(len(lines), cursor_line + 50)

            local = '\n'.join(lines[start:end])
            return clean_code(local, preserve_types=True)
        except Exception:
            return ""

    def get_imported_symbols(self, current_file: str, prefix: str) -> Dict[str, str]:
        """
        Find definitions of symbols explicitly imported in current file.

        Handles:
        - from x import y
        - from x.y import z (qualified imports)
        - import x
        - import x as y (aliases)
        """
        imported = {}
        try:
            tree = ast.parse(prefix)
            imported_names: Set[tuple] = set()  # (import_name, module_name, symbol_name)

            # Extract all imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        import_name = alias.asname if alias.asname else alias.name.split('.')[0]
                        imported_names.add((import_name, module_name, None))

                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ''
                    for alias in node.names:
                        symbol_name = alias.name
                        import_name = alias.asname if alias.asname else symbol_name
                        imported_names.add((import_name, module_name, symbol_name))

            # Search knowledge graph for each import
            for import_name, module_name, symbol_name in imported_names:
                found = False

                # Strategy 1: Look for exact match in kg.nodes
                for node_id, node_info in self.kg.nodes.items():
                    node_file = node_info.get('file', '')
                    node_name = node_info.get('name', '')

                    # Match: module/file contains module name AND node name matches symbol
                    if symbol_name:
                        # from X import Y - look for Y
                        if node_name == symbol_name:
                            if module_name.replace('.', '/') in node_file or module_name.split('.')[-1] in node_file:
                                filepath = os.path.join(self.root, node_file)
                                try:
                                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                        content = f.read()
                                    imported[node_id] = clean_code(content[:2500], preserve_types=True)
                                    found = True
                                    break
                                except Exception:
                                    pass
                    else:
                        # import X - look for X module
                        if node_name == module_name.split('.')[-1]:
                            filepath = os.path.join(self.root, node_file)
                            try:
                                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                imported[node_id] = clean_code(content[:2500], preserve_types=True)
                                found = True
                                break
                            except Exception:
                                pass

                # Strategy 2: If not found by symbol name, search by module file path
                if not found and module_name:
                    module_path = module_name.replace('.', os.sep)
                    for py_file in self.kg.py_files:
                        rel_path = self.kg.get_relative_path(py_file)
                        if module_path in rel_path or rel_path.endswith(f"{module_path}.py"):
                            try:
                                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                imported[f"{rel_path}::{import_name}"] = clean_code(content[:2500], preserve_types=True)
                                found = True
                                break
                            except Exception:
                                pass
        except Exception:
            pass

        return imported

    def extract_called_symbols(self, prefix: str, current_file: str) -> Dict[str, str]:
        """
        Extract symbols (functions/methods) that are CALLED in the prefix code.

        This is HIGH PRIORITY context - the model needs to know what functions
        are being used to generate accurate completions.

        Strategy:
        1. Parse prefix AST
        2. Find all Call nodes
        3. Map to function definitions in kg
        """
        called = {}
        try:
            tree = ast.parse(prefix)
            called_names: Set[str] = set()

            # Find all function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Direct function call: func_name()
                    if isinstance(node.func, ast.Name):
                        called_names.add(node.func.id)
                    # Method call: obj.method()
                    elif isinstance(node.func, ast.Attribute):
                        called_names.add(node.func.attr)

            # Find these in knowledge graph
            for call_name in called_names:
                for node_id, node_info in self.kg.nodes.items():
                    if node_info.get('name') == call_name and node_info.get('type') in ('FunctionDef', 'AsyncFunctionDef'):
                        filepath = os.path.join(self.root, node_info['file'])
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            # Get just the function definition
                            called[node_id] = clean_code(content[:1500], preserve_types=True)
                        except Exception:
                            pass

        except Exception:
            pass

        return called

    def find_inheritance_chain(self, class_name: str, current_file: str) -> Dict[str, str]:
        """Find parent class definitions if current class is a subclass."""
        inheritance = {}
        try:
            with open(current_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    # Found our class, get bases
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            base_name = base.id
                            # Search for this class in kg
                            for node_id, node_info in self.kg.nodes.items():
                                if node_info['name'] == base_name and node_info['type'] == 'ClassDef':
                                    filepath = os.path.join(self.root, node_info['file'])
                                    try:
                                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                            base_content = f.read()
                                        inheritance[node_id] = clean_code(base_content, preserve_types=True)
                                    except Exception:
                                        pass
        except Exception:
            pass

        return inheritance

    def find_sibling_implementations(self, class_name: str, limit: int = 3) -> Dict[str, str]:
        """Find similar classes (similar naming patterns)."""
        siblings = {}
        try:
            # Extract base name (e.g., "AuthS3Provider" -> "Auth")
            base_patterns = [
                class_name[:3],  # First 3 chars
                class_name.split('Provider')[0],  # Before 'Provider'
            ]

            matches = []
            for node_id, node_info in self.kg.nodes.items():
                if node_info['type'] != 'ClassDef':
                    continue
                for pattern in base_patterns:
                    if pattern and pattern in node_info['name'] and node_info['name'] != class_name:
                        matches.append((node_id, node_info))

            # Take top matches
            for node_id, node_info in matches[:limit]:
                filepath = os.path.join(self.root, node_info['file'])
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    siblings[node_id] = clean_code(content[:1500], preserve_types=True)
                except Exception:
                    pass
        except Exception:
            pass

        return siblings

    def assemble_context(self, datapoint: Dict) -> str:
        """
        Assemble context using optimized sliding priority approach.

        Priority order:
        1. Local context (complete function/class containing cursor)
        2. Called symbols (functions/methods actually invoked in prefix) [NEW - HIGH VALUE]
        3. Imported symbols (definitions of imported modules/classes)
        4. Inheritance chain (parent class definitions)
        5. Sibling implementations (similar classes)
        6. Secondary context (fill remaining budget)
        """
        current_file = os.path.join(self.root, datapoint['path'])
        prefix = datapoint['prefix']

        context_parts = []
        token_count = 0

        # Priority 1: Local context (smart boundary detection)
        local = self.get_local_context(current_file, prefix)
        local_tokens = self.count_tokens(local)
        if token_count + local_tokens < self.mellum_budget:
            context_parts.append((local, "local", local_tokens))
            token_count += local_tokens

        # Priority 1.5: Called symbols (HIGH VALUE - functions actually used)
        called = self.extract_called_symbols(prefix, current_file)
        for sym_id, content in list(called.items())[:3]:  # Top 3 called functions
            sym_tokens = self.count_tokens(content)
            if token_count + sym_tokens < self.mellum_budget:
                context_parts.append((content, "called", sym_tokens))
                token_count += sym_tokens

        # Priority 2: Imported symbols
        imported = self.get_imported_symbols(current_file, prefix)
        for sym_id, content in list(imported.items())[:2]:
            sym_tokens = self.count_tokens(content)
            if token_count + sym_tokens < self.mellum_budget:
                context_parts.append((content, "imported", sym_tokens))
                token_count += sym_tokens

        # Priority 3: Inheritance chain
        try:
            prefix_tree = ast.parse(prefix)
            for node in ast.walk(prefix_tree):
                if isinstance(node, ast.ClassDef):
                    inheritance = self.find_inheritance_chain(node.name, current_file)
                    for sym_id, content in list(inheritance.items())[:1]:
                        sym_tokens = self.count_tokens(content)
                        if token_count + sym_tokens < self.mellum_budget:
                            context_parts.append((content, "inheritance", sym_tokens))
                            token_count += sym_tokens
                    break
        except Exception:
            pass

        # Priority 4: Sibling implementations
        try:
            prefix_tree = ast.parse(prefix)
            for node in ast.walk(prefix_tree):
                if isinstance(node, ast.ClassDef):
                    siblings = self.find_sibling_implementations(node.name, limit=1)
                    for sym_id, content in siblings.items():
                        sym_tokens = self.count_tokens(content)
                        if token_count + sym_tokens < self.mellum_budget:
                            context_parts.append((content, "sibling", sym_tokens))
                            token_count += sym_tokens
                    break
        except Exception:
            pass

        # Priority 5: Fill remaining tokens with secondary context
        remaining_budget = self.max_tokens - token_count
        # Filter: exclude test files, focus on related implementation files
        secondary_files = [
            f for f in self.kg.py_files
            if ('test' not in f.lower() and
                '__pycache__' not in f.lower() and
                '.pyc' not in f.lower())
        ][:5]

        for filepath in secondary_files:
            if remaining_budget <= 0:
                break
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                content = clean_code(content[:2000], preserve_types=True)  # Limit to 2000 chars
                sym_tokens = self.count_tokens(content)
                if sym_tokens < remaining_budget:
                    context_parts.append((content, "secondary", sym_tokens))
                    remaining_budget -= sym_tokens
            except Exception:
                pass

        # Compose final context
        result = ""
        for content, source, tokens in context_parts:
            rel_path = os.path.relpath(current_file, self.root) if source == "local" else source
            formatted = FILE_COMPOSE_FORMAT.format(
                file_sep=FILE_SEP_SYMBOL,
                file_name=rel_path,
                file_content=content
            )
            result += formatted + "\n"

        return result.strip()
