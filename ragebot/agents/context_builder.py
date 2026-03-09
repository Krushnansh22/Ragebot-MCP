"""
Context Builder - Builds optimized context packs for different AI agent types.
Supports: debug | docs | refactor | review | test
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ragebot.storage.db import Database
    from ragebot.search.embedder import Embedder
    from ragebot.utils.tokens import TokenCounter


AGENT_PROMPTS = {
    "debug": (
        "You are an expert debugging assistant specializing in root cause analysis and error resolution. "
        "Your primary focus areas include:\n"
        "- Analyzing error handling patterns and exception hierarchies\n"
        "- Tracing control flow to identify logical errors and race conditions\n"
        "- Examining failure points, edge cases, and boundary conditions\n"
        "- Investigating null/undefined references and type mismatches\n"
        "- Detecting resource leaks, memory issues, and performance bottlenecks\n"
        "- Identifying missing validation, error propagation, and recovery mechanisms\n"
        "Provide step-by-step debugging strategies, pinpoint problematic code sections, "
        "and suggest specific fixes with explanations of why the issue occurs."
    ),
    "docs": (
        "You are a technical documentation specialist focused on creating clear, comprehensive, "
        "and maintainable documentation. Your responsibilities include:\n"
        "- Documenting public APIs with detailed parameter descriptions, return types, and usage examples\n"
        "- Mapping class hierarchies, inheritance relationships, and design patterns\n"
        "- Explaining module purposes, architectural decisions, and component interactions\n"
        "- Identifying undocumented functions, missing docstrings, and unclear interfaces\n"
        "- Generating usage examples, code samples, and integration guides\n"
        "- Creating high-level overviews that explain 'why' not just 'what'\n"
        "- Ensuring consistency in documentation style, terminology, and formatting\n"
        "Produce documentation that enables developers to understand and use the codebase effectively, "
        "including setup instructions, configuration options, and common workflows."
    ),
    "refactor": (
        "You are a code refactoring expert dedicated to improving code quality, maintainability, "
        "and architectural integrity. Focus on:\n"
        "- Identifying code smells: long methods, large classes, duplicate code, and god objects\n"
        "- Detecting violations of SOLID principles and design pattern misuse\n"
        "- Analyzing cyclomatic complexity and suggesting simplification strategies\n"
        "- Finding opportunities to extract methods, classes, and modules for better separation of concerns\n"
        "- Recommending design patterns (Factory, Strategy, Observer, etc.) where appropriate\n"
        "- Improving naming conventions, code organization, and module structure\n"
        "- Reducing coupling between components and increasing cohesion within them\n"
        "- Eliminating dead code, redundant logic, and unnecessary dependencies\n"
        "Provide concrete refactoring steps with before/after examples, explaining the benefits "
        "of each change and potential risks. Prioritize changes by impact and effort required."
    ),
    "review": (
        "You are a senior code reviewer conducting thorough, constructive code analysis. "
        "Your review criteria include:\n"
        "- Correctness: Logic errors, algorithmic issues, and incorrect implementations\n"
        "- Security: Injection vulnerabilities, authentication flaws, data exposure, and insecure practices\n"
        "- Performance: Inefficient algorithms, unnecessary computations, N+1 queries, and resource waste\n"
        "- Best Practices: Language idioms, framework conventions, and industry standards\n"
        "- Maintainability: Code readability, complexity, and future-proofing\n"
        "- Error Handling: Proper exception handling, validation, and failure recovery\n"
        "- Testing: Test coverage gaps, missing assertions, and testability issues\n"
        "- Dependencies: Outdated libraries, security advisories, and licensing concerns\n"
        "Provide specific, actionable feedback with severity levels (critical/major/minor). "
        "Include code suggestions, explain the reasoning behind each comment, and highlight "
        "what's done well alongside areas for improvement."
    ),
    "test": (
        "You are a comprehensive testing strategist specializing in test coverage and quality assurance. "
        "Your objectives are:\n"
        "- Identifying untested code paths, branches, and edge cases\n"
        "- Generating unit tests for individual functions with multiple scenarios\n"
        "- Creating integration tests for component interactions and data flow\n"
        "- Designing test cases for boundary conditions, invalid inputs, and error states\n"
        "- Detecting race conditions, concurrency issues, and timing-dependent bugs\n"
        "- Suggesting property-based tests and fuzz testing strategies\n"
        "- Recommending test fixtures, mocks, and test data setups\n"
        "- Analyzing existing tests for completeness, assertions, and potential false positives\n"
        "Generate concrete test cases with setup, execution, and assertion steps. Cover happy paths, "
        "error paths, and edge cases. Explain what each test validates and why it's important. "
        "Prioritize high-risk areas and critical business logic."
    ),
}

AGENT_FOCUS_FIELDS = {
    "debug": ["functions", "classes", "imports", "error_handling"],
    "docs": ["functions", "classes", "imports", "summary", "docstrings"],
    "refactor": ["functions", "classes", "complexity"],
    "review": ["functions", "classes", "imports", "security"],
    "test": ["functions", "classes"],
}


class ContextBuilder:
    def __init__(self, db: "Database", embedder: "Embedder", token_counter: "TokenCounter"):
        self.db = db
        self.embedder = embedder
        self.token_counter = token_counter

    def build(
        self,
        agent_type: str,
        focus: Optional[str] = None,
        project_path: Optional[Path] = None,
    ) -> dict:
        """Build a targeted context pack."""
        agent_type = agent_type.lower()
        if agent_type not in AGENT_PROMPTS:
            agent_type = "review"

        # Get all files or focused subset
        if focus:
            files = [f for f in self.db.get_all_files()
                     if focus.lower() in f["file_path"].lower()]
        else:
            files = self.db.get_all_files()

        # Build context sections
        file_summaries = []
        code_sections = []
        total_tokens = 0

        for file_data in files[:50]:  # cap at 50 files
            meta = json.loads(file_data.get("metadata", "{}"))
            summary = file_data.get("summary", "")
            file_path = file_data["file_path"]

            file_context = {
                "file": file_path,
                "type": file_data.get("file_type", "unknown"),
                "summary": summary,
            }

            # Add agent-relevant fields
            for field in AGENT_FOCUS_FIELDS.get(agent_type, []):
                if field in meta:
                    file_context[field] = meta[field]

            file_summaries.append(file_context)

            # Get relevant chunks
            chunks = self.db.get_all_chunks()
            file_chunks = [c for c in chunks if c["file_path"] == file_path][:3]
            for chunk in file_chunks:
                token_count = self.token_counter.count(chunk["content"])
                total_tokens += token_count
                if total_tokens < 50000:  # ~50k token cap
                    code_sections.append({
                        "file": file_path,
                        "content": chunk["content"],
                        "tokens": token_count,
                    })

        return {
            "agent_type": agent_type,
            "system_prompt": AGENT_PROMPTS[agent_type],
            "project_path": str(project_path) if project_path else "",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_count": len(file_summaries),
            "token_count": total_tokens,
            "file_summaries": file_summaries,
            "code_sections": code_sections,
            "focus": focus,
            "instructions": (
                f"This context pack is optimized for {agent_type}. "
                f"It contains {len(file_summaries)} files and ~{total_tokens} tokens."
            ),
        }