#!/usr/bin/env python3
"""Terminal AI Companion for AI Development Workspace.

A command-line AI assistant that provides code explanation, refactoring suggestions,
and development guidance directly in the terminal.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not installed. Install with: pip install openai")

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AICompanion:
    """Terminal AI companion for development assistance."""

    def __init__(self, model_provider: str = "openai"):
        """Initialize the AI companion.

        Args:
            model_provider: AI model provider (openai, anthropic, local)
        """
        self.model_provider = model_provider
        self.conversation_history = []
        self.workspace_context = self._load_workspace_context()

    def _load_workspace_context(self) -> dict[str, Any]:
        """Load workspace context for better AI responses."""
        context = {"languages": [], "frameworks": [], "recent_files": []}

        try:
            # Detect languages in workspace
            cwd = Path.cwd()

            if list(cwd.glob("*.py")):
                context["languages"].append("Python")
            if list(cwd.glob("*.js")) or list(cwd.glob("*.ts")):
                context["languages"].append("JavaScript/TypeScript")
            if list(cwd.glob("*.cpp")) or list(cwd.glob("*.c")) or list(cwd.glob("*.h")):
                context["languages"].append("C/C++")
            if list(cwd.glob("*.java")):
                context["languages"].append("Java")
            if list(cwd.glob("*.cs")):
                context["languages"].append("C#")

            # Detect frameworks
            if (cwd / "requirements.txt").exists():
                context["frameworks"].append("Python/pip")
            if (cwd / "package.json").exists():
                context["frameworks"].append("Node.js")
            if (cwd / "pom.xml").exists() or (cwd / "build.gradle").exists():
                context["frameworks"].append("Java/Maven/Gradle")
            if list(cwd.glob("*.csproj")):
                context["frameworks"].append(".NET")

        except Exception as e:
            print(f"Warning: Could not load workspace context: {e}")

        return context

    def explain_code(self, file_path: str | None = None, code: str | None = None) -> str:
        """Explain code from file or direct input.

        Args:
            file_path: Path to code file to explain
            code: Direct code input to explain

        Returns:
            Code explanation
        """
        if file_path:
            try:
                with open(file_path, encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                return f"Error reading file: {e}"

        if not code:
            return "No code provided to explain."

        prompt = f"""
        Please explain this code in detail:
        
        Context: Working in a {", ".join(self.workspace_context["languages"])} project
        
        Code:
        ```
        {code}
        ```
        
        Please provide:
        1. Overall purpose and functionality
        2. Key components and how they work
        3. Notable patterns or techniques used
        4. Potential improvements or issues
        5. Usage examples if applicable
        """

        return self._get_ai_response(prompt)

    def suggest_refactoring(self, file_path: str | None = None, code: str | None = None) -> str:
        """Suggest code refactoring improvements.

        Args:
            file_path: Path to code file to refactor
            code: Direct code input to refactor

        Returns:
            Refactoring suggestions
        """
        if file_path:
            try:
                with open(file_path, encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                return f"Error reading file: {e}"

        if not code:
            return "No code provided for refactoring."

        prompt = f"""
        Please analyze this code and suggest refactoring improvements:
        
        Context: {", ".join(self.workspace_context["languages"])} project
        
        Code:
        ```
        {code}
        ```
        
        Please provide:
        1. Code quality assessment
        2. Specific refactoring suggestions
        3. Performance improvements
        4. Better design patterns to apply
        5. Refactored code examples for key improvements
        
        Focus on:
        - Readability and maintainability
        - Performance optimization
        - Best practices for the language
        - Error handling improvements
        - Security considerations
        """

        return self._get_ai_response(prompt)

    def debug_error(self, error_message: str, code: str | None = None) -> str:
        """Help debug an error.

        Args:
            error_message: Error message to debug
            code: Related code context

        Returns:
            Debug assistance
        """
        prompt = f"""
        Help debug this error:
        
        Error: {error_message}
        
        Context: {", ".join(self.workspace_context["languages"])} project
        
        """

        if code:
            prompt += f"""
            Related Code:
            ```
            {code}
            ```
            """

        prompt += """
        Please provide:
        1. Likely cause of the error
        2. Step-by-step debugging approach
        3. Specific fixes to try
        4. Prevention strategies for the future
        5. Related best practices
        """

        return self._get_ai_response(prompt)

    def generate_tests(self, file_path: str | None = None, code: str | None = None) -> str:
        """Generate unit tests for code.

        Args:
            file_path: Path to code file
            code: Direct code input

        Returns:
            Generated test code
        """
        if file_path:
            try:
                with open(file_path, encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                return f"Error reading file: {e}"

        if not code:
            return "No code provided for test generation."

        # Determine test framework based on language
        test_framework = "pytest"  # Default
        if "Java" in self.workspace_context["languages"]:
            test_framework = "JUnit 5"
        elif "C#" in self.workspace_context["languages"]:
            test_framework = "xUnit"
        elif "C/C++" in self.workspace_context["languages"]:
            test_framework = "Google Test"

        prompt = f"""
        Generate comprehensive unit tests for this code using {test_framework}:
        
        Code to test:
        ```
        {code}
        ```
        
        Please provide:
        1. Complete test file with proper imports
        2. Test cases for normal operation
        3. Edge case testing
        4. Error condition testing
        5. Mock usage where appropriate
        6. Test data setup and teardown
        
        Follow best practices for {test_framework} and ensure high coverage.
        """

        return self._get_ai_response(prompt)

    def code_review(self, file_path: str | None = None, code: str | None = None) -> str:
        """Perform AI code review.

        Args:
            file_path: Path to code file
            code: Direct code input

        Returns:
            Code review feedback
        """
        if file_path:
            try:
                with open(file_path, encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                return f"Error reading file: {e}"

        if not code:
            return "No code provided for review."

        prompt = f"""
        Perform a thorough code review of this code:
        
        Context: {", ".join(self.workspace_context["languages"])} project
        
        Code:
        ```
        {code}
        ```
        
        Please review for:
        1. Code quality and readability
        2. Performance issues
        3. Security vulnerabilities
        4. Best practices compliance
        5. Error handling
        6. Documentation completeness
        7. Testing considerations
        
        Provide specific, actionable feedback with examples.
        """

        return self._get_ai_response(prompt)

    def chat(self, message: str) -> str:
        """General chat with AI about development topics.

        Args:
            message: User message

        Returns:
            AI response
        """
        context_info = f"Context: Working on {', '.join(self.workspace_context['languages'])} project"
        full_message = f"{context_info}\n\nQuestion: {message}"

        return self._get_ai_response(full_message)

    def _get_ai_response(self, prompt: str) -> str:
        """Get response from AI model.

        Args:
            prompt: Prompt to send to AI

        Returns:
            AI response
        """
        try:
            if self.model_provider == "openai" and OPENAI_AVAILABLE:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful AI programming assistant. Provide clear, practical advice and code examples.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2000,
                    temperature=0.3,
                )
                return response.choices[0].message.content

            elif self.model_provider == "anthropic" and ANTHROPIC_AVAILABLE:
                client = anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-3-sonnet-20240229", max_tokens=2000, messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

            else:
                return f"AI model '{self.model_provider}' not available. Please install required dependencies."

        except Exception as e:
            return f"Error getting AI response: {e}"


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="AI Companion - Terminal AI assistant for development")

    parser.add_argument(
        "command", choices=["explain", "refactor", "debug", "test", "review", "chat"], help="Command to execute"
    )

    parser.add_argument("--file", "-f", help="File path for code analysis")

    parser.add_argument("--code", "-c", help="Direct code input")

    parser.add_argument("--message", "-m", help="Message for chat or error for debug")

    parser.add_argument(
        "--provider", "-p", choices=["openai", "anthropic", "local"], default="openai", help="AI model provider"
    )

    args = parser.parse_args()

    # Initialize AI companion
    companion = AICompanion(model_provider=args.provider)

    # Execute command
    try:
        if args.command == "explain":
            result = companion.explain_code(file_path=args.file, code=args.code)

        elif args.command == "refactor":
            result = companion.suggest_refactoring(file_path=args.file, code=args.code)

        elif args.command == "debug":
            if not args.message:
                print("Error: --message required for debug command")
                sys.exit(1)
            result = companion.debug_error(error_message=args.message, code=args.code)

        elif args.command == "test":
            result = companion.generate_tests(file_path=args.file, code=args.code)

        elif args.command == "review":
            result = companion.code_review(file_path=args.file, code=args.code)

        elif args.command == "chat":
            if not args.message:
                print("Error: --message required for chat command")
                sys.exit(1)
            result = companion.chat(message=args.message)

        print("\n" + "=" * 80)
        print(f"AI COMPANION - {args.command.upper()}")
        print("=" * 80)
        print(result)
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
