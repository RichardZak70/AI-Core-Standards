"""Autonomous Agent Template for AI Development Workspace.

This template provides a framework for building autonomous agents that can
plan, execute, and monitor AI development tasks with minimal human intervention.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(Enum):
    """Task priority levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a single task for the autonomous agent."""

    id: str
    name: str
    description: str
    priority: Priority
    dependencies: list[str]
    estimated_duration: int  # minutes
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class AutonomousAgent:
    """Autonomous agent for AI development tasks.

    This agent can plan, prioritize, and execute development tasks
    with minimal human supervision.
    """

    def __init__(self, name: str, workspace_path: Path):
        """Initialize the autonomous agent.

        Args:
            name: Agent name
            workspace_path: Path to workspace directory
        """
        self.name = name
        self.workspace_path = workspace_path
        self.tasks: dict[str, Task] = {}
        self.executors: dict[str, Callable] = {}
        self.max_concurrent_tasks = 3
        self.running_tasks: dict[str, asyncio.Task] = {}

        # Register default task executors
        self._register_default_executors()

        logger.info(f"Initialized AutonomousAgent '{name}' in {workspace_path}")

    def _register_default_executors(self):
        """Register default task executors."""
        self.executors.update(
            {
                "code_generation": self._execute_code_generation,
                "code_review": self._execute_code_review,
                "testing": self._execute_testing,
                "documentation": self._execute_documentation,
                "deployment": self._execute_deployment,
                "monitoring": self._execute_monitoring,
            }
        )

    def add_task(self, task: Task) -> str:
        """Add a task to the agent's queue.

        Args:
            task: Task to add

        Returns:
            Task ID
        """
        self.tasks[task.id] = task
        logger.info(f"Added task '{task.name}' with priority {task.priority.name}")
        return task.id

    def create_task(
        self,
        name: str,
        description: str,
        task_type: str,
        priority: Priority = Priority.MEDIUM,
        dependencies: list[str] = None,
        estimated_duration: int = 30,
    ) -> str:
        """Create and add a new task.

        Args:
            name: Task name
            description: Task description
            task_type: Type of task (must have registered executor)
            priority: Task priority
            dependencies: List of task IDs this task depends on
            estimated_duration: Estimated duration in minutes

        Returns:
            Task ID
        """
        if task_type not in self.executors:
            raise ValueError(f"No executor registered for task type '{task_type}'")

        task_id = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task = Task(
            id=task_id,
            name=name,
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            estimated_duration=estimated_duration,
        )
        task.task_type = task_type

        return self.add_task(task)

    async def execute_task(self, task_id: str) -> dict[str, Any]:
        """Execute a single task.

        Args:
            task_id: ID of task to execute

        Returns:
            Task execution result
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        if task.status != TaskStatus.PENDING:
            logger.warning(f"Task {task_id} is not pending (status: {task.status})")
            return {"error": "Task is not in pending status"}

        # Check dependencies
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                logger.info(f"Task {task_id} waiting for dependency {dep_id}")
                return {"waiting": f"Dependency {dep_id} not completed"}

        # Execute task
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        try:
            executor = self.executors[task.task_type]
            result = await executor(task)

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result

            logger.info(f"Task {task_id} completed successfully")
            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()

            logger.error(f"Task {task_id} failed: {e}")
            return {"error": str(e)}

    async def run_autonomous_mode(self, duration_minutes: int = 60):
        """Run the agent in autonomous mode for specified duration.

        Args:
            duration_minutes: How long to run autonomously
        """
        start_time = datetime.now()
        end_time = start_time.timestamp() + (duration_minutes * 60)

        logger.info(f"Starting autonomous mode for {duration_minutes} minutes")

        while datetime.now().timestamp() < end_time:
            # Get next tasks to execute
            ready_tasks = self._get_ready_tasks()

            # Execute tasks up to concurrent limit
            while len(self.running_tasks) < self.max_concurrent_tasks and ready_tasks:
                task_id = ready_tasks.pop(0)
                task_coroutine = self.execute_task(task_id)
                self.running_tasks[task_id] = asyncio.create_task(task_coroutine)

            # Check completed tasks
            completed = []
            for task_id, task_obj in self.running_tasks.items():
                if task_obj.done():
                    try:
                        result = await task_obj
                        logger.info(f"Autonomous execution completed: {task_id}")
                    except Exception as e:
                        logger.error(f"Autonomous execution failed: {task_id} - {e}")
                    completed.append(task_id)

            # Remove completed tasks
            for task_id in completed:
                del self.running_tasks[task_id]

            # Wait a bit before next iteration
            await asyncio.sleep(5)

        logger.info("Autonomous mode completed")

        # Wait for remaining tasks
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)

    def _get_ready_tasks(self) -> list[str]:
        """Get list of tasks ready for execution.

        Returns:
            List of task IDs sorted by priority
        """
        ready_tasks = []

        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING and task_id not in self.running_tasks:
                # Check if dependencies are met
                deps_met = all(
                    self.tasks.get(dep_id, {}).status == TaskStatus.COMPLETED for dep_id in task.dependencies
                )

                if deps_met:
                    ready_tasks.append((task_id, task.priority.value))

        # Sort by priority (higher number = higher priority)
        ready_tasks.sort(key=lambda x: x[1], reverse=True)
        return [task_id for task_id, _ in ready_tasks]

    # Task Executors
    async def _execute_code_generation(self, task: Task) -> dict[str, Any]:
        """Execute code generation task."""
        logger.info(f"Generating code for: {task.description}")

        # Simulate code generation (replace with actual AI integration)
        await asyncio.sleep(2)  # Simulate processing time

        generated_code = f"""# Generated code for: {task.description}
def generated_function():
    '''AI-generated function based on requirements.'''
    # TODO: Implement actual functionality
    pass
"""

        return {
            "type": "code_generation",
            "code": generated_code,
            "language": "python",
            "lines_of_code": len(generated_code.split("\n")),
        }

    async def _execute_code_review(self, task: Task) -> dict[str, Any]:
        """Execute code review task."""
        logger.info(f"Reviewing code for: {task.description}")

        await asyncio.sleep(3)  # Simulate review time

        return {
            "type": "code_review",
            "score": 85,
            "issues_found": 2,
            "suggestions": ["Add type hints to function parameters", "Consider adding error handling for edge cases"],
            "approved": True,
        }

    async def _execute_testing(self, task: Task) -> dict[str, Any]:
        """Execute testing task."""
        logger.info(f"Running tests for: {task.description}")

        await asyncio.sleep(4)  # Simulate test execution

        return {
            "type": "testing",
            "tests_run": 15,
            "tests_passed": 14,
            "tests_failed": 1,
            "coverage": 87.5,
            "duration": 3.2,
        }

    async def _execute_documentation(self, task: Task) -> dict[str, Any]:
        """Execute documentation task."""
        logger.info(f"Creating documentation for: {task.description}")

        await asyncio.sleep(2)  # Simulate documentation creation

        return {"type": "documentation", "pages_created": 3, "words_written": 1250, "format": "markdown"}

    async def _execute_deployment(self, task: Task) -> dict[str, Any]:
        """Execute deployment task."""
        logger.info(f"Deploying: {task.description}")

        await asyncio.sleep(5)  # Simulate deployment

        return {
            "type": "deployment",
            "environment": "staging",
            "status": "success",
            "url": "https://staging.example.com",
            "health_check": "passed",
        }

    async def _execute_monitoring(self, task: Task) -> dict[str, Any]:
        """Execute monitoring setup task."""
        logger.info(f"Setting up monitoring for: {task.description}")

        await asyncio.sleep(1)  # Simulate monitoring setup

        return {
            "type": "monitoring",
            "metrics_configured": 8,
            "alerts_created": 5,
            "dashboard_url": "https://monitoring.example.com",
        }

    def get_status_report(self) -> dict[str, Any]:
        """Get comprehensive status report.

        Returns:
            Status report with task statistics
        """
        status_counts = {status.value: 0 for status in TaskStatus}
        total_tasks = len(self.tasks)

        for task in self.tasks.values():
            status_counts[task.status.value] += 1

        return {
            "agent_name": self.name,
            "workspace": str(self.workspace_path),
            "total_tasks": total_tasks,
            "status_breakdown": status_counts,
            "running_tasks": len(self.running_tasks),
            "completion_rate": (status_counts["completed"] / total_tasks * 100) if total_tasks > 0 else 0,
            "current_time": datetime.now().isoformat(),
        }

    def export_tasks(self, file_path: Path):
        """Export tasks to JSON file.

        Args:
            file_path: Path to export file
        """
        tasks_data = {
            task_id: {
                **asdict(task),
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            }
            for task_id, task in self.tasks.items()
        }

        with open(file_path, "w") as f:
            json.dump(tasks_data, f, indent=2, default=str)

        logger.info(f"Tasks exported to {file_path}")


# Example usage
async def main():
    """Example usage of the AutonomousAgent."""
    # Initialize agent
    agent = AutonomousAgent(name="DevAgent", workspace_path=Path("./workspace"))

    # Create a series of related tasks
    task1 = agent.create_task(
        name="Generate User Authentication Module",
        description="Create a secure user authentication system with JWT tokens",
        task_type="code_generation",
        priority=Priority.HIGH,
        estimated_duration=45,
    )

    task2 = agent.create_task(
        name="Review Authentication Code",
        description="Review the generated authentication code for security and quality",
        task_type="code_review",
        priority=Priority.HIGH,
        dependencies=[task1],
        estimated_duration=30,
    )

    task3 = agent.create_task(
        name="Test Authentication Module",
        description="Create and run comprehensive tests for authentication",
        task_type="testing",
        priority=Priority.MEDIUM,
        dependencies=[task1, task2],
        estimated_duration=60,
    )

    task4 = agent.create_task(
        name="Document Authentication API",
        description="Create comprehensive API documentation",
        task_type="documentation",
        priority=Priority.MEDIUM,
        dependencies=[task1],
        estimated_duration=30,
    )

    # Run autonomous mode for 10 minutes
    await agent.run_autonomous_mode(duration_minutes=10)

    # Print final status
    status = agent.get_status_report()
    print(json.dumps(status, indent=2))

    # Export tasks
    agent.export_tasks(Path("./task_history.json"))


if __name__ == "__main__":
    asyncio.run(main())
