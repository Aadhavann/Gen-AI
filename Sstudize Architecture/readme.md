The architecture developed @ [Sstudize](https://play.google.com/store/apps/details?id=com.InnovAppWorks.Sstudize&amp%3Bhl=en_IN) using CrewAI

![Sstudize](https://github.com/user-attachments/assets/97157356-3efa-4e39-9c8a-bb1c9595fd8d)

Why CrewAI?
- Enabled us to achieve the "Mixture of Experts" (MoE) architecture with modular specialization.
- Functional role distribution, which is ideal in agentic design.
- Incorporating tools/plugins like API calls or databases for non-LLM tasks
- Using task routing strategies for dynamic MoE behaviors

Challenges
- Cost needed to be balanced with number of Agents used
- Higher chance of hallucination with increase in number of Agent instances
- Higher chance of output not conforming to expectations

Solutions developed
- Use RAG to reduce token usage and hallucinations
- Explored grammar files (using llama.cpp) to streamline output
- Improved non-LLM tools to further push MoE architecture

Sample Task Architecture

task_1 = Task(
    description="Break down the given software requirement into a sequence of actionable steps.",
    agent=planner,
    expected_output="A numbered list of subtasks or development steps."
)

task_2 = Task(
    description="Generate appropriate test cases for the plan produced by the planner.",
    agent=test_generator,
    expected_output="List of unit tests or validation prompts for each subtask.",
    context=[task_1]
)

task_3 = Task(
    description="Validate the results produced by the test generator.",
    agent=validator,
    expected_output="A report on test case quality or any identified errors.",
    context=[task_2]
)
