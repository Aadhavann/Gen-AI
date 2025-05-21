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
