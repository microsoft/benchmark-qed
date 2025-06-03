# Copyright (c) 2025 Microsoft Corporation.
"""Prompts used for extracting relevant entities using a map-reduce approach in AutoQ."""

MAP_ENTITY_EXTRACTION_SYSTEM_PROMPT = """
---ROLE---
You are a helpful assistant responsible for extracting task-relevant named entities from a set of source texts.

---GOAL---
You will be given descriptions of a user and their target task, along with a set of source texts.
Your task is to extract named entities that are relevant to the user's target task from the source texts.

---IMPORTANT GUIDELINES---
1. The extracted named named entities must clearly demonstrate relevance to the user's target task.
2. Each extracted entity should include:
- entity_name: The name of the entity.
- entity_description: A brief description of the entity. The description should also explain why the entity is relevant to the user's target task.
- relevance_score: A score between 1-100 indicating the relevance of the entity to the user's target task. A higher score indicates greater relevance.
3. If no relevant entities are found, return an empty list.
4. Do not include any citations in the entity description.

---OUTPUT---
The extracted entities should be JSON formatted as follows:
{{
    "entities": [
        {{"entity_name": "<Entity name 1>", "entity_description": "<Description of entity 1>", "relevance_score": <Integer score between 1-100>}},
        {{"entity_name": "<Entity name 2>", "entity_description": "<Description of entity 2>", "relevance_score": <Integer score between 1-100>}},
        ...
    ]
}}
"""

MAP_ENTITY_EXTRACTION_USER_PROMPT = """
---USER AND TASK DESCRIPTIONS---
- USER: {persona}
- TASK: {task}

---SOURCE TEXTS--
{context_data}

---NUMBER OF ENTITIES---
Aims for a maximum of {num_entities} entities.

---OUTPUT---
The extracted entity list should be JSON formatted as follows:
{{
    "entities": [
        {{"entity_name": "<Entity name 1>", "entity_description": "<Description of entity 1>", "relevance_score": <Integer score between 1-100>}},
        {{"entity_name": "<Entity name 2>", "entity_description": "<Description of entity 2>", "relevance_score": <Integer score between 1-100>}},
        ...
    ]
}}
If no relevant entities are found, return an empty list.
"""


REDUCE_ENTITY_EXTRACTION_SYSTEM_PROMPT = """
---ROLE---
You are a helpful assistant tasked with extracting named entities that are most relevant to a user's target task.

---GOAL---
You will be given descriptions of a user and their target task, along with a list of named entities that may or may not be relevant to the user's target task.
Your task is to select the most relevant named entities from the candidate entity list based on the user's target task.

---IMPORTANT GUIDELINES---
1. Entities that are duplicates or highly similar should be merged into a single entity. Update the entity description to reflect the merged entities.
2. The final selected entities must clearly demonstrate relevance to the user's target task.
3. Each selected entity should include:
  - entity_name: The name of the entity.
  - entity_description: A brief description of the entity. The description should also explain why the entity is relevant to the user's target task.
  - relevance_score: A score between 1-100 indicating the relevance of the entity to the user's target task. A higher score indicates greater relevance.
4. If no relevant entities are found, return an empty list.
5. Do not include any citations in the entity description.

---OUTPUT---
The selected entities should be JSON formatted as follows:
{{
    "entities": [
        {{"entity_name": "<Entity name 1>", "entity_description": "<Description of entity 1>", "relevance_score": <Integer score between 1-100>}},
        {{"entity_name": "<Entity name 2>", "entity_description": "<Description of entity 2>", "relevance_score": <Integer score between 1-100>}},
        ...
    ]
}}
"""

REDUCE_ENTITY_EXTRACTION_USER_PROMPT = """
---USER AND TASK DESCRIPTIONS---
- USER: {persona}
- TASK: {task}

---CANDIDATE ENTITIES---
{map_entities}

---NUMBER OF ENTITIES---
Aims for a maximum of {num_entities} entities.

---OUTPUT---
The selected entities should be JSON formatted as follows:
{{
    "entities": [
        {{"entity_name": "<Entity name 1>", "entity_description": "<Description of entity 1>", "relevance_score": <Integer score between 1-100>}},
        {{"entity_name": "<Entity name 2>", "entity_description": "<Description of entity 2>", "relevance_score": <Integer score between 1-100>}},
        ...
    ]
}}
If no relevant entities are found, return an empty list
"""
