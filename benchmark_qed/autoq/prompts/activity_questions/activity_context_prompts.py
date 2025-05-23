ACTIVITY_IDENTIFICATION_PROMPT = """\
You are a helpful assistant tasked with identifying activities related to a dataset of interest based on the dataset description.

---Guidance---
Based on the dataset description, create a list of {user_count} personas who might be interested in the dataset.

For each identified persona, generate a list of {task_count} tasks that they might want to perform using the dataset.

If the dataset description indicates a prominent topic area, ensure that the personas and tasks are linked to that topic area in some way.

---Output---
Output should be in JSON format, with the following structure:
{{
    "personas": [
        {{
            "persona": "A description of the persona, including their potential interests",
            "tasks": ["A list of non-duplicating tasks that the persona might want to perform using the dataset"]
        }},
        ...
    ]
}}
Output JSON only, with no additional text or commentary.


---Dataset description---
{dataset_description}

---Number of personas to identify---
{user_count}

---Number of tasks to generate per persona---
{task_count}
"""
