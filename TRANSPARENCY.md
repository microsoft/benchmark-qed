# BenchmarkQED, AP News articles & Behind The Tech Podcast Transcripts 

## Overview 

BenchmarkQED is a set of tools that evaluate the quality of generated text from RAG technologies.   

 

The AP News dataset is a collection of health-related news articles by The Associated Press in 2023 and 2024, licensed by Microsoft for research purposes. 

The Podcast Transcripts dataset is a collection of transcripts from the Microsoft Podcast series Behind the Tech in 2024 and 2025. 

### What Can BenchmarkQED Do

BenchmarkQED was developed to help evaluate the efficacy of RAG technologies, such as GraphRAG. RAG technologies enable a user to provide a novel dataset to a language model and then have a Q&A experience over the dataset. The tools in this suite: 

1. evaluate the user's RAG dataset and automatically generate appropriate queries to effectively judge the quality of generated responses, 
2. allow a user to compare performance of competing RAG methods using LLM-as-a-judge methods, 
3. evaluate the user's RAG dataset and provide utility functions like data summarization and data sampling. 

More discussion of the BenchmarkQED-style evaluation approach can be found in our paper at: https://arxiv.org/abs/2404.16130 

The AP News and Behind the Tech Podcast Transcripts are a companion dataset for BenchmarkQED and the example notebooks. 

### Intended Uses

BenchmarkQED and the AP News and Podcast Transcripts datasets are intended to be used by researchers of RAG technologies. 

Users should be domain experts who are independently capable of evaluating the quality of outputs before acting on them. 

The AP News dataset and Podcast Transcripts dataset are intended to be used together with BenchmarkQED. 

BenchmarkQED, the AP News datasets, and the Podcast Transcripts datasets are being shared with the research community to facilitate reproduction of our results and foster further research in this area. 

### Out-of-Scope-Uses

BenchmarkQED is not well suited for information discovery in a new or unfamiliar domain. BenchmarkQED is best suited for users with familiarity or expertise in the domain of the generated content being evaluated, and for research on RAG technologies. 

We do not recommend using BenchmarkQED, the AP News dataset, or the Podcast Transcripts dataset in commercial or real-world applications without further testing and development. They are being released for research purposes. 

BenchmarkQED, the AP News dataset, and the Podcast Transcripts dataset were not designed or evaluated for all possible downstream purposes. Developers should consider their inherent limitations as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness concerns specific to each intended downstream use. 

BenchmarkQED, the AP News dataset, and the Podcast Transcripts dataset should not be used in highly regulated domains where inaccurate outputs could suggest actions that lead to injury or negatively impact an individual's legal, financial, or life opportunities. 

We do not recommend using BenchmarkQED, the AP News dataset, or the Podcast Transcripts dataset in the context of high-risk decision making (e.g. in law enforcement, legal, finance, or healthcare).  

## Dataset Details 

### Dataset Contents

AP News dataset consists of 1397 health related news articles written between November 2023 to April 2024 by The Associated Press. Articles were organized by publication time and were labeled with a unique id. 

The Behind the Tech Podcast Transcripts consist of 70 episodes of Kevin Scott’s Behind the Tech podcast series. Each episode is associated with a label describing the episode number and title. 

The Podcast Transcript data was collected between 2024-2025.  

The AP News dataset contains links to external data sources, while the Podcast Transcripts dataset does not. 

Data points in AP News dataset and the Behind the Tech Podcast Transcripts correspond to individual’s opinions and identifiable information that was shared during the interview or that were publicly known if they were a public figure. It may include data pertaining to children.  

Measures have not been taken to remove potentially identifying information, since these are publicly available podcast interviews, and/or news articles that were licensed for research purposes.  

Measures have not been taken to remove sensitive or private data, since the information contained in these datasets are already publicly available. 

  

### Data Creation & Processing

The AP News and Podcast Transcripts datasets are an original dataset created by our research team for RAG technology research, but the original content of news articles and podcast interview transcripts was created by the Associated Press and the Behind the Tech Podcast Series.  

The original content contained information that could be used to directly identify a person, such as their name, title, or details relating to their employment or academic pursuits. 

The original content may contain information that might be considered sensitive or private, such as racial or ethnic origins, sexual orientations, religious beliefs, disability status, political opinions, criminal history, or other, since these are conversational interviews with guests for either news articles or podcast episodes. 

The AP News dataset contains a number of metadata fields created by the Associated Press. Each Podcast Transcript episode had existing labels for episode number and title, that was then manually scrubbed by the research team for inconsistencies.  

## How To Get Started

To begin using BenchmarkQED review the README and related blog post to learn about each of the tools and their uses. Also review our example notebooks that contain example queries and datasets to walk through using BenchmarkQED for the first time. 

 

## Evaluation 

BenchmarkQED was evaluated on its ability to evaluate generated text responses of question-and-answer RAG systems. 

A detailed discussion of our evaluation methods and results can be found in our paper at: https://arxiv.org/abs/2404.16130 

 

### Evaluation Methods

We used human inspection to evaluate BenchmarkQED’s performance. 

We compared the performance of BenchmarkQED’s adjudications to human-based evaluation and inspection of the results using small scale test datasets of news articles and podcast transcripts that could be easily verified against the generated content. 

The models used for evaluation and development were GPT-4.1, GPT-4o, GPT-4-turbo, and o3-mini. For more on these specific models, please see https://openai.com/index/gpt-4-1/, https://openai.com/index/gpt-4o-system-card/, https://platform.openai.com/docs/models/gpt-4-turbo, and https://openai.com/index/openai-o3-mini/.  

Results may vary if BenchmarkQED is used with a different model, based on its unique design, configuration and training.  

 

## Limitations 

BenchmarkQED was designed and tested using the English language, and the AP News and Podcast Transcript datasets consist of English language instances only. Performance in other languages may vary and should be assessed by someone who is both an expert in the expected outputs and a native speaker of that language. 

Outputs generated by AI may include factual errors, fabrication, or speculation. Users are responsible for assessing the accuracy of generated content. All decisions leveraging outputs of the system should be made with human oversight and not be based solely on system outputs. 

BenchmarkQED inherits any biases, errors, or omissions produced by the user’s chosen LLM (GPT 4.1  in our experiments and default configuration). Developers are advised to choose an appropriate base LLM carefully, depending on the intended use case. 

BenchmarkQED uses GPT-4.1 as the default model. See https://openai.com/index/gpt-4-1/ to understand the capabilities and limitations of these models.  

There has not been a systematic effort to ensure that systems using BenchmarkQED are protected from security vulnerabilities such as indirect prompt injection attacks. Any systems using it should take proactive measures to harden their systems as appropriate. 

The ability to access external links in the AP News dataset is beyond the control of the research team.  

The AP News and Podcast Transcript datasets have not been systematically evaluated for sociocultural/economic/demographic/linguistic bias. Developers should consider the potential for bias as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness concerns specific to each intended downstream use.  

 

## Best Practices

Better performance of BenchmarkQED can be achieved by using datasets that are cohesive in their overall topic, and that are entity rich. 

We recommend using the example datasets independently of one another during use and evaluations, since they are central in their own theme and topical area.  

We strongly encourage users to use LLMs that support robust Responsible AI mitigations, such as Azure Open AI (AOAI) services. Such services continually update their safety and RAI mitigations with the latest industry standards for responsible use. For more on AOAI’s best practices when employing foundations models for scripts and applications: 

* [Blog post on responsible AI features in AOAI that were presented at Ignite 2023](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/announcing-new-ai-safety-amp-responsible-ai-features-in-azure/ba-p/3983686) 
* [Overview of Responsible AI practices for Azure OpenAI models](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/overview) 
* [Azure OpenAI Transparency Note](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/transparency-note) 
* [OpenAI’s Usage policies](https://openai.com/policies/usage-policies) 
* [Azure OpenAI’s Code of Conduct](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct) 

Users are reminded to be mindful of data privacy concerns and are encouraged to review the privacy policies associated with any models and data storage solutions interfacing with BenchmarkQED.  

It is the user’s responsibility to ensure that the use of BenchmarkQED, the AP News dataset, and the Podcast Transcript Dataset complies with relevant data protection regulations and organizational guidelines. 

 

## License 

MIT License 

 

## Trademarks 

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies. 

 

## Contact 

We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected/offensive behavior in our technology, please contact us at BenchmarkQED@microsoft.com 

If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations. 
