# Automatic Generative Code with Neural Machine Translation (NMT)

Transformers, including the T5 and MarianMT, have enabled effective understanding and generating complex programming codes. Let's see how!

## Introduction

In our project, “Automatic Generative Code with Neural Machine Translation for Data Security Purposes,” we utilize the T5 Transformer and MarianMT model. Our dataset is derived from the violent-python repository, containing Python code snippets from offensive software and their plain English descriptions. This dataset aids in training the model to translate English Commands into Code snippets.

We further fine-tune the model to automatically generate code, with a focus on data security. This endeavor lies at the intersection of Neural Machine Translation and code creation, contributing towards simpler and safer software development practices.

## Violent-python Dataset

![Dataset Overview](https://sattari.org/wp-content/uploads/2024/03/Dataset-1024x532.png "Violent-python Dataset, Image by Pouya Sattari")

The violent-python dataset is meticulously curated from T.J. O’Connor’s book "Violent Python", presenting a collection of Python code from offensive software, each paired with a corresponding plain English description. Covering diverse areas of offensive security such as penetration testing, forensic analysis, network traffic analysis, and OSINT, the dataset mirrors examples from the book, including exploits, forensic tools, network analysis tools, and social engineering techniques.

With a total of 1,372 samples, the dataset is structured into three subsets: Individual Lines (1,129 pairs), Multi-line Blocks (171 pairs), and Functions (72 pairs). Notably, data security considerations are paramount, given the sensitive nature of offensive programs, emphasizing the need for robust security practices during both dataset curation and model training.

You can find the Dataset on this [GitHub Repository](https://github.com/yourgithub/violent-python) which is Forked from DESSERT Research Lab Github.

## Transformers

Transformers, particularly the T5 model and MarianMT we use, revolutionize machine learning with attention mechanisms and self-attention. These mechanisms enable the effective capture of contextual information, which is essential for understanding and generating complex programming structures.

In our project, encoding captures input code patterns, while decoding generates coherent and syntactically correct code snippets. This architecture significantly enhances our model’s ability to automate code generation while addressing data security concerns.

![Transformers Architecture](https://sattari.org/wp-content/uploads/2024/03/Transformer-Architecture-1024x835.png "Transformers Architecture, Image by Pouya Sattari")


### T5 TRANSFORMER MODEL

The T5 is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks, where each task is converted into a text-to-text format. T5 demonstrates exceptional performance on a variety of tasks out-of-the-box by prepending a different prefix to the input corresponding to each task.

T5 is available in different sizes: t5-small, t5-base, t5-large, t5-3b, and t5-11b.

![T5 transformer](https://sattari.org/wp-content/uploads/2024/03/T5-1024x457.png "T5 Transformer Arch, Image by Pouya Sattari")

### MarianMT

MarianMT, developed by Microsoft, is a transformer-based architecture designed specifically for machine translation. Focusing on multilingual translation tasks, it excels in accurately translating text across different languages. This specialization makes MarianMT a powerful tool for enhancing language-related applications and advancing the field of machine translation.


### Key Distinctions Between T5 and MarianMT

| T5 Architecture | MarianMT Architecture |
|-----------------|-----------------------|
| Developed by Google. | Developed by the language technology research group at Microsoft. |
| General-purpose text-to-text model. | Specifically designed for machine translation tasks. |
| Frames all NLP tasks as converting input text to target text. | Tailored for optimizing translation performance. |
| Designed for a wide range of natural language processing tasks. | Focuses on the task of translating text from one language to another. |
| Capable of handling tasks like summarization, translation, and question-answering within a unified framework. | Specialized in multilingual machine translation and supports various language pairs. |

