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

## DATA COLLECTION AND PREPROCESSING

In the data preprocessing phase, we carefully curated the violent-python dataset, aligning with ethical considerations for offensive programs. With 1,372 samples structured into three subsets, we prioritized data security. Shuffling the data further ensured robust training by minimizing biases associated with the original order. This strategic preprocessing laid the foundation for training a resilient automatic code generation model.

## ZERO-SHOT CLASSIFICATION

In preparation for training, we initialized the tokenizer and model using our model, configuring the system for GPU or CPU usage as specified.

The focus of this initial phase is to assess the model’s baseline performance without fine-tuning. Through a series of zero-shot classifications, we aim to evaluate the model’s capability to achieve our predefined objectives.

This approach allows us to gauge the model’s inherent strengths and weaknesses on generic examples before any specific training modifications are applied, providing a valuable benchmark for subsequent optimization efforts.

![ZERO-SHOT CLASSIFICATION](https://sattari.org/wp-content/uploads/2024/03/Image-from-Presentation-Data-Security-page-8-1024x592.png "ZERO-SHOT CLASSIFICATION, Image by Pouya Sattari")



## GENERIC PURPOSE

In this context, the Python code and accompanying instructions showcase the model’s capability to generate code for general programming purposes.

Utilizing a pre-trained language model and tokenizer, the code accurately responds to instructions like adding two numbers, defining a main function, and declaring a variable of type int.

This suggests its efficacy in providing relevant and coherent code solutions for a wide range of generic programming scenarios.

![GENERIC PURPOSE](https://sattari.org/wp-content/uploads/2024/03/Image-from-Presentation-Data-Security-page-9-1024x762.png "GENERIC PURPOSE, Image by Pouya Sattari")


## SPECIFIC PURPOSE

In the context of our specific task, the provided code iterates through a set of examples, printing both the command and its corresponding ground truth.

The objective is to evaluate the model’s performance on our tailored task, distinct from its pre-trained capabilities. Each command is tokenized and fed into the model, with the generated output compared against the ground truth for assessment.

This process, repeated for five iterations, underscores the unique challenges and nuances our specific task poses to the pre-trained model, emphasizing the need to scrutinize and potentially fine-tune the model to enhance its efficacy in addressing the intricacies of our targeted use case.

![SPECIFIC PURPOSE](https://sattari.org/wp-content/uploads/2024/03/Image-from-Presentation-Data-Security-page-10-1024x807.png
 "SPECIFIC PURPOSE, Image by Pouya Sattari")


## TRANSFER LEARNING AND FINE-TUNING

The code introduces essential parameters for fine-tuning a pre-trained model on a specific task, including the output directory, evaluation strategy, learning rate, and batch sizes.

The Trainer object is then configured with the model, tokenizer, and these parameters, using tokenized datasets for training and validation.

The subsequent trainer.train() call initiates the fine-tuning process, allowing the pre-trained model to adapt to the specifics of our task.

This underscores the crucial role of fine-tuning in optimizing the model for our goals and ensuring its proficiency in addressing the nuances of the targeted use case through iterative training epochs.

![TRANSFER LEARNING AND FINE-TUNING](https://sattari.org/wp-content/uploads/2024/03/Image-from-Presentation-Data-Security-page-11-879x1024.png
 "TRANSFER LEARNING AND FINE-TUNING, Image by Pouya Sattari")

## COMPARISON OF THE MODELS

The presented results showcase the mean similarity scores for zero-shot and fine-tuned classifications using T5 and MarianMT models.

Notably, both models exhibit a comparable trend in zero-shot classification, with mean similarity scores of approximately 32.02 for T5 and 32.06 for MarianMT.

However, the divergence becomes evident after fine-tuning, where MarianMT achieves a substantial increase in accuracy, reaching a mean similarity score of 81.66.

This discrepancy underscores the effectiveness of fine-tuning in significantly enhancing the performance of the MarianMT model for our specific task, suggesting its adaptability and responsiveness to targeted optimizations.

![COMPARISON OF THE MODELS](https://sattari.org/wp-content/uploads/2024/03/Image-from-Presentation-Data-Security-page-11-879x1024.png
 "COMPARISON OF The Two Models (T5, MarianMT), Image by Pouya Sattari")

## Final Result

To sum up, in this project, we demonstrated how to input code descriptions to the Transformer models and receive executable codes with an 81% accuracy by the MarianMT model.

![Final Result](https://sattari.org/wp-content/uploads/2024/03/Screenshot-2024-03-05-at-02.12.04-1024x572.png
 "Final Result, Image by Pouya Sattari")
