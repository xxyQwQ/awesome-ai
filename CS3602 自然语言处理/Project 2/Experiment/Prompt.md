# Exploring Large Language Models for Chinese Spoken Language Understanding



## Prompt Engineering

### Plain Input

**GPT-4 English**

> **User Input**
>
> input sentence: 附近火锅店在什么位置
> semantic triples: 
>
> **Model Output**
>
> The input sentence "附近火锅店在什么位置" can be broken down into the following semantic triples:
> (附近火锅店, 在, 什么位置)
> Subject: 附近火锅店 (nearby hotpot restaurant)
> Predicate: 在 (is located at)
> Object: 什么位置 (what location)

### Precise Description

**GPT-4 English**

> **User Input**
>
> Interpret the given sentence into semantic triples, namely the form of "<act>(<slot>=<value>)".
> Note that <act> should be either "inform" or "deny", <slot> can be "操作", "对象", "起点", "终点", "途径", “位置", "方式", "偏好".
> If there are multiple semantic triples, output them separately instead of including multiple slots in one action.
> The given sentence is: 附近火锅店在什么位置
>
> **Model Output**
>
> inform(操作=查询)
> inform(对象=火锅店)
> inform(位置=附近)

### Chain-of-Thought

**GPT-4 English**

> **User Input**
>
> Interpret the given sentence into semantic triples, namely the form of "<act>(<slot>=<value>)". Note that <act> should be either "inform" or "deny", <slot> can be "操作", "对象", "起点", "终点", "途径", “位置", "方式", "偏好". If there are multiple semantic triples, output them separately instead of including multiple slots in one action.
> The given sentence is: 附近火锅店在什么位置
> Let's think step by step.
>
> **Model Output**
>
> The given sentence "附近火锅店在什么位置" translates to a request for information about the location of a nearby hotpot restaurant. Here are the semantic triples:
> inform(对象=附近火锅店)
> inform(位置=什么位置)

### Emotional Stimulation

**GPT-4 English**

> **User Input**
>
> You are an experienced servant and never misunderstand the intention of guests.
> Your task is to interpret the given sentence into semantic triples, namely the form of "<act>(<slot>=<value>)". Note that <act> should be either "inform" or "deny", <slot> can be "操作", "对象", "起点", "终点", "途径", “位置", "方式", "偏好". If there are multiple semantic triples, output them separately instead of including multiple slots in one action.
> The given sentence is: 附近火锅店在什么位置
>
> **Model Output**
>
> inform(对象=火锅店)
> inform(位置=附近)

### Chinese Prompt

**GPT-4 Chinese**

> **User Input**
>
> 将指定句子解析为语义三元组，即“<act>(<slot>=<value>)”的形式，其中<act>只能为“inform”或“deny”，<slot>可以是“操作”、“对象”、“起点”、“终点”、“途径”、“位置”、“方式”、“偏好”。如果句子包含多个语义三元组，请将它们分别输出，不要在一个<act>中包含多个<slot>。
> 句子：附近火锅店在什么位置
>
> **Model Output**
>
> inform(对象=火锅店)
> inform(位置=附近)



## Learning from Samples

### Zero-shot

**GPT-4 English**

> **User Input**
>
> Interpret the given sentence into semantic triples, namely the form of "<act>(<slot>=<value>)". Note that <act> should be either "inform" or "deny", <slot> can be "操作", "对象", "起点", "终点", "途径", “位置", "方式", "偏好". If there are multiple semantic triples, output them separately instead of including multiple slots in one action.
> The given sentence is: 我要去娄桥高速收费站
>
> **Model Output**
>
> inform(操作=去)
> inform(对象=娄桥高速收费站)

### One-shot

**GPT-4 English**

> **User Input**
>
> Interpret the given sentence into semantic triples, namely the form of "<act>(<slot>=<value>)". Note that <act> should be either "inform" or "deny", <slot> can be "操作", "对象", "起点", "终点", "途径", “位置", "方式", "偏好". If there are multiple semantic triples, output them separately instead of including multiple slots in one action.
> The given sentence is: 导航到安德第六地税所拍照
> The semantic triples are: inform(操作=导航), inform(终点=安德第六地税所)
> The given sentence is: 我要去娄桥高速收费站
>
> **Model Output**
>
> The semantic triples are: inform(操作=去), inform(终点=娄桥高速收费站)

### Few-shot

**GPT-4 English**

> **User Input**
>
> Interpret the given sentence into semantic triples, namely the form of "<act>(<slot>=<value>)". Note that <act> should be either "inform" or "deny", <slot> can be "操作", "对象", "起点", "终点", "途径", “位置", "方式", "偏好". If there are multiple semantic triples, output them separately instead of including multiple slots in one action.
> The given sentence is: 导航到安德第六地税所拍照
> The semantic triples are: inform(操作=导航), inform(终点=安德第六地税所)
> The given sentence is: 我要走宁夏回族自治区党委大院
> The semantic triples are: inform(操作=导航), inform(途径=宁夏回族自治区党委大院)
> The given sentence is: 附近火锅店在什么位置
> The semantic triples are: inform(对象=火锅店), inform(位置=附近)
> The given sentence is: 我要去娄桥高速收费站
>
> **Model Output**
>
> The semantic triples are: inform(操作=导航), inform(终点=娄桥高速收费站)



## Cross-model Comparison

### ChatGPT

> **User Input**
>
> Interpret the given sentence into semantic triples, namely the form of "<act>(<slot>=<value>)". Note that <act> should be either "inform" or "deny", <slot> can be "操作", "对象", "起点", "终点", "途径", “位置", "方式", "偏好". If there are multiple semantic triples, output them separately instead of including multiple slots in one action.
> The given sentence is: 附近火锅店在什么位置
> 
> **Model Output**
>
> The given sentence "附近火锅店在什么位置" can be interpreted into the following semantic triples:
>inform(对象=火锅店, 位置=附近)
> inform(对象=火锅店, 位置=什么位置)

### Claude

> **User Input**
>
> Interpret the given sentence into semantic triples, namely the form of "<act>(<slot>=<value>)". Note that <act> should be either "inform" or "deny", <slot> can be "操作", "对象", "起点", "终点", "途径", “位置", "方式", "偏好". If there are multiple semantic triples, output them separately instead of including multiple slots in one action.
> The given sentence is: 附近火锅店在什么位置
> 
> **Model Output**
>
> inform(位置=附近)

### LLaMA

> **User Input**
>
> Interpret the given sentence into semantic triples, namely the form of "<act>(<slot>=<value>)". Note that <act> should be either "inform" or "deny", <slot> can be "操作", "对象", "起点", "终点", "途径", “位置", "方式", "偏好". If there are multiple semantic triples, output them separately instead of including multiple slots in one action.
> The given sentence is: 附近火锅店在什么位置
> 
> **Model Output**
>
> Sure, I'd be happy to help! Here are the semantic triples for the given sentence:
>inform(位置="附近")
> inform(对象="火锅店")
> So, the sentence can be interpreted as "Inform me about the location of a hot pot restaurant nearby."

### ChatGLM

> **User Input**
>
> Interpret the given sentence into semantic triples, namely the form of "<act>(<slot>=<value>)". Note that <act> should be either "inform" or "deny", <slot> can be "操作", "对象", "起点", "终点", "途径", “位置", "方式", "偏好". If there are multiple semantic triples, output them separately instead of including multiple slots in one action.
> The given sentence is: 附近火锅店在什么位置
> 
> **Model Output**
>
> The sentence “附近火锅店在什么位置” can be interpreted as a query asking for the location of a nearby hotpot restaurant. In the form of semantic triples, this would be:
>inform(位置=附近火锅店的位置)

### Ernie

> **User Input**
>
> Interpret the given sentence into semantic triples, namely the form of "<act>(<slot>=<value>)". Note that <act> should be either "inform" or "deny", <slot> can be "操作", "对象", "起点", "终点", "途径", “位置", "方式", "偏好". If there are multiple semantic triples, output them separately instead of including multiple slots in one action.
> The given sentence is: 附近火锅店在什么位置
> 
> **Model Output**
>
> inform(位置=附近火锅店的位置)

### Qwen

> **User Input**
>
> Interpret the given sentence into semantic triples, namely the form of "<act>(<slot>=<value>)". Note that <act> should be either "inform" or "deny", <slot> can be "操作", "对象", "起点", "终点", "途径", “位置", "方式", "偏好". If there are multiple semantic triples, output them separately instead of including multiple slots in one action.
> The given sentence is: 附近火锅店在什么位置
> 
> **Model Output**
>
> inform(位置=附近)
>inform(对象=火锅店)