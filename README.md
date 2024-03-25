# openllm-framework
Mathematical Framework Describing the Goal of Open-LLM's for UMN Math Presentation

### The Goal of Encoding

In general, Precision, or Output Precision is a metric used in binary classification tasks to evaluate the accuracy of the positive predictions made by a model. It's defined as:

```math
\text{Output Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
```

Whereas Encoding Precision is fundamentally different. As an analogy, UTF-8 encoding is a way to represent characters in a format that can be stored and processed by computers. 

| Character | Encoding | Representation                      |
|-----------|----------|-------------------------------------|
| A         | UTF-8    | 41 (hex)                            |
| A         | UTF-16   | 0041 (hex)                          |
| 🙂        | UTF-8    | F0 9F 99 82 (hex)                   |
| 🙂        | UTF-16   | D83D DE42 (hex)                     |


Roughly speaking, Output Precision will be some how inversely proportional to the Encoding Errors:

```math
\text{Output Precision} \propto \frac{1}{\text{Encoding Errors}}
```
Which is why Output Precision and Encoding Precision get conflated.
