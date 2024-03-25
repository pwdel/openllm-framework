# openllm-framework
Mathematical Framework Describing the Goal of Open-LLM's for UMN Math Presentation

### The Goal of Encoding

Physical resources in the Universe are scarce. At this point in time, there is a lot of demand for Graphical Processing Units (GPU's) which are used for a wide variety of Artificial Intelligence computational tasks such as cryptocurrency mining, image recognition, large language model training.

The goal of encoding is to compress information such that we can get outputs which are overall more pleasing or acceptable to humans from the fixed amount of computational resources in the Universe, in this case, the limited number of GPU's.

```math
\begin{flalign*}
&\text{Let:}\\
&U \text{: The utility or satisfaction derived from the outputs of computational tasks.}\\
&R \text{: The total available computational resources, measured in GPU-hours.}\\
&D \text{: Demand for computational tasks, reflecting the volume and complexity of tasks.}\\
&E \text{: The efficiency of encoding algorithms. Higher values indicate more efficient encoding.}\\
\end{flalign*}
```

```math
\begin{align*}
&\text{The goal is to maximize } U \text{, given by the equation:}\\
&U = \frac{E \times R}{D}
\end{align*}
```


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

Given:

- $n$: Number of bytes processed per token.
- $T$: Total number of bytes in the input text.

The time complexity can be described using Big O notation as $O\left(\frac{T}{n}\right)$. This notation is used to express the upper bound of the algorithm's running time, suggesting how the time to process the text grows as the size of the input ($T$) increases, relative to the number of bytes processed per token ($n$).

Similarly, if the time complexity is described using Theta notation as $\Theta\left(\frac{T}{n}\right)$, it indicates the exact asymptotic behavior of the algorithm. This means the algorithm's running time grows at this precise rate as the input size increases, implying a tight bound where the running time is directly proportional to $\frac{T}{n}$.

