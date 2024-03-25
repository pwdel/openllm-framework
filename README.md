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

The goal then is to maximize U, given the equation:

```math
\begin{align*}
&max_U &: \quad U = \frac{E \times R}{D}
\end{align*}
```

### Efficiency Through Precision

We can't control Demand (D), and we only may be able to access a given small R at any given time due to scarce resources, so ultimately the only thing we can do to maximize how acceptable the outputs we get as defined by people, U, is to manipulate E, the efficiency of encoding algorithms. This brings up the notion of precision.

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
Which is why Output Precision and Encoding Precision get conflated - generally having better Encoding Precision leads to better Output Precision, and both may be wrapped up together in the concept we introduced above, "E," which is the efficiency of encoding algorithms.

### Introducing a Way to Define Time, Big O and Theta

Implicit in the above but not mentioned is the time we have to run things on computers. If we had infinite time, none of the above would matter, we could just take a tiny computing resource and just set it to solve whatever we want it to solve beyond the heat death of the Universe. But obviously this is fanciful and there are competitive pressures in life, so we may not even have a week or even a few days to compute somthing, it might need to be instantanous or within an hour or so.

This brings up Big O notation (O) which is used to describe the upper bound of an algorithm's running time. You might say that for a particular task, we only will allow a certain number of seconds or hours and assign that the notation O, and anything beyond that would be called Theta $\Theta\left(\frac{T}{n}\right)$.

Within Large Language Models (LLM's), the smallest unit of text for a given application is called a, "token." Tokens can be a words, parts of a words (such as a syllable or a subword), or even punctuation, depending on how the model's input is tokenized. Tokenization is the process of breaking down text into these manageable pieces (tokens) so that it can be read and written by a computer. So that being said, Encoding in the context of tokens is a way to translate human language to computer bits and bytes.

That being said, there are many ways to tokenize language, and so different tokenization systems are going to process different amounts of bytes. Also the amount of text that a task has to go through can vary - one task may be to have a computer read an entire book, or another task may be to have a computer just read a paragraph. Hence we introduce a couple other new variables:

Given:

- $n$: Number of bytes processed per token.
- $T$: Total number of bytes in the input text.

That is to say, if the tokenization system being used holds a lot of bytes per token, analogous to using UTF-16 rather than UTF-8 shown above, then n is going to be larger. If we're having a computer read an entire book rather than just a paragraph, then T is going to be larger.

So that being said, we can express the upper bound of an algorithm's running time with:


```math
\begin{align*}
&O\left(\frac{T}{n}\right)
\end{align*}
```

Going beyond O would give us the dreaded $\Theta\left(\frac{T}{n}\right)$, which means, "it took too long."

### Avoiding the CPU

Managing memory requirements is crucial for maintaining optimal performance, especially when processing large batches of text tokens with limited GPU memory. When the required memory exceeds the available GPU memory, the system experiences a fallback to CPU and system memory, leading to a significant decrease in performance.

```math
\begin{flalign*}
Let G be the available GPU memory.

Let M represent the memory required for each token.

Let B be the batch size.
\end{flalign*}
```

```math
\begin{align*}
R = B \cdot M \cdot \left( \frac{T}{n} \right)
\end{align*}
```

```math
\begin{flalign*}
&\text{Let:} &\\
&P_{\text{opt}}: \text{The optimal performance when } R \leq G. &\\
&P_{\text{fall}}(R,G): \text{The performance when } R > G, \text{ a function of } R \text{ and } G \text{ representing the degraded performance due to the fallback to CPU and RAM.} &\\
&\alpha: \text{A constant representing the degree of performance degradation when falling back to CPU and RAM, our FallbackFade.} &
\end{flalign*}
```

```math
P(R, G) = 
\begin{cases} 
P_{\text{opt}} & \text{if } R \leq G \\
\alpha \cdot P_{\text{fall}}(R, G) & \text{if } R > G
\end{cases}
```
