# openllm-framework
Mathematical Framework Describing the Goal of Open-LLM's for UMN Math Presentation

### The Goal of Encoding

Physical resources in the Universe are scarce. At this point in time, there is a lot of demand for Graphical Processing Units (GPU's) which are used for a wide variety of Artificial Intelligence computational tasks such as cryptocurrency mining, image recognition, large language model training.

The goal of encoding is to compress information such that we can get outputs which are overall more pleasing or acceptable to humans from the fixed amount of computational resources in the Universe, in this case, the limited number of GPU's.

```math
\begin{flalign*}
&\text{Let:}\\
& U \text{: The utility or satisfaction derived from the outputs of computational tasks.} &\\
& R \text{: The total available computational resources, measured in GPU-hours.} &\\
& D \text{: Demand for computational tasks, reflecting the volume and complexity of tasks.} &\\
& E \text{: The efficiency of encoding algorithms. Higher values indicate more efficient encoding.} &
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

This brings up Big O notation (O) which is used to describe the upper bound of an algorithm's running time. One might say that for a particular task, we only will allow a certain number of seconds or hours and assign that the notation O, and anything beyond that would be called Theta $\Theta\left(\frac{T}{n}\right)$.

Within Large Language Models (LLM's), the smallest unit of text for a given application is called a, "token." Tokens can be a words, parts of a words (such as a syllable or a subword), or even punctuation, depending on how the model's input is tokenized. Tokenization is the process of breaking down text into these manageable pieces (tokens) so that it can be read and written by a computer. So that being said, Encoding in the context of tokens is a way to translate human language to computer bits and bytes.

That being said, there are many ways to tokenize language, and so different tokenization systems are going to process different amounts of bytes. Also the amount of text that a task has to go through can vary - one task may be to have a computer read an entire book, or another task may be to have a computer just read a paragraph. Hence we introduce a couple other new variables:

```math
\begin{flalign*}
&\text{Let:}\\
& n \text{: Number of bytes processed per token.}  &\\
& T \text{: Total number of bytes in the input text.} &
\end{flalign*}
```

That is to say, if the tokenization system being used holds a lot of bytes per token, analogous to using UTF-16 rather than UTF-8 shown above, then n is going to be larger. If we're having a computer read an entire book rather than just a paragraph, then T is going to be larger.

So that being said, we can express the upper bound of an algorithm's running time with:


```math
\begin{align*}
&O\left(\frac{T}{n}\right)
\end{align*}
```

Going beyond O would give us the dreaded $\Theta\left(\frac{T}{n}\right)$, which means, "it took too long."

### An Aside - Avoiding the CPU

Managing memory requirements is crucial for maintaining optimal performance, especially when processing large batches of text tokens with limited GPU memory. When the required memory exceeds the available GPU memory, settings within the HuggingFace library for LLM's allow a fallback to CPU and system memory, leading to a significant decrease in performance. There are ways to adjust one's settings to not allow a fallback, but this results in an error, so the operation doesn't get performed.

Drawing off of our Big O equation above, we'll define some variables to express this:

```math
\begin{flalign*}
&\text{Let:}\\
& R_{\text{{TOTALMEM}}} \text{: be the total memory requirement for processing a batch of text.} &\\
& G_{\text{{GPUMEM}}} \text{: be the available GPU memory.} &\\
& M_{\text{{PERTOKEN}}} \text{: represent the memory required for each token.} &\\
& B \text{: be the batch size.} &\\
& O \text{: represent Big O Notation, taking into account that we only have so much time to complete the task.} &
\end{flalign*}
```
To communincate that we can't allow the total memory requirement exceed our time requirement, O we can simply say:

```math
\begin{align*}
&\text{Let } R_{\text{{TOTALMEM}}} = B \cdot M_{\text{{PERTOKEN}}} ∣ \left( O \right) &\\
\end{align*}
```

As mentioned above, saying, "O" is not enough though, because there are built-in functions within the HuggingFace library which allow fallback to a CPU. In some instances this might be acceptable and not completely invalidate O, the time it takes to perform an operation. So here we just mention that, hypothetically there might be an acceptable performance P and a factor $\alpha$ which might represent a degree of performance degredation that is acceptable.

```math
\begin{flalign*}
&\text{Let:} &\\
&P_{\text{opt}}: \text{The optimal performance when } R_{\text{{TOTALMEM}}} \leq G_{\text{{GPUMEM}}}. &\\
&P_{\text{fall}}(R,G): \text{The performance when } R > G, \text{ a function of } R \text{ and } G \text{ meaning degraded performance due to using the CPU/RAM.} &\\
&\alpha: \text{A constant representing the degree of performance degradation when falling back to CPU and RAM.} &
\end{flalign*}
```

```math
P(R_{\text{{TOTALMEM}}}, G_{\text{{GPUMEM}}}) = 
\begin{cases} 
P_{\text{opt}} & \text{if } R \leq G \\
\alpha \cdot P_{\text{fall}}(R, G) & \text{if } R > G
\end{cases}
```

Tying this all together with our above equation, our overall Big O Notation equation is inversely proportional to the Performance due to CPU/MEM fallback, which should be fairly intuitive - if the machine exceeds its system resources, then performance will suffer.

```math
\begin{align*}
O\left( \frac{T}{n} \right) \propto \frac{1}{P(R, G)}
\end{align*}
```
But by how much will performance suffer? This is difficult to generalize, because it really depends upon what is being done. However, since we are in the age of, "Large Language Models," with the key word being, "Large," it's probably safe to assume that for a lot of the newer stuff, this means models with lots of parameters, or tasks involving lots of text, will involve orders of magnitude slowdowns.

### Customizing Your Large Language Model - Fine Tuning, Training

Consider that all of the above applies to both:

* Using an existing LLM to provide text, e.g. feeding in some task and performing an, "inference,"
* As well as fine-tuning an LLM, (also referred to as pre-training) e.g. customizing some of the parameters in a model that can be customized, such that specific types of outputs will be given with specific types of inputs when performing an inference.

Merely using a hammer is quite different than customizing a hammer, making it bigger, putting a claw on the back, putting a rubber tip on the front and so on. Just as in inference, in training an LLM, all of the Demand, Efficiency and Resource factors are still constraints talked about in the, "Goal of Encoding," section above.

#### An Example of Non-Encoding Technique to Adapt Fine-Tuning

LoRA stands for "Low-Rank Adaptation of Large Language Models" or, “Layer-wise Learning Rate Adaptation” within the context of Hugging Face's Diffusers documentation. This is a technique to build efficiency within the fine-tuning, training and adaptation phase of diffusion models (the broader term for large language models and other probabilistic models such as image generators.

To understand what, "Layer-wise," means, one first must accept that underlying LLM's are [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network), which are a sort of mathematical transform in which nodes pass data to another node, the second node applies a weight, and then passes on again to an additional node and so on. If one imagines a gigantic spreadsheet, with the first column containing a vector of starting data, the next multiplying that virst vector by something, and then the third multiplying on that second vector and so on for many columns, this would be a super simple analogy of how a neural network works.

So one could reasonably imagine that there is a computational time involved with applying a weighted factor from one column to the next. While intuitively based upon how we see computers work, one might think that these computations happen instantaneously, but really there is always a bit of a delay, even if it's in the nanoseconds, and with sufficiently large, "spreadsheets," this delay becomes larger and larger. We'll call this delay, "η" (eta).

Now, suppose one could actually tweak that learning rate from layer to layer, so that some layers will process faster than others. This would mean that if we're going to fine-tune the parameters, we could do it, "faster," using this special, "LoRA," method.

So for a given layer in a neural network (analogous to a column in a spreadsheet), the transformation can be described as follows:

```math
\begin{flalign*}
&\text{Let:}\\
&y \text{: be the output, e.g. the result of the layer operation} &\\
&W \text{: be the weight matrix of the layer, the amount being factored or transformed} &\\
&x \text{: be the input to the layer, which could have been an output from a previous layer} &\\
&b \text{: be the bias term, which is an adjustable constant value } &
\end{flalign*}
```

So we use a 

```math
\begin{align*}
&y = Wx + b &
\end{align*}
```

So then, we use Adaptation, the A in "Layer-wise Learning Rate Adaptation," to adapt the W, by adding the AB matrix elementwise.

```math
\begin{flalign*}
&\text{Where:} &\\
&W \text{: remain the original, pre-trained weight matrix, which is kept frozen.} &\\
&AB \text{: represent the low-rank adaptation to the original weights, capturing the essence of changes} &\\
&\text{needed for adapting to the new task.} &\\
&A \text{ and } B \text{: be the only parameters that are updated during the fine-tuning process.} &
\end{flalign*}
```

So our new result can be expressed by Matrix operation, where A and B are the low-rank matrices introduced as part of the LoRA technique, where their product AB is also a matrix, with the same dimensions of W:

```math
\begin{align*}
&y' = (W + AB)x + b &
\end{align*}
```
This approach allows for efficient fine-tuning because it maintains the general capabilities learned during pre-training, it provides a mechanism to adapt the model to specific tasks with minimal adjustments, represented by the low-rank matrices.

LoRA's efficiency comes from AB. In traditional LLM training, W is trained directly.


#### HuggingFace Functionality Example

In our discussion on weights and biases above, we were fairly hand-wavy because the purpose of this document is to describe the goal of encoding, to describe why from a software development perspecitive someone may undergo the exercise of actual training. If one were going to do that, assuming you had a nice environment set up, with all of the proper dependencies installed, and you have torch linking to a GPU, some pseudocode demonstrating how LoRA training is the following:

```
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
# Assuming CustomGPT2Model is your model class adjusted for LoRA
from your_model_file import CustomGPT2Model

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = CustomGPT2Model.from_pretrained('gpt2')

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="your_dataset.txt",
    block_size=128,
)

def custom_optimizer(params):
    return bnb.optim.Adam8bit(params, lr=5e-5)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=100,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    optimizers=(custom_optimizer, None),  # Custom optimizer and no scheduler
)

trainer.train()
```

Assuming you had a proper dataset, and the model architecture you are working with is GPT2-based, the above format would be basically how one would employ LoRA. All of the mathematical stuff described 
