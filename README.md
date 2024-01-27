# Flash attention NanoGPT

## Overview

This repository implement and optimize the key components of a transformer-based deep neural network that synthesizes Shakespeare text(only CPU). While this DNN is a fairly small model, the basic components of the model are the same as those featured in large-language models (LLMs) that form the basis of technologies like ChatGPT today. Specifically, I implement the attention layer of this model in C++, focusing on optimizations that improve arithmetic intensity, reduce memory footprint, and utilize multi-core and potentially SIMD parallelism on the CPU. This implementation will then be used as part of a complete [NanoGPT](https://github.com/karpathy/nanoGPT) model that
we can run to produce novel Shakespeare-like text.

## Environment Setup

To get started, clone the repo from github:

    git clone https://github.com/ZeusXuan

Run the command below to run inference using a model trained already. You will see some randomly generated Shakespeare text.

     python3 gpt.py part0 --inference -m shakes128

Note that the first time you run the program, it will perform a compilation step that may take a few seconds, you'll see the text `Compiling code into a PyTorch module...`. <br><br>
After this is complete, you'll see some text that begins something like this:

    Running inference using dnn model shakes128
    number of parameters: 0.80M
    Loading meta from data/shakespeare_char/meta.pkl...

    BOTtaps along my lord.

    DUKE OF AUMERLE:
    The this is needs! Camillo, put I will make be strong.

    QUEEN MARGARET:
    My lord, yet t
    -------------------------------------------------------------
    CAMILLO:
    The shadows men sweet thy will burn comes.
    
    FLORIZEL:
    But of appear, good from thy heart
    As I be come of repeal of a w
    -------------------------------------------------------------

Sure, NanoGPT's output may not be literary excellence, but it is still pretty neat! What you see on screen is the output of the standard PyTorch implementation of NanoGPT. Feel free to change to larger sequence lengths by changing the `-m` parameter to larger models like `shakes256`, `shakes1024`, or `shakes2048`. You'll see the performance of NanoGPT token generation slow considerably with the bigger models.

You may experience issues when your compilation randomly starts hanging even though it was working before. When Python JIT compiles your code, it uses locks so multiple threads can compile it as once for efficiency. If you ever compiler your code and it hangs it means that for some reason Python thinks that the lock to your file is held. In order to fix this you can run:

    rm ~/.cache/torch_extensions/py310_cpu/custom_module/lock

We notice that there are many looped-based nondivergent floating point operations. This is a great place to use vector intrinsics! We have provided ISPC support for you to use ISPC for better speedup. 

To enable them in your module.cpp file, all you need to simply uncomment some lines in the file:

    ispc -O3 --target=avx2-i32x8 --arch=x86-64 --pic module.ispc -h module_ispc.h -o module_ispc.o 

## Native Attention with No Optimizations
Run the following test to this native attention layer(Baseline):

    python3 gpt.py part1
 
You can also see this DNN use this native attention layer to generate text, optionally changing the model to `shakes256`, `shakes1024`, or `shakes2048` if you wish to output more text:

    python3 gpt.py part1 --inference -m shakes128

You can find ISPC version in `module.cpp`, just try to uncomment these lines.

## Blocked Matrix Multiply and Unfused Softmax
Here are two opportunities for blocked matrix multiplication here: QK^t and PV. We have utilized blocked matrix multiply on both in order to achieve the better speedup. We use 8x8 submatrix as example.

Run the following test to check your program's correctness:

    python3 gpt.py part2

## Fused Attention
Doing the matrix multiplies and softmax in seperate functions requires that we write each row of our NxN matrix, and then do another pass over this NxN matrix in the subsequent softmax, and then do a third pass over the softmax'd matrix when multipling it by V. Not only is this bad for cache performance, but it is very bad for our program's memory footprint.

Fortunately, we can resolve both issues by "fusing" the calculation, such that we only require one Nx1 temporary vector instead of an NxN temporary matrix.We can do this by observing the following fact. Once we've calculated a single row of the NxN matrix, we are actually ready to softmax that entire row, and we don't have to calculate the rest of the NxN matrix to do so.

Once that row is softmax'd, we can then immediately multiply the softmax'd row by V to fully compute the first row of our attention output (which is of reasonable size: Nxd). In other words, we can calculate just one row of , softmax it, then multiply that softmax's row by V. Doing this does not require creating the NxN matrix...it requires creating only one Nx1 size intermediate vector to hold the first row of and then its softmax. We can then re-use this same Nx1 array to calculate the 2nd row of attention, and then the third, etc. This means that we never materialize the NxN matrix, which is great because that matrix is never used again later in the network anyways.

Run the following test to check your program's correctness:

    python3 gpt.py part3


## Putting it all Together - Flash Attention

### Why Are Matrix Multiply and Softmax Hard to Fuse as Blocks?
The attention formula is very awkward to fuse for a couple reasons. Notice how the formula consists of a matrix multiply, followed by a row-wise calculation from softmax, and concluded with another matrix multiplication. The true thing that makes it difficult from fusing these three operations as blocks is the fact that softmax has to operate on the entire row. So, if we want to bypass this dependency we really have to think outside the box. That is where Flash Attention comes in.

Therefore, for each block, We will multiply `Q` (BLOCKROWSIZE x d) with `K^t`(d x BLOCKCOLUMNSIZE) to get`QK^t` (BLOCKROWSIZE x BLOCKCOLUMNSIZE). Then, we will calculate 
`Softmax(QK^t)` (BLOCKROWSIZE x BLOCKCOLUMNSIZE) and multiply this with `V`(BLOCKCOLUMNSIZE x d) to get `O`(BLOCKROWSIZE x d). Remember, this is an accumulative process just like blocked matrix multiply!

Run the following test to check your program's correctness:

    python3 gpt.py part4

We have given you commandline flags to change the parameters of the attention algorithm. You can do this with the flags `-br <value>` and `-bc <value>`. The default values for each are 256 . For example, if I wanted to change `Br` to 128 and `Bc` to 512I would run:

    python3 gpt.py part4 -br 128 -bc 512
