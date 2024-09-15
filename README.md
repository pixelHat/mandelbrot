# Mandelbrot Implementation using StarPU (CPU Only)

This repository contains a Mandelbrot set implementation using the StarPU parallel programming library. The program divides the computation of the Mandelbrot set across multiple tasks running on the CPU and renders the result as an ASCII chart in the terminal.
Features
- Parallel computation using StarPU.
- CPU-only implementation (no GPU support).
- Outputs the Mandelbrot set as an ASCII chart in the terminal.

# Prerequisites

To compile and run the code, you need the following installed on your system:

- GCC (GNU Compiler Collection)
- StarPU (version 1.4 or later)
- pkg-config
- Math library (-lm)

# Compilation

To compile the Mandelbrot program, run the following command in your terminal:

```bash
gcc -o mandelbrot mandelbrot.c $(pkg-config --libs --cflags starpu-1.4) -lm
```

This will generate an executable file.

# Usage

Once compiled, run the program to render the Mandelbrot set:

```bash
./mandelbrot
```

The output will be displayed as an ASCII chart in the terminal, illustrating the Mandelbrot set.
Code Structure

# How It Works

- Task Creation: The Mandelbrot set is computed in parallel by breaking the work into small tasks handled by StarPU.
- Computation: Each task calculates whether each point in the complex plane belongs to the Mandelbrot set.
- Rendering: The result is rendered as an ASCII chart directly in the terminal.
