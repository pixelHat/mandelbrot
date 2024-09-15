#include <stdio.h>
#include <complex.h>
#include <starpu.h>

#define ROWS 63
#define COLS 100
#define ITER 2000

/**
 * @brief Determines if a complex number remains stable under iteration of the Mandelbrot function.
 *
 * This function iterates the Mandelbrot equation: `z = z^2 + c`, starting with `z = 0`
 * for a given complex number `c`. After the specified number of iterations, it checks if
 * the magnitude (absolute value) of the resulting complex number `z` is less than or equal
 * to 2. If the magnitude exceeds 2 at any point, the sequence is considered unstable.
 *
 * @param c The complex number to test for stability.
 * @param iter The number of iterations to apply the Mandelbrot function.
 * @return int Returns 1 if the complex number is stable (i.e., its magnitude is <= 2 after the iterations),
 *         or 0 if it is unstable (magnitude exceeds 2).
 */
int is_stable(double complex c, int iter) {
	double complex z = 0;
	for (int i = 0; i < iter; i++) {
		z = z * z + c;
	}
	return cabs(z) <= (double) 2.0;
}

/**
 * @brief Fills an array with complex numbers representing a grid in the complex plane.
 *
 * This function generates a grid of complex numbers over a specified range of real and
 * imaginary components. The real part ranges from -2.0 to 0.5, and the imaginary part
 * ranges from -1.5 to 1.5. The array is filled such that each element corresponds to
 * a point on the grid defined by the `ROWS` and `COLS` constants, representing the
 * dimensions of the complex plane.
 *
 * @param array A pointer to the array where the complex numbers will be stored. The array
 *        should have a size of `ROWS * COLS`.
 *
 * @note The filled array is typically used as input to check the stability of each complex number
 *       for the Mandelbrot set, determining whether each point lies within the set based on
 *       iterative calculations.
 */
void fill_array(double complex* array) {
	double real_start = -2.0;
	double real_end = 0.5;
	double imag_start = -1.5;
	double imag_end = 1.5;

	// Determine the step for real and imaginary components
	double real_step = (real_end - real_start) / (COLS - 1);
	double imag_step = (imag_end - imag_start) / (ROWS - 1);

    // Fill the array with complex numbers
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			double real_part = real_start + j * real_step;
			double imag_part = imag_start + i * imag_step;
            array[i * COLS + j] = real_part + imag_part * I;
		}
	}
}

/**
 * @brief Prints a simple ASCII chart representing the Mandelbrot set.
 *
 * This function takes an array of integers that represent the results of Mandelbrot
 * set calculations and prints a visual representation of the set using ASCII characters.
 * Each element in the array corresponds to a point on a 2D grid, where `1` indicates
 * that the corresponding complex number is part of the Mandelbrot set (or within a stable
 * boundary), and `0` indicates that it is outside the set.
 *
 * The function prints a `.` character for stable points (value `1`), representing points
 * that belong to the Mandelbrot set, and a space (` `) for unstable points (value `0`),
 * representing points outside the set.
 *
 * @param array A pointer to an integer array that contains the Mandelbrot set results.
 *        The array should be of size `ROWS * COLS`, where each element corresponds to a
 *        point on the complex plane grid.
 *
 * @note This function provides a very basic ASCII visualization of the Mandelbrot set
 *       in a terminal or console. Each line printed corresponds to a row of points in
 *       the complex plane, while each character printed corresponds to a column.
 */
void print_chart(int* array) {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int value = array[i * COLS + j];
            if (value) {
                printf(".");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }
}

/**
 * @brief StarPU CPU kernel function for computing Mandelbrot set stability for a specific point.
 *
 * This function is designed to be executed as a StarPU task on the CPU. It computes whether
 * a given complex number `c` is stable based on the Mandelbrot iteration and stores the result
 * in a provided array. The complex number `c` is unpacked from the task's arguments, and the
 * result is stored at the appropriate index `i` in the provided result vector.
 *
 * The stability of the complex number is determined using the `is_stable` function, which
 * checks if the number remains bounded under Mandelbrot iterations (with a predefined maximum
 * iteration count, `ITER`). If the point is stable (i.e., part of the Mandelbrot set), the
 * result will be `1`; otherwise, it will be `0`.
 *
 * @param buffers[] A StarPU buffer containing the result vector in which the stability
 *                  result will be stored. `buffers[0]` should point to a `starpu_vector_interface`
 *                  structure, and the result is stored in its internal array at index `i`.
 * @param cl_arg A pointer to the arguments passed to the task, which includes the complex
 *               number `c` to evaluate and the index `i` where the result will be stored.
 *               These arguments are unpacked using `starpu_codelet_unpack_args`.
 */
void cpu_func(void *buffers[], void *cl_arg) {
    double complex c;
    int i;
    starpu_codelet_unpack_args(cl_arg, &c);
    starpu_codelet_unpack_args(cl_arg, &i);

    struct starpu_vector_interface *vector = buffers[0];
    int *val = (int *)STARPU_VECTOR_GET_PTR(vector);
    int r = is_stable(c, ITER);
    val[i] = is_stable(c, ITER);
}

static struct starpu_codelet cl = {
  .cpu_funcs = {cpu_func},
  .nbuffers = 1,
  .modes = {STARPU_R},
};

/**
 * @brief Main function for computing and visualizing the Mandelbrot set using parallel tasks with StarPU.
 *
 * This function initializes the necessary data structures for computing the Mandelbrot set, registers
 * the data with StarPU, and submits tasks for parallel computation of the set's stability across a grid
 * of complex numbers. After the computations are completed, it prints a simple ASCII representation of
 * the Mandelbrot set.
 *
 * The function allocates memory for a matrix of complex numbers (`matrix`) representing points in the
 * complex plane, and for a mask (`mask`) to store the stability results (1 for stable points, 0 for
 * unstable). It then:
 *  - Fills the `matrix` with complex numbers using `fill_array`.
 *  - Registers the `mask` array with StarPU to hold task results.
 *  - Iterates over all points in the complex plane grid, submitting StarPU tasks to compute the Mandelbrot
 *    set stability for each complex number. Each task uses `cpu_func` to compute stability and store
 *    the result in the `mask`.
 *  - Waits for all tasks to complete, then unregisters the StarPU data handle and shuts down StarPU.
 *  - Prints the Mandelbrot set using `print_chart` to display an ASCII visualization.
 *
 * Memory for both the complex number matrix and the result mask is freed at the end of the function.
 *
 * @note We are using a 1D array (`matrix`) to represent the complex number grid instead of a 2D array.
 *       The matrix is accessed using row-major order, where the element at position `(i, j)` in a
 *       2D array is accessed as `i * COLS + j` in the 1D array.
 *
 * @return int Returns 0 on successful execution, or 1 if a memory allocation fails.
 */
int main(void) {
    double complex *matrix = malloc(ROWS * COLS * sizeof(double complex));
    if (matrix == NULL) {
        perror("malloc");
        return 1;
    }

    int *mask = malloc(ROWS * COLS * sizeof(int));
    if (mask == NULL) {
        perror("malloc");
        return 1;
    }

    fill_array(matrix);

    starpu_init(NULL);
    starpu_data_handle_t matrix_handle;
    starpu_vector_data_register(&matrix_handle, 0, (uintptr_t)mask, ROWS * COLS, sizeof(int));

    for (int i = 0; i < ROWS * COLS; i++) {
        double complex c = matrix[i];
        starpu_task_insert(
            &cl, STARPU_R, matrix_handle,
            STARPU_VALUE, &c, sizeof(double complex),
            STARPU_VALUE, &i, sizeof(int),
            0);
    }

    starpu_task_wait_for_all();
    starpu_data_unregister(matrix_handle);
    starpu_shutdown();

    print_chart(mask);

    free(matrix);
    free(mask);
    return 0;
}
