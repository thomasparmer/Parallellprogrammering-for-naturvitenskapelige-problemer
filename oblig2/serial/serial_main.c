#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


double** allocate_matrix(int rows, int cols)
{
	double *data = (double *)malloc(rows*cols*sizeof(double));
	double **array= (double **)malloc(rows*sizeof(double*));
	int i;
	for (i=0; i<rows; i++)
		array[i] = &(data[cols*i]);

	return array;
}


void deallocate_matrix(double*** matrix) {
	free((*matrix)[0]);
	free(*matrix);
}


void read_matrix_binaryformat(char* filename, double*** matrix, int* num_rows, int* num_cols)
{
	FILE* fp = fopen(filename,"rb");
	fread(num_rows, sizeof(int), 1, fp);
	fread(num_cols, sizeof(int), 1, fp);
	
	/* storage allocation of the matrix */
	*matrix = allocate_matrix(*num_rows, *num_cols);
	
	/* read in the entire matrix */
	fread((*matrix)[0], sizeof(double), (*num_rows)*(*num_cols), fp);
	fclose(fp);
}


void write_matrix_binaryformat(char* filename, double** matrix, int num_rows, int num_cols)
{
	FILE *fp = fopen (filename, "wb");
	fwrite (&num_rows, sizeof(int), 1, fp);
	fwrite (&num_cols, sizeof(int), 1, fp);
	fwrite (matrix[0], sizeof(double), num_rows*num_cols, fp);
	fclose (fp);
}


void matrix_multiplication(double **matrix_a, double **matrix_b, double ***matrix_c, int num_rows_a, int num_cols_a, int num_rows_b, int num_cols_b)
{
	int i, j, k;
	for (i = 0; i < num_rows_a; ++i) {
		for (j = 0; j < num_cols_b; ++j) {
			for (k = 0; k < num_cols_a; ++k) { 
				(*matrix_c)[i][j] += matrix_a[i][k] * matrix_b[k][j];
			}
		}
	}
}


void transposed_matrix_multiplication(double **matrix_a, double **matrix_b, double ***matrix_c, int num_rows_a, int num_cols_a, int num_rows_b, int num_cols_b)
{
	int i, j, k;
	for (i = 0; i < num_rows_a; ++i) {
		for (j = 0; j < num_rows_b; ++j) {
			for (k = 0; k < num_cols_a; ++k) { 
				(*matrix_c)[i][j] += matrix_a[i][k] * matrix_b[j][k];
			}
		}
	}
}

/* Transposing matrix for better cache lineup */
void transpose_matrix(double ***matrix, int *num_rows, int *num_cols) 
{
	double **tmp_matrix = allocate_matrix(*num_cols, *num_rows);
	double **tmp_swap = *matrix;
	int tmp = *num_rows;
	int i, j;

	for (i = 0; i < *num_rows; ++i) {
		for (j = 0; j < *num_cols; ++j) {
			tmp_matrix[j][i] = (*matrix)[i][j];
		}
	}

	*num_rows = *num_cols;
	*num_cols = tmp;
	*matrix = tmp_matrix;
	deallocate_matrix(&tmp_swap);
}



int main(int argc, char *argv[])
{
	if (argc != 4) {
		printf("argc = %d\n", argc);
		printf("Use: %s [infile a][infile b][outfile]\n", argv[0]);
		return -1;
	}

	clock_t begin, end;
	double time_spent;
	begin = clock();


	double **matrix_a;
	double **matrix_b;
	double **matrix_c;

	int num_rows_a;
	int num_cols_a;
	int num_rows_b;
	int num_cols_b;


	read_matrix_binaryformat(argv[1], &matrix_a, &num_rows_a, &num_cols_a);
	read_matrix_binaryformat(argv[2], &matrix_b, &num_rows_b, &num_cols_b);
	matrix_c = allocate_matrix(num_rows_a, num_cols_b);
	memset(&(matrix_c[0][0]), 0, sizeof(matrix_c[0][0]) * num_rows_a * num_cols_b);

	
	transpose_matrix(&matrix_b, &num_rows_b, &num_cols_b);
	transposed_matrix_multiplication(matrix_a, matrix_b, &matrix_c, num_rows_a, num_cols_a, num_rows_b, num_cols_b);
	//matrix_multiplication(matrix_a, matrix_b, &matrix_c, num_rows_a, num_cols_a, num_rows_b, num_cols_b);


	write_matrix_binaryformat(argv[3], matrix_c, num_rows_a, num_rows_b); //switch to num_cols_b without transpose num_rows_b with;

	deallocate_matrix(&matrix_a);
	deallocate_matrix(&matrix_b);
	deallocate_matrix(&matrix_c);

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("%f seconds used in serial implementation\n", time_spent);
	return 0;
}
