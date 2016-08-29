#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define MASTER 0


/* allocating contagious 2d array */
double** allocate_matrix(int rows, int cols)
{
    double *data = (double *)malloc(rows*cols*sizeof(double));
    double **array= (double **)malloc(rows*sizeof(double*));
    int i;

    #pragma omp parallel for
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


/* Transposes matrix for good cache lineup */
void transpose_matrix(double ***matrix, int *num_rows, int *num_cols) 
{
    double **tmp_matrix = allocate_matrix(*num_cols, *num_rows);
    int tmp = *num_rows;
    int i, j;

    #pragma omp parallel for private(j)
    for (i = 0; i < *num_rows; ++i) {
	for (j = 0; j < *num_cols; ++j) {
	    tmp_matrix[j][i] = (*matrix)[i][j];
	}
    }

    *num_rows = *num_cols;
    *num_cols = tmp;
    deallocate_matrix(&(*matrix));
    *matrix = tmp_matrix;
}


/* each process defines its region size */ 
void partition_region(int num_procs, int my_rank, int l, int *start, int *size, int *bsize, int *offset)
{
    *bsize =  l/num_procs;
    *offset = l%num_procs;
    *size = my_rank != MASTER ? *bsize : *bsize + *offset;
    *start = my_rank == MASTER ? 0 : (*bsize * my_rank) + *offset;
}


/*each process finds its predecessor and successor */
void find_partner(int num_procs, int my_rank, int *from, int *to)
{
    *from = my_rank > MASTER ? my_rank-1 : num_procs-1;
    *to = my_rank+1 < num_procs ? my_rank+1 : MASTER;
}


/* Each process sends its region og b to its successor */
void comunicate_b_matrix(int my_rank, int from, int to, int num_rows_b, int num_cols_b, int offset, int count, int tmp_count, double ***matrix_b) 
{
    double **tmp_matrix;
    int tmp;
    if (my_rank % 2 == 0) {
	tmp = num_rows_b;
	if (count == MASTER) {
	    tmp += offset;
	}

	MPI_Send(&((*matrix_b)[0][0]), tmp*num_cols_b, MPI_DOUBLE, to, 1, MPI_COMM_WORLD);
	tmp = num_rows_b;
	if (tmp_count == MASTER) {
	    tmp_matrix = allocate_matrix(num_rows_b + offset, num_cols_b);
	    tmp += offset;
	} else {
	    tmp_matrix = allocate_matrix(num_rows_b, num_cols_b);
	}

	MPI_Recv(&(tmp_matrix[0][0]), tmp*num_cols_b, MPI_DOUBLE, from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	deallocate_matrix(&(*matrix_b));
	*matrix_b = tmp_matrix;
	return;
    }

    tmp = num_rows_b;
    if (tmp_count == MASTER) {
	tmp_matrix = allocate_matrix(num_rows_b + offset, num_cols_b); 
	tmp += offset;
    } else {
	tmp_matrix = allocate_matrix(num_rows_b, num_cols_b); 
    }

    MPI_Recv(&(tmp_matrix[0][0]), tmp*num_cols_b, MPI_DOUBLE, from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    tmp = num_rows_b;
    if (count == MASTER) {
	tmp += offset;
    }

    MPI_Send(&((*matrix_b)[0][0]), tmp*num_cols_b, MPI_DOUBLE, to, 1, MPI_COMM_WORLD);
    deallocate_matrix(&(*matrix_b));
    *matrix_b = tmp_matrix;
}


/* finding count. count is used to find the correct position in c matrix during multiplication */
void update_count(int num_procs, int *count)
{
    *count = *count > MASTER ? *count-1 : num_procs-1;
}


/* each process asks process 0 to distribute a region */ //lag struct?
void distribute_region(int num_procs, int my_rank, int size, int start, int num_cols_a, int num_cols_b, double ***matrix_a, double ***matrix_b) 
{
    int info[2];
    int i;
    if (my_rank == MASTER) {
	for (i = 1; i < num_procs; i++) {
	    MPI_Recv(&info, 2, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    MPI_Send(&((*matrix_a)[info[1]][0]), info[0]*num_cols_a, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
	    MPI_Send(&((*matrix_b)[info[1]][0]), info[0]*num_cols_b, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
	}
	return;
    }
    info[0] = size;
    info[1] = start;
    MPI_Send(&info, 2, MPI_INT, MASTER, 1, MPI_COMM_WORLD);
    MPI_Recv(&((*matrix_a)[0][0]), info[0]*num_cols_a, MPI_DOUBLE, MASTER, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&((*matrix_b)[0][0]), info[0]*num_cols_b, MPI_DOUBLE, MASTER, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}


/* multiplaying the matrices */
void matrix_multiplication(int count, double **matrix_a, double **matrix_b, double ***matrix_c, int l, int size, int bsize, int offset)
{
    int i, j, k, pos, tmp;
    if (count != 0) {
	tmp = bsize;
	pos = (bsize * count) + offset;
    } else {
	pos = 0;
	tmp = bsize + offset;
    }

    #pragma omp parallel for private(j, k)
    for (i = 0; i < size; ++i) {
        for (j = 0; j < tmp; ++j) {
            for (k = 0; k < l; ++k) { 
		(*matrix_c)[i][j+pos] += matrix_a[i][k] * matrix_b[j][k];
            }
        }
    }
}


/* Putting all the regions back together for the final result */
void combine_regions(int num_procs, int my_rank, int l, int start, int num_rows_b, double ***matrix_c)
{
    int size[2];
    if (my_rank != MASTER) {
	size[0] = l;
	size[1] = start;	  
	MPI_Send(&size, 2, MPI_INT, MASTER, 1, MPI_COMM_WORLD);
	MPI_Send(&((*matrix_c)[0][0]), size[0]*num_rows_b, MPI_DOUBLE, MASTER, 1, MPI_COMM_WORLD);
    } else {
	int i;
	for (i = 1; i < num_procs; i++) {
	    MPI_Recv(&size, 2, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    MPI_Recv(&((*matrix_c)[size[1]][0]), size[0]*num_rows_b, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
    }
}



int main(int argc, char *argv[])
{
    if (argc != 4) {
	printf("argc = %d\n", argc);
	printf("Use: mpirun -np [cores] %s [infile a][infile b][outfile]\n", argv[0]);
	return -1;
    }

    int i;
    int my_rank, num_procs;
    int start, size, bsize, from, to, count, offset, tmp_count;
    int num_rows_a, num_cols_a, num_rows_b, num_cols_b;
    double t1, t2;
    double **matrix_a, **matrix_b, **matrix_c;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == MASTER)
	t1 = MPI_Wtime(); 

    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (my_rank == MASTER) {
	read_matrix_binaryformat(argv[1], &matrix_a, &num_rows_a, &num_cols_a);
	read_matrix_binaryformat(argv[2], &matrix_b, &num_rows_b, &num_cols_b);
	transpose_matrix(&matrix_b, &num_rows_b, &num_cols_b);		
	matrix_c = allocate_matrix(num_rows_a, num_rows_b);		
	memset(&(matrix_c[0][0]), 0, sizeof(matrix_c[0][0]) * num_rows_a * num_rows_b);
    }

    MPI_Bcast(&num_rows_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_cols_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_rows_b, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_cols_b, 1, MPI_INT, 0, MPI_COMM_WORLD);
    partition_region(num_procs, my_rank, num_rows_a, &start, &size, &bsize, &offset);
    if (my_rank != MASTER) {
	matrix_a = allocate_matrix(bsize, num_cols_a);
	matrix_b = allocate_matrix(bsize, num_cols_b);
    }

    distribute_region(num_procs, my_rank, size, start, num_cols_a, num_cols_b, &matrix_a, &matrix_b);
    if (my_rank != MASTER) {
	matrix_c = allocate_matrix(size, num_rows_b);
	memset(&(matrix_c[0][0]), 0, sizeof(matrix_c[0][0]) * size * num_rows_b);
    }

    find_partner(num_procs, my_rank, &from, &to);
    count = my_rank;
    tmp_count = count;
    for (i = 0; i < num_procs; i++) {
	matrix_multiplication(count,  matrix_a, matrix_b, &matrix_c, num_cols_a,  size, bsize, offset);
	update_count(num_procs, &tmp_count);
	comunicate_b_matrix(my_rank, from, to, bsize, num_cols_b, offset, count, tmp_count, &matrix_b);
	update_count(num_procs, &count);
    }

    combine_regions(num_procs, my_rank, size, start, num_rows_b, &matrix_c);
    if (my_rank == MASTER) {
	write_matrix_binaryformat(argv[3], matrix_c, num_rows_a, num_rows_b);
	t2 = MPI_Wtime(); 
	printf( "Elapsed time is %f\n", t2 - t1 ); 
    }

    deallocate_matrix(&matrix_b);
    deallocate_matrix(&matrix_c);
    deallocate_matrix(&matrix_a);
    MPI_Finalize();
    return 0;
}
