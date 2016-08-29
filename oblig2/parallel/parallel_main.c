#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define MASTER 0


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


/* each process defines its region size */ 
void partition_region(int num_procs, int my_rank, int l, int *start, int *size) 
{
  *size = my_rank < num_procs-1 ? l/num_procs :  l/num_procs + l%num_procs;
  *start = (l/num_procs) * my_rank;
}


/* each process asks process 0 to distribute a region */ //lag struct?
void distribute_region(int num_procs, int my_rank, int size, int start, int num_cols_a, int num_rows_b, int num_cols_b, double ***matrix_a, double ***matrix_b) 
{
  int info[2];
  int i;
  if (my_rank == MASTER) {
    for (i = 1; i < num_procs; i++) {
      MPI_Recv(&info, 2, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&((*matrix_a)[info[1]][0]), info[0]*num_cols_a, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
      MPI_Send(&((*matrix_b)[0][0]), num_rows_b*num_cols_b, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
    }
  } else {
    info[0] = size;
    info[1] = start;
    MPI_Send(&info, 2, MPI_INT, MASTER, 1, MPI_COMM_WORLD);
    MPI_Recv(&((*matrix_a)[0][0]), info[0]*num_cols_a, MPI_DOUBLE, MASTER, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&((*matrix_b)[0][0]), num_rows_b*num_cols_b, MPI_DOUBLE, MASTER, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}


void master_region(double ***matrix_a, double ***matrix_b, int num_rows_a, int num_rows_b, int size) 
{
  double **tmp_a = allocate_matrix(num_rows_a, size);
  double **tmp_b = allocate_matrix(num_rows_b, size);
  memcpy(&(tmp_a[0][0]), &((*matrix_a)[0][0]), num_rows_a * size * sizeof(double));
  memcpy(&(tmp_b[0][0]), &((*matrix_b)[0][0]), num_rows_b * size * sizeof(double));
  //ikke ferdig
}


void matrix_multiplication(int num_procs, double **matrix_a, double **matrix_b, double ***matrix_c, int num_rows_a, int num_cols_a, int num_rows_b, int num_cols_b, int start, int size)
{
  int i, j, k;
  for (i = 0; i < num_rows_a/num_procs; ++i) {
    for (j = 0; j < num_rows_b; ++j) {
      for (k = 0; k < num_cols_a; ++k) { 
        (*matrix_c)[i][j] += matrix_a[i][k] * matrix_b[j][k];
      }
    }
  }
}


void combine_regions(int num_procs, int my_rank, int l, int start, int num_rows_a, int num_rows_b, double **sub_matrix_c, double ***matrix_c)
{
  int size[2];
  if (my_rank != MASTER) {
    size[0] = l;
    size[1] = start;	  
    MPI_Send(&size, 2, MPI_INT, MASTER, 1, MPI_COMM_WORLD);
    MPI_Send(&(sub_matrix_c[0][0]), size[0]*num_rows_b, MPI_DOUBLE, MASTER, 1, MPI_COMM_WORLD);
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

  clock_t begin, end;
  double time_spent;
  begin = clock();


  int my_rank, num_procs;
  int start, size;
  int num_rows_a, num_cols_a, num_rows_b, num_cols_b;
  double **matrix_a, **matrix_b, **matrix_c;


  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
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

  partition_region(num_procs, my_rank, num_rows_a, &start, &size);

  if (my_rank != MASTER) {
    matrix_a = allocate_matrix(size, num_cols_a);
    matrix_b = allocate_matrix(num_rows_b, num_cols_b);
  }
  

  distribute_region(num_procs, my_rank, size, start, num_cols_a, num_rows_b, num_cols_b, &matrix_a, &matrix_b);
  //master_region(&matrix_a, &matrix_b, num_rows_a, num_rows_b, size);


  if (my_rank == MASTER) {
    matrix_multiplication(num_procs, matrix_a, matrix_b, &matrix_c, num_rows_a, num_cols_a, num_rows_b, num_cols_b, start, size);
  }

  
  if (my_rank != MASTER) {
    matrix_c = allocate_matrix(size, num_rows_b);
    memset(&(matrix_c[0][0]), 0, sizeof(matrix_c[0][0]) * size * num_rows_b);
    matrix_multiplication(num_procs, matrix_a, matrix_b, &matrix_c, num_rows_a, num_cols_a, num_rows_b, num_cols_b, start, size);
    combine_regions(num_procs, my_rank, size, start, num_rows_a, num_rows_b, matrix_c, NULL);
  }

  if (my_rank == MASTER) {
    combine_regions(num_procs, my_rank, size, start, num_rows_a, num_rows_b, NULL, &matrix_c);
    write_matrix_binaryformat(argv[3], matrix_c, num_rows_a, num_rows_b);
    

    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("%f seconds used in parallel implementation\n", time_spent);

  }


  deallocate_matrix(&matrix_b);
  deallocate_matrix(&matrix_c);
  deallocate_matrix(&matrix_a);



  MPI_Finalize();
  return 0;
}
