package edu.coursera.distributed;

import edu.coursera.distributed.util.MPI;
import edu.coursera.distributed.util.MPI.MPIException;

/**
 * A wrapper class for a parallel, MPI-based matrix multiply implementation.
 */
public class MatrixMult {
    /**
     * A parallel implementation of matrix multiply using MPI to express SPMD
     * parallelism. In particular, this method should store the output of
     * multiplying the matrices a and b into the matrix c.
     *
     * This method is called simultaneously by all MPI ranks in a running MPI
     * program. For simplicity MPI_Init has already been called, and
     * MPI_Finalize should not be called in parallelMatrixMultiply.
     *
     * On entry to parallelMatrixMultiply, the following will be true of a, b,
     * and c:
     *
     *   1) The matrix a will only be filled with the input values on MPI rank
     *      zero. Matrix a on all other ranks will be empty (initialized to all
     *      zeros).
     *   2) Likewise, the matrix b will only be filled with input values on MPI
     *      rank zero. Matrix b on all other ranks will be empty (initialized to
     *      all zeros).
     *   3) Matrix c will be initialized to all zeros on all ranks.
     *
     * Upon returning from parallelMatrixMultiply, the following must be true:
     *
     *   1) On rank zero, matrix c must be filled with the final output of the
     *      full matrix multiplication. The contents of matrix c on all other
     *      ranks are ignored.
     *
     * Therefore, it is the responsibility of this method to distribute the
     * input data in a and b across all MPI ranks for maximal parallelism,
     * perform the matrix multiply in parallel, and finally collect the output
     * data in c from all ranks back to the zeroth rank. You may use any of the
     * MPI APIs provided in the mpi object to accomplish this.
     *
     * A reference sequential implementation is provided below, demonstrating
     * the use of the Matrix class's APIs.
     *
     * @param a Input matrix
     * @param b Input matrix
     * @param c Output matrix
     * @param mpi MPI object supporting MPI APIs
     * @throws MPIException On MPI error. It is not expected that your
     *                      implementation should throw any MPI errors during
     *                      normal operation.
     */
    public static void parallelMatrixMultiply(Matrix a, Matrix b, Matrix c,
            final MPI mpi) throws MPIException {

        // Current computing node id.
        final int currRank = mpi.MPI_Comm_rank(mpi.MPI_COMM_WORLD);

        // Number of computing nodes in cluster.
        final int numRanks = mpi.MPI_Comm_size(mpi.MPI_COMM_WORLD);

        final int numRows = c.getNRows();
        final int chunkSize = (numRows + numRanks - 1) / numRanks;
        final int startRow = currRank * chunkSize;
        int endRow = (currRank + 1) * chunkSize;
        if(endRow > numRows) endRow = numRows;

        // Broadcast matrices a and b to all other ranks FROM RANK 0.
        mpi.MPI_Bcast(a.getValues(), 0, a.getNRows() * a.getNCols(), 0, mpi.MPI_COMM_WORLD);
        mpi.MPI_Bcast(b.getValues(), 0, b.getNRows() * b.getNCols(), 0, mpi.MPI_COMM_WORLD);

        // matrix-multiplication.
        for (int i = startRow; i < endRow; i++) {
            for (int j = 0; j < c.getNCols(); j++) {
                c.set(i, j, 0.0); // init value as 0.
                for (int k = 0; k < b.getNRows(); k++)
                    c.incr(i, j, a.get(i, k) * b.get(k, j));
            }
        }

        // Rank-0 will combine the results
        // else send the result to rank 0.

        if(currRank != 0) {
            // Blocking SEND operation.
            mpi.MPI_Send(c.getValues(), // values
                    startRow * c.getNCols(), // offset index
                    (endRow - startRow) * c.getNCols(), // num of elements to send
                    0, // destination rank
                    currRank, // source rank
                    mpi.MPI_COMM_WORLD);
        } else {
            MPI.MPI_Request[] requests = new MPI.MPI_Request[numRanks - 1];

            // Start at 1 since we don't need request from rank-0.
            for(int i = 1; i < numRanks; ++i) {
                final int rankStartRow = i * chunkSize;
                int rankEndRow = (i + 1) * chunkSize;
                if(rankEndRow > numRows) rankEndRow = numRows;

                final int rowOffset = rankStartRow * c.getNCols();
                final int numElements = (rankEndRow - rankStartRow) * c.getNCols();

                requests[i - 1] = mpi.MPI_Irecv(c.getValues(),
                        rowOffset,
                        numElements,
                        i,
                        i,
                        mpi.MPI_COMM_WORLD
                );
            }

            mpi.MPI_Waitall(requests);
        }
    }
}
