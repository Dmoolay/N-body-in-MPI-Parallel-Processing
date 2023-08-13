#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
//===============================================================================================================
#define N 992
#define SIZE 100.0
#define LIGHT_YEAR 9e15
#define G 6.674e-11
#define M_SUN 2e30
#define V_AVG 200000.0
#define TIME_STEP 10e10
#define TOTAL_TIME_STEPS 6000
//===============================================================================================================
typedef struct {
    double x;
    double y;
    double vx;
    double vy;
} Body;
//===============================================================================================================
// Function to save x or y coordinates of all bodies to a file
void saveXCoordinatesToFile(Body* ARR, int numPoints, const char* filename,int k) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Failed to create the file.\n");
        return;
    }
    int i;    
    for (i = 0; i < numPoints; i++) {
        if(k==0){
        fprintf(file, "%f", ARR[i].x);
        }else{
             fprintf(file, "%f", ARR[i].y);
             }
        // Add a comma if it's not the last coordinate
         if (i != numPoints - 1) {
            fprintf(file, ",");
         }
    }
    fclose(file);
}

// Function to initialize the bodies with random positions and velocities
void initialize(Body* bodies, int rank, int size, int start, int end1) {
    int i;
    for (i = start; i < end1; i++) {
        // Generate random positions within the defined size
        bodies[i].x = (SIZE ) * (double)rand() / RAND_MAX;
        bodies[i].y = (SIZE ) * (double)rand() / RAND_MAX;

        // Generate random speed within the range of 0.5 * V_AVG to 1.5 * V_AVG
        double speed = 0.5 * V_AVG + ((double)rand() / RAND_MAX) * V_AVG;
        speed = speed * TIME_STEP / LIGHT_YEAR;  // Convert the speed to a suitable scale for the simulation
        double angle = 2.0 * M_PI * (double)rand() / RAND_MAX;   // Generate a random angle

        // Calculate the velocity components based on speed and angle
        bodies[i].vx = speed * cos(angle);
        bodies[i].vy = speed * sin(angle);
    }
}

// Function to compute the forces between bodies
void compute_forces(Body* bodies, double* forces_x, double* forces_y, int rank, int size, int start, int end1) {
    int i, j;
    for (i = start; i < end1; i++) {
        forces_x[i] = forces_y[i] = 0.0;
        for (j = 0; j < N; j++) {
            if (i != j) {
                // Calculate the distance components between bodies (meter)
                double dx = LIGHT_YEAR * (bodies[j].x - bodies[i].x);
                double dy = LIGHT_YEAR * (bodies[j].y - bodies[i].y);
                double dist = sqrt(dx * dx + dy * dy);
                // Ensure a minimum distance of LIGHT_YEAR to avoid division by zero
		        if(dist < 1000000){ dist = 1000000;}

                // Calculate the gravitational force between bodies using Newton's law of universal gravitation
                double force = G * M_SUN * M_SUN / (dist * dist) ;
                // Calculate the force components based on distance and force magnitude
                // Accumulate forces acting on the body
                forces_x[i] += force * (dx/dist);
                forces_y[i] += force * (dy/dist);
            }
        }
    }
}

// Function to update the positions and velocities of the bodies
void update_positions_velocities(Body* bodies, double* forces_x, double* forces_y, double dt, int rank, int size, int start, int end1) {
    int i;
    double tempx, tempy, tempvx, tempvy;

    for (i = start; i < end1; i++) {
        // Calculate the acceleration components
        double ax = forces_x[i] / M_SUN;
        double ay = forces_y[i] / M_SUN;

        // Convert velocity units to meter per second
        tempvx = bodies[i].vx * LIGHT_YEAR / TIME_STEP;
        tempvy = bodies[i].vy * LIGHT_YEAR / TIME_STEP;
        // Update temporary variables for position and velocity
	    tempvx += ax * dt;
        tempvy += ay * dt;
        // Calculate the position change based on the updated velocity
        tempx = tempvx * dt;
        tempy = tempvy * dt;
        // Convert the position change to units of light year
        tempx = tempx/9e15;
	    tempy = tempy/9e15;

        // Update the position and velocity of the body
	    bodies[i].x += tempx;
        bodies[i].y += tempy;
        bodies[i].vx = bodies[i].vx + (tempvx * dt / LIGHT_YEAR);
        bodies[i].vy = bodies[i].vy + (tempvy * dt / LIGHT_YEAR);
        //===============================================================================================================
        // Handle boundary conditions if they exceed the simulation space
	    if (bodies[i].x < 0.0){
            bodies[i].x += SIZE;
            bodies[i].vx = -(0.5 * V_AVG + ((double)rand() / RAND_MAX) * V_AVG) * TIME_STEP / LIGHT_YEAR;
	    }
        if (bodies[i].x > SIZE){
            bodies[i].x -= SIZE;
	        bodies[i].vx = (0.5 * V_AVG + ((double)rand() / RAND_MAX) * V_AVG) * TIME_STEP / LIGHT_YEAR;
	    }
        if (bodies[i].y < 0.0){
            bodies[i].y += SIZE;
	        bodies[i].vy = -(0.5 * V_AVG + ((double)rand() / RAND_MAX) * V_AVG) * TIME_STEP / LIGHT_YEAR;
	    }
        if (bodies[i].y > SIZE){
            bodies[i].y -= SIZE;
	        bodies[i].vy = (0.5 * V_AVG + ((double)rand() / RAND_MAX) * V_AVG) * TIME_STEP / LIGHT_YEAR;
	    }
    }
}

int main(int argc, char** argv) {
    //===============================================================================================================
    // Parameter Initialization
    int rank, size, t, i, seed;
    Body* bodies;
    double* forces_x;
    double* forces_y;
    double start_time,end_time;
    //===============================================================================================================
    // MPI Initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    //===============================================================================================================
    int extra, offset;
    int start = 0;
    int end = 0;
    // send different seed to every process
    if (rank == 0) {// If the current process is the master process
        start_time = MPI_Wtime();// Get the start time for measuring elapsed time
        srand(12);// Set the random seed for generating points
        for (i = 1; i < size; i++) {// Loop over the remaining processes
            seed = rand();// Generate a different random seed for each process
            MPI_Send(&seed, 1, MPI_INT, i, 0, MPI_COMM_WORLD);// Send the seed to the corresponding process
        }
    } else {// If the current process is not the master process
        MPI_Recv(&seed, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);// Receive the seed from the master process
        srand(seed);// Set the random seed for generating points
    }
    //===============================================================================================================
    // Compute the range of bodies for each process (start & end)
    offset = N / size;
    if (size == 6){
        extra = N % size;
        offset = N / size;
        if (rank == 5){
            start = rank * offset;
            offset += 2;
            end = start + offset + extra;
        }
        else{
            start = rank * offset;
            end = start + offset;
        }
    }
    else{
        offset = N / size;
        start = rank * offset;
        end = start + offset;
    }
    //===============================================================================================================
    // Allocate memory for bodies and forces arrays
    bodies = (Body*)malloc(N * sizeof(Body));
    forces_x = (double*)malloc(N * sizeof(double));
    forces_y = (double*)malloc(N * sizeof(double));
    //===============================================================================================================
    // Initialize bodies with random positions and velocities
    initialize(bodies, rank, size, start, end);
    // Gather updated bodies to all process
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_BYTE, bodies, offset * sizeof(Body), MPI_BYTE, MPI_COMM_WORLD);

    // Save (x,y) initial-coordinates of bodies to a file
    if (rank == 0){
        saveXCoordinatesToFile(bodies ,N ,"start_x.txt", 0);
        saveXCoordinatesToFile(bodies ,N ,"start_y.txt", 1);
    }
    //===============================================================================================================
    // Loop over time steps
    for (t = 0; t < TOTAL_TIME_STEPS; t++) {
        // Compute forces between bodies
        compute_forces(bodies, forces_x, forces_y, rank, size, start, end);

        // Update positions and velocities
        update_positions_velocities(bodies, forces_x, forces_y, TIME_STEP, rank, size, start, end);

        // Save (x,y) middle-coordinates of bodies to a file
        if (rank == 0 && (t == TOTAL_TIME_STEPS/2)){
            saveXCoordinatesToFile(bodies ,N ,"mid_x.txt", 0);
            saveXCoordinatesToFile(bodies ,N ,"mid_y.txt", 1);
     	}
        // Gather updated bodies to all process
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_BYTE, bodies, offset * sizeof(Body), MPI_BYTE, MPI_COMM_WORLD);
   }
    //===============================================================================================================
    // Save (x,y) final-coordinates of bodies to a file
    saveXCoordinatesToFile(bodies ,N ,"final_x.txt", 0);
    saveXCoordinatesToFile(bodies ,N ,"final_y.txt", 1);
    //===============================================================================================================
    // Deallocate memory and finalize MPI
    free(bodies);
    free(forces_x);
    free(forces_y);
    //===============================================================================================================
    if(rank == 0 ){
       end_time = MPI_Wtime();
       printf("Time taken = %f seconds\n", end_time - start_time);
    }
    //===============================================================================================================
    MPI_Finalize();
    return 0;
}

