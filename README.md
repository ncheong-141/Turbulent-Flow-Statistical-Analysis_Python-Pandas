# Turbulent-Flow-Statistical-Analysis_Python-Pandas
Turbulent flow statistical analysis using Python with Panda's data structures for 4D turbulent flow data of a channel flow

The turbulent flow data files are: 
- U.mat (streamwise velocity)
- V.mat (Wall-normal velocity)
- W.mat (Spanwise velocity)
- yloc.mat (wall normal distances) 
- time.mat (time steps) 

The data of the velocity files are in format of [ti, xi, zi, yi] which correspond to velocity at different spatial and temporal locations. The index values of the velocity arrays correspond to index values of another data file (i.e. yi -> yloc.mat).

The data structures used to process the data was the Panda's DataFrame (for the velocities) and Series (for the time and yloc). 
To enable usage of the Panda's DF, the data must be 2 dimensional and multi-indexing (or hierarchical) was used in the format of: 


             |    L1       L2    L3 cols 
     0       |   xi_1    zi_1    [ti,yi]
     1       |           zi_2    [ti,yi]
     2       |            ...
     nz      |           zi_nz   [ti,yi]
     nz+1    |   xi_2    zi_1    [ti,yi]
    
 
 The data was accessed/proccessed using Panda's and Scipy functions for: 
 
 - Instantenous velocity profiles at different time steps. (figure 1) 
    
 - Mean (in xi,zi,ti) turbulent and corresponding laminar velocity profiles (figure 1). 
    
 - Displaying the discretization of the spatial domain for the DNS simulation the wall-normal direction (figure 1). 
    
 - Velocity time signals at different wall normal locations (figure 2). 
    
 - Corresponding Probability Density Functions with statistical values of mean, skewness and variance of the different 
   wall normal locations (figure 2). 
   
   
The figures of the output are in the repo and unfortunately the data files (.mat) are too big to upload onto github. 
