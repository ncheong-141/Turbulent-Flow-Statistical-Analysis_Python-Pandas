"""===========================================================================
Turbulent flow statistical analysis using Pandas
Data obtained from a DNS simulation of a channel flow

Spatially homogenous in the streamwise and spanwise directions (x and z)
==========================================================================="""

# Import libraries 
import numpy as np; 
import pandas as pd; 
import scipy.io; 
import scipy.stats as norm;
import time as time; 

import matplotlib.pyplot as plt; 

plt.close('all'); 
# Import data from .mat files and data understanding
if (init == 0) : 
    
    # Time contains the time points at which the velocity values have been recorded.
    # Information was  recorded for 2000 different time points.
    time_mat_data = scipy.io.loadmat("D:\Turbulent data/time.mat"); # [1,2000]

    
    # In each case the information has been recorded in an array of  size (2000,16,8,128)
    # The first array index is the time index (ti). 
    # The second and the third index indicate the streamwise and spanwise number of the  recording location (xi,zi). 
    # The fourth index is the index for the wall ­normal recording locations (yi)
    U_mat_data = scipy.io.loadmat("D:\Turbulent data/U.mat"); # Dimension [2000,16,8,128] [time, xi, zi, yi]
    V_mat_data = scipy.io.loadmat("D:\Turbulent data/V.mat");
    W_mat_data = scipy.io.loadmat("D:\Turbulent data/W.mat");

    # yloc contains the wall ­normal locations at which the velocity values have been recorded.
    # The  wall­normal locations are not uniformly spaced; a higher density of points has been used near the walls.
    yloc_mat_data = scipy.io.loadmat("D:\Turbulent data/yloc.mat"); # [128,1]
    
    # Set init = 1 to not constantly reload initial data. 
   # init = 1; 
    

    # -- Addtional notes --
    # xi,yi and zi element values refer to the indices in other arrays (such as yloc); they are not the actual spatial location.
    # The streamwise and spanwise  locations do not matter since the flow is statistical homogeneous in the streamwise and spanwise direction.


    """===========================================================================
                        Extract raw data and establish indices 
    ==========================================================================="""
    # Index values
    max_index_time  = U_mat_data['U'].shape[0];
    max_index_xi    = U_mat_data['U'].shape[1];
    max_index_zi    = U_mat_data['U'].shape[2]; 
    max_index_yi    = U_mat_data['U'].shape[3];
    
    
    """============================================================================
                                Proccess data for analysis 
    ===============================================================================
    
    Need to convert 4D data to Panda's 2D DataFrame data structure 
    Multi indexing to collapse 4D data down to 2D. 
    rows                        cols 
     0          xi_1    zi_1    [ti,yi]
     1                  zi_2    [ti,yi]
     2                   ...
     nz                 zi_nz   [ti,yi]
     nz+1       xi_2    zi_1    [ti,yi]
                ...         
    """
    """ ==== Insert 2D arrays of raw data into dataframe in [ti,yi] slices ==== """
    
    """
    Timing notes: 
         - Method:  df.append : Appending a 2D array of ti,yi slices from 4D data:             
           Code:    df_U = df_U.append(pd.DataFrame(U_data[:,xi,zi,:]));  
           Time:    12.58 seconds.
           
         - Method:  df.loc : Inserting rows into dataframe using yi slices and pd.loc
           Code:    loop over ti -> df_U.loc[(xi,zi,ti),:] = U_data[ti,xi,zi,:];
           Time:    98.97 - 140 seconds
        
        Can access single elements with: (or df.iat)
           Access data with : df_U.loc[(xi,zi,time), yi]    
    """

    # Initialize empty dataframe structures for appending 2D arrays. 
    df_U = pd.DataFrame(); 
    df_V = pd.DataFrame();
    df_W = pd.DataFrame();
    
    # Start timer for verbose output 
    start = time.time()
    
    # Insert raw velocity data into dataframes 
    for xi in range(0,max_index_xi) : 
        for zi in range(0,max_index_zi) : 
            df_U = df_U.append(pd.DataFrame(U_mat_data['U'][:,xi,zi,:])); 
            df_V = df_V.append(pd.DataFrame(V_mat_data['V'][:,xi,zi,:])); 
            df_W = df_W.append(pd.DataFrame(W_mat_data['W'][:,xi,zi,:])); 
       
    end  = time.time()
    print(f'\n\nTime taken to transfer all raw data to Pandas DateFrames: {end-start:.2f} seconds')
    
    # Transfer yloc and time data to pandas Series data structures (1D)
    s_Yloc = pd.Series(pd.DataFrame(yloc_mat_data['yloc'])[0]); # df needed as data is 2D [128,1]
    s_Time = pd.Series(pd.DataFrame(time_mat_data['time']).iloc[0]);
    
    # Initialize hierarchical index for collapsing 2 dimensions of the 4D data (to input 4D data into the 2D data structure.)
    dimensions_xi_zi = [list(range(0, max_index_xi)), list(range(0,max_index_zi)), list(range(0,max_index_time))];
    
    multi_ind = pd.MultiIndex.from_product(dimensions_xi_zi, names=['xi','zi', 'ti']); 
    
    # Apply multi-index to dataframes. 
    df_U.set_index(multi_ind, append = False, inplace = True); 
    df_V.set_index(multi_ind, append = False, inplace = True); 
    df_W.set_index(multi_ind, append = False, inplace = True); 
    
    

    # Set init = 1 to not constantly reload initial data. 
    init = 1; 


"""============================================================================
                Start of statistical analysis and visualization 
============================================================================"""

# Plot class for holding plot variables (keeps variable explorer clean)
class plot_vars: 
    pass

fig = plt.figure(1)


# -----------------------------------------------------------------------------
## Viewing velocity at different time steps across wall-normal locations at fixed streamwise/spanwise location 
# -----------------------------------------------------------------------------
plot_vars.ax_1 = fig.add_subplot(221);
plot_vars.ax_1.set_title(r'Instantaneous velocity profiles at $x_i,z_i,t_i$  = [8,4,0:4]');
plot_vars.ax_1.set_ylabel(r'Normalized velocity ($u/u_\tau$)');
plot_vars.ax_1.set_xlabel(r'Wall-normal locations ($y/\delta$)')
plot_vars.ax_1.grid(True); 
plot_vars.ax_1.set_xlim(-1,1);
plot_vars.ax_1.set_ylim(0,25); 

for i in range(0,4) :     
    plot_vars.ax_1.plot(s_Yloc,df_U.loc[(8,4,i),slice(0,max_index_yi)], label = f'TS - {i}');
plot_vars.ax_1.legend();


# -----------------------------------------------------------------------------
## Calculating the mean turbulent velocity profile and corresponding laminar profile (from theory)
# -----------------------------------------------------------------------------
plot_vars.ax_2 = fig.add_subplot(222);
plot_vars.ax_2.set_title(r'Comparison of laminar and turbulent mean (in $x_i,z_i,t_i$) velocity profile');
plot_vars.ax_2.set_ylabel(r'Normalized velocity ($u/u_\tau$)');
plot_vars.ax_2.set_xlabel(r'Wall-normal locations ($y/\delta$)')
plot_vars.ax_2.set_xlim(-1,1);
plot_vars.ax_2.set_ylim(0,25); 
plot_vars.ax_2.grid(True); 

## Calculate the mean turbulent profile
# Average the xi, zi (levels in multiindex) (returns df [max_index_time, max_index_yi])
mean_tvp = df_U.mean(level=2); 

# Average all time steps
mean_tvp = mean_tvp.mean(axis=0); 

## Calculate corresponding laminar profile ()
U_bulk = np.trapz(mean_tvp,s_Yloc)/2;    # Bulk velocity. 
u_lam  = (1.5*U_bulk*(1- s_Yloc**2));

# Plot
plot_vars.ax_2.plot(s_Yloc, mean_tvp, 'darkred', label = 'Turbulent profile'); 
plot_vars.ax_2.plot(s_Yloc, u_lam, 'darkgreen', label = 'Laminar profile');
plot_vars.ax_2.legend(); 


# -----------------------------------------------------------------------------
## Viewing the wall normal grid lines 
# -----------------------------------------------------------------------------
plot_vars.ax_3 = fig.add_subplot(212);
plot_vars.ax_3.set_title(r'Wall normal grid lines');
plot_vars.ax_3.set_ylabel(r'Wall-normal locations ($y/\delta$)');
plot_vars.ax_3.set_xlabel(r'$x$');
plot_vars.ax_3.set_xlim(0,1);
plot_vars.ax_3.set_ylim(-1,-0.75); 

# Set up data for plotting 
plot_vars.quater_max_yi = int(max_index_yi/4); 
plot_vars.x = np.linspace(0,1,plot_vars.quater_max_yi); 
plot_vars.yline = np.zeros(plot_vars.quater_max_yi); 

for i in range(0,plot_vars.quater_max_yi):
    
    # Calculate the line and set up list for plotting 
    plot_vars.yline[:] = np.tile(s_Yloc[i],plot_vars.quater_max_yi); 
    
    # Plot the line 
    plot_vars.ax_3.plot(plot_vars.x, plot_vars.yline, 'silver'); 




# -----------------------------------------------------------------------------
## Viewing velocity time graphs 
# -----------------------------------------------------------------------------
plot_vars.fig2 = plt.figure(2)
plot_vars.ax_4 = plot_vars.fig2.add_subplot(221); 
plot_vars.ax_4.set_title(r'Normalized velocity-time signal: $x_i,z_i,y_i$ = [8,4,(0,14,34)]');
plot_vars.ax_4.set_ylabel(r'Normalized velocity ($u/u_\tau$)');
plot_vars.ax_4.set_xlabel(r'Time ($tu_\tau/\delta$)');
plot_vars.ax_4.grid(True); 
plot_vars.ax_4.set_xlim(300,320);
plot_vars.ax_4.set_ylim(0,21);

# Set up and get data from yloc data for plot verbose
plot_vars.list_of_yloc =[0,14,34];
plot_vars.list_of_corresponding_vals = [s_Yloc[0], s_Yloc[14], s_Yloc[34]]; 

for i in range(0,len(plot_vars.list_of_yloc)) : 
    plot_vars.ax_4.plot(s_Time, df_U.loc[(8,4,slice(0,max_index_time)),plot_vars.list_of_yloc[i]], label = f'$y/\delta$ : {plot_vars.list_of_corresponding_vals[i]:.4f}');
plot_vars.ax_4.legend(loc = 'lower right');   


# -----------------------------------------------------------------------------
##              Probability density functions at yi = 0, 14, 34
# -----------------------------------------------------------------------------

## yi = 0
# Using pandas dataframe histogram function (pd.DataFrame.hist)
plot_vars.ax_5 = plot_vars.fig2.add_subplot(222); 
plot_vars.ax_5.set_xlabel(r'Normalized velocity ($u/u_\tau$)');
plot_vars.ax_5.set_ylabel(r'$P(U)/(b - a)$');
plot_vars.ax_5.grid(True); 


# Plot histogram as the PDF (takes all xi,zi and ti readings for column 0 of yloc and inserts into bins)
# The probability of a bin is the area under the bin (matplotlib have the yaxis as P(U)/bin width...)
df_U.hist(column = 0, ax = plot_vars.ax_5, bins = 100, density = True, stacked = True);

# Get statistics 
plot_vars.stats_pdf_yi0 = norm.stats.describe(df_U.loc[(slice(0,max_index_xi),slice(0,max_index_zi),slice(0,max_index_time)),0])
plot_vars.ax_5.set_title(r'Approximate Probability Density Function for $y_i = 0$');

plot_vars.statistics = '\n'.join((
                        r'$Mean: %.2f$'     % (plot_vars.stats_pdf_yi0[2], ),
                        r'$Variance: %.2f$' % (plot_vars.stats_pdf_yi0[3], ),
                        r'$Skewness: %.2f$' % (plot_vars.stats_pdf_yi0[4], )));
plot_vars.ax_5.text(0.05, 0.95, plot_vars.statistics, transform=plot_vars.ax_5.transAxes, verticalalignment='top');


## yi = 14
# Using pandas dataframe histogram function (pd.DataFrame.hist)
plot_vars.ax_6 = plot_vars.fig2.add_subplot(223); 
plot_vars.ax_6.set_xlabel(r'Normalized velocity ($u/u_\tau$)');
plot_vars.ax_6.set_ylabel(r'$P(U)/(b - a)$');
plot_vars.ax_6.grid(True); 

# Plot histogram as the PDF (takes all xi,zi and ti readings for column 0 of yloc and inserts into bins)
# The probability of a bin is the area under the bin (matplotlib have the yaxis as P(U)/bin width...)
df_U.hist(column = 14, ax = plot_vars.ax_6, bins = 100, density = True, stacked = True, color = 'darkorange');

# Get statistics 
plot_vars.stats_pdf_yi14 = norm.stats.describe(df_U.loc[(slice(0,max_index_xi),slice(0,max_index_zi),slice(0,max_index_time)),14])
plot_vars.ax_6.set_title(r'Approximate Probability Density Function for $y_i = 14$');

plot_vars.statistics = '\n'.join((
                        r'$Mean: %.2f$'     % (plot_vars.stats_pdf_yi14[2], ),
                        r'$Variance: %.2f$' % (plot_vars.stats_pdf_yi14[3], ),
                        r'$Skewness: %.2f$' % (plot_vars.stats_pdf_yi14[4], )));
plot_vars.ax_6.text(0.05, 0.95, plot_vars.statistics, transform=plot_vars.ax_6.transAxes, verticalalignment='top');


## yi = 34
# Using pandas dataframe histogram function (pd.DataFrame.hist)
plot_vars.ax_7 = plot_vars.fig2.add_subplot(224); 
plot_vars.ax_7.set_xlabel(r'Normalized velocity ($u/u_\tau$)');
plot_vars.ax_7.set_ylabel(r'$P(U)/(b - a)$');
plot_vars.ax_7.grid(True); 

# Plot histogram as the PDF (takes all xi,zi and ti readings for column 0 of yloc and inserts into bins)
# The probability of a bin is the area under the bin (matplotlib have the yaxis as P(U)/bin width...)
df_U.hist(column = 34, ax = plot_vars.ax_7, bins = 100, density = True, stacked = True, color = 'forestgreen');
plot_vars.ax_7.set_title(r'Approximate Probability Density Function for $y_i = 34$');

# Get statistics 
plot_vars.stats_pdf_yi34 = norm.stats.describe(df_U.loc[(slice(0,max_index_xi),slice(0,max_index_zi),slice(0,max_index_time)),34])

plot_vars.statistics = '\n'.join((
                        r'$Mean: %.2f$'     % (plot_vars.stats_pdf_yi34[2], ),
                        r'$Variance: %.2f$' % (plot_vars.stats_pdf_yi34[3], ),
                        r'$Skewness: %.2f$' % (plot_vars.stats_pdf_yi34[4], )));
plot_vars.ax_7.text(0.05, 0.95, plot_vars.statistics, transform=plot_vars.ax_7.transAxes, verticalalignment='top');
