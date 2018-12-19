%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clustering and segmentation of time series (including with regime
% changes) by mixture of gaussian Hidden Markov Models (MixFHMMs) and the EM algorithm; i.e functional data
% clustering and segmentation
%
%
%
%
% by Faicel Chamroukhi, 2009
%
%% Please cite the following references for this code
% 
% @InProceedings{Chamroukhi-IJCNN-2011,
%   author = {F. Chamroukhi and A. Sam\'e  and P. Aknin and G. Govaert},
%   title = {Model-based clustering with Hidden Markov Model regression for time series with regime changes},
%   Booktitle = {Proceedings of the International Joint Conference on Neural Networks (IJCNN), IEEE},
%   Pages = {2814--2821},
%   Adress = {San Jose, California, USA},
%   year = {2011},
%   month = {Jul-Aug},
%   url = {https://chamroukhi.com/papers/Chamroukhi-ijcnn-2011.pdf}
% }
% 
% @PhdThesis{Chamroukhi_PhD_2010,
% author = {Chamroukhi, F.},
% title = {Hidden process regression for curve modeling, classification and tracking},
% school = {Universit\'e de Technologie de Compi\`egne},
% month = {13 december},
% year = {2010},
% type = {Ph.D. Thesis},
% url ={https://chamroukhi.com/papers/FChamroukhi-Thesis.pdf}
% }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear 
close all;
clc;


%% Model specification
K = 3;% number of clusters
R = 3;% number of regimes (HMM states)

% % options
variance_type = 'common';
%variance_type = 'free'; 
ordered_states = 1;
total_EM_tries = 1;
max_iter_EM = 1000;
init_kmeans = 1;
threshold = 1e-6;
verbose = 1; 


%% toy time series with regime changes
load simulated_data.mat
Y; % 
% [n, m] = size(Y);
% n: nbr of time sries
% m: number of observations


%%
mixFHMM =  learn_MixFHMM(Y, K , R, ...
    variance_type, ordered_states, total_EM_tries, max_iter_EM, init_kmeans, threshold, verbose);        
 
%%
show_mixFHMM_results(Y, mixFHMM)


 






