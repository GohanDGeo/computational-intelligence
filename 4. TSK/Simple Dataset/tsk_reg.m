%%
% Koutroumpis Georgios, AEM 9668
% COMPUTATIONAL INTELLIGENCE
% ECE AUTh 2022
% Project 2, TSK
%%
close all
clear
clc

% Open file to save metrics in
fid = fopen( 'metrics.txt', 'wt' );
%% Load data and normalize it
data = importdata("airfoil_self_noise.dat");

X = data(:,1:end-1);
Y = data(:,end);
X = normalize(X);
data = cat(2, X, Y);

%% Split to train, validation and test sets
num_data = size(data,1);
num_features = size(data,2);

[train_idx, val_idx, test_idx] = dividerand(num_data, 0.6, 0.2, 0.2);

data_train = data(train_idx,:);
X_train = data_train(:,1:end-1);
Y_train = data_train(:,end);

data_val = data(val_idx,:);
X_val = data_val(:,1:end-1);
Y_val = data_val(:,end);

data_test = data(test_idx,:);
X_test = data_test(:,1:end-1);
Y_test = data_test(:,end);

%% Train the 4 models
mf_types = ["constant", "constant", "linear", "linear"];
num_mf = [2, 3, 2, 3];

for i=1:4
    %% Set options and generate FIS
    gen_opt = genfisOptions("GridPartition", ...
                            "InputMembershipFunctionType", "gbellmf", ...
                            "NumMembershipFunctions", num_mf(i), ...
                            "OutputMembershipFunctionType", mf_types(i));
    tsk_model = genfis(X_train, Y_train, gen_opt);
    
    %% Train the FIS 
    an_opt = anfisOptions("InitialFis", tsk_model, ...
                          "ValidationData", [X_val Y_val], ...
                          "EpochNumber", 100, ...
                          "OptimizationMethod", 1);
    [fis,trainError,stepSize,valFIS,valError] = anfis([X_train Y_train],...
                                                       an_opt);
                                                   
    y_pred = evalfis(valFIS, X_test);
    [rmse, nmse, ndei, r2] = get_metrics(Y_test, y_pred);
    
    fprintf(fid, ...
            'Model %d \n RMSE: %f\n NMSE: %f\n NDEI: %f\n R2:%f\n', ...
            i, rmse, nmse, ndei, r2);
    %% Plots
    plot_requirements(tsk_model, valFIS,...
                            Y_test, y_pred, trainError, valError, i)
end

fclose(fid);

