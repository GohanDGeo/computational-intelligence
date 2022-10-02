%%
% Koutroumpis Georgios, AEM 9668
% COMPUTATIONAL INTELLIGENCE
% ECE AUTh 2022
% Project 2, TSK

%%
close all
clear
clc
%% Load data and normalize it
data = readmatrix("superconductivity.csv");
X = data(:,1:end-1);
Y = data(:,end);
X = normalize(X,'range',[-1,1]);
data = cat(2, X, Y);

%% Split to train, validation and test sets
num_data = size(data,1);
num_features = size(data,2);
[train_idx, val_idx, test_idx] = dividerand(num_data, 0.6, 0.2, 0.2);

data_train = data(train_idx,:);
X_train = data_train(:,1:end-1);
Y_train = data_train(:,end);
num_train = size(data_train,1);

data_val = data(val_idx,:);
X_val = data_val(:,1:end-1);
Y_val = data_val(:,end);

data_test = data(test_idx,:);
X_test = data_test(:,1:end-1);
Y_test = data_test(:,end);

%% Grid search params
feature_vals = [8, 15, 20, 25];
sc_radi = [0.3, 0.5, 0.8, 1];
k = 5;

errors = zeros(length(feature_vals), length(sc_radi));
rules = zeros(length(sc_radi), 1);

%% Tune hyperparameters with 5-fold CV
for i_f=1:length(feature_vals)
    features = feature_vals(i_f);
    for i_r=1:length(sc_radi)
        radius = sc_radi(i_r);
        c = cvpartition(num_train,'KFold',k);
        cv_errors = zeros(k, 1);
        temp_rules = zeros(k, 1);
        
        fprintf("CV for %d features and %.1f radius starting...\n",...
            features, radius )
        for i=1:k
            %% Get current partition's train and val indices
            train_idx = training(c,i);
            val_idx = test(c,i);
            
            %% And set the current partition's data
            x_train = X_train(train_idx,:);
            y_train = Y_train(train_idx,:);
            
            x_val = X_train(val_idx,:);
            y_val = Y_train(val_idx,:);
            
            %% Find best features in current train data using relieff
            [feat_idx,~] = relieff(x_train, y_train,10);
            feat_idx = feat_idx(1:features);
            
            %% Select only the selected features in the partition's fata
            x_train = x_train(:,feat_idx);
            x_val = x_val(:,feat_idx);
            
            %% Set options and generate FIS
            gen_opt = genfisOptions('SubtractiveClustering',...
                                    'ClusterInfluenceRange',radius);
            tsk_model = genfis(x_train, y_train, gen_opt);

            %% Train the FIS 
            an_opt = anfisOptions("InitialFis", tsk_model, ...
            "ValidationData", [x_val y_val], ...
            "EpochNumber", 100, ...
            "OptimizationMethod", 1);
            
            [~,~,~,valFIS,valError] = ...
                anfis([x_train y_train], an_opt);

            cv_errors(i) = abs(mean(valError));
            temp_rules(i) = size(showrule(valFIS),1);
        end
        errors(i_f, i_r) = mean(cv_errors);
        rules(i_r) = round(mean(temp_rules));
    end
end

%% Plot errors depending on rules and features
figure;
t = tiledlayout(2,2,'TileSpacing','Compact');

for i=1:4
    nexttile
    plot(rules, errors(i,:))
    title(sprintf('Number of Features = %d', feature_vals(i)))
end
title(t,'# of rules vs Error')
xlabel(t,'Rules')
ylabel(t,'Mean Absolute Error')
saveas(gcf,'rules_error.png')

figure;
t = tiledlayout(2,2,'TileSpacing','Compact');

for i=1:4
    nexttile
    plot(feature_vals, errors(:,i))
    title(sprintf('Number of rules = %d', rules(i)))
end
title(t,'# of features vs Error')
xlabel(t,'Features')
ylabel(t,'Mean Absolute Error')
saveas(gcf,'features_error.png')

%% Find optimal hyperparameters
min_error = min(min(errors));
[optimal_feat_idx, optimal_radius_idx]=find(errors==min_error);

optimal_features = feature_vals(optimal_feat_idx);
optimal_radius = sc_radi(optimal_radius_idx);

%% Train optimal model

%% Get top optimal_features
[feat_idx,~] = relieff(X_train, Y_train,10);
feat_idx = feat_idx(1:optimal_features);

%% Select only the selected features in the partition's fata
X_train = X_train(:,feat_idx);
X_val = X_val(:,feat_idx);

%% Set options and generate FIS
gen_opt = genfisOptions('SubtractiveClustering',...
                        'ClusterInfluenceRange',optimal_radius);
tsk_model = genfis(X_train, Y_train, gen_opt);

%% Train the FIS 
an_opt = anfisOptions("InitialFis", tsk_model, ...
"ValidationData", [X_val Y_val], ...
"EpochNumber", 100, ...
"OptimizationMethod", 1);

[fis,trainError,stepSize,valFIS,valError] = ...
    anfis([X_train Y_train], an_opt);


X_test = X_test(:,feat_idx);

y_pred = evalfis(valFIS, X_test);
[rmse, nmse, ndei, r2] = get_metrics(Y_test, y_pred);

%% Plots
figure;
scatter(1:length(Y_test),Y_test, 2)
xlabel("Data Point")
ylabel("Critical Temperature")
legend("Real Value")
saveas(gcf, "real_values.png")

figure;
scatter(1:length(y_pred), y_pred, 2, 'r')
xlabel("Data Point")
ylabel("Critical Temperature")
legend("Predicted Value")
saveas(gcf, "pred_values.png")

figure;
plot([trainError valError]);
xlabel('Iterations');
ylabel('Error');
legend('Training Error', 'Validation Error');
title("Learning Curve, Optimal Model");
saveas(gcf,'learning_curve_optimal.png')

fid = fopen( 'results_optimal.txt', 'wt' );
fprintf(fid, ...
        'Optimal Model, with %d features, %.2f radius (Rules: %d), \n RMSE: %f\n NMSE: %f\n NDEI: %f\n R2:%f\n', ...
        optimal_features, optimal_radius, size(showrule(valFIS),1), rmse, nmse, ndei, r2);
fclose(fid);

figure;
t = tiledlayout(2,2,'TileSpacing','Compact');

nexttile
plotmf(tsk_model,'input',1);
xlabel('First Feature')
title('Before', 'FontSize', 15);

nexttile
plotmf(valFIS,'input',1);
xlabel('First Feature')
title('After', 'FontSize', 15);


nexttile
plotmf(tsk_model,'input',2);
xlabel('Second Feature')
title('Before', 'FontSize', 15);

nexttile
plotmf(valFIS,'input',2);
xlabel('Second Feature')
title('After', 'FontSize', 15);
saveas(gcf,'mf_before_after.png')






