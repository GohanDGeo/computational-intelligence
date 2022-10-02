% Koutroumpis Georgios, AEM 9668
% COMPUTATIONAL INTELLIGENCE
% ECE AUTh 2022
% Project 2, TSK

% Function that plots the graphs required by the project
% @args:
% init_fis      => initiali FIS
% trained_fis   => FIS after training
% Y_test        => y values of test dataset
% y_pred        => y values predicted from model
% trainError    => training error of model
% valError      => error on validation data
% i             => model number
function plot_requirements(init_fis, trained_fis,...
                            Y_test, y_pred, trainError, valError, i)
    
    % Plot MFs before training
    plot_mf_tsk(init_fis, sprintf("MF before training, model %d",i))
    saveas(gcf,sprintf('mf_before_model_%d.png',i))
    
    % Plot MFs after training
    plot_mf_tsk(trained_fis, sprintf("MF after training, model %d",i))
    saveas(gcf,sprintf('mf_after_model_%d.png',i))

    % Plot training curve
    figure;
    plot([trainError valError]);
    xlabel('Iterations');
    ylabel('Error');
    legend('Training Error', 'Validation Error');
    title(sprintf("Learning Curve, Model %d",i));
    saveas(gcf,sprintf('learning_curve_%d.png',i))
    
    % Plot prediction error
    pred_error = Y_test - y_pred;
    
    figure;
    plot(pred_error);
    xlabel('Testing Data');
    ylabel('Prediction Error');
    title(sprintf("Prediction Error, Model %d",i));
    saveas(gcf,sprintf('prediction_error_model_%d.png',i))
end

% Function that plots the MFs for the airfoil_self_noise dataset
% @args:
% fis  => the FIS to plot MFs for
% name => name to give graph
function plot_mf_tsk(fis, name)
        figure;
        subplot(2,3,1)
        plotmf(fis,'input',1);
        xlabel('Frequency (Hz)')
        title('1', 'FontSize', 15);
        
        subplot(2,3,2)
        plotmf(fis,'input',2);
        xlabel('Angle of attack (Deg)')
        title('2', 'FontSize', 15);
        
        subplot(2,3,3)
        plotmf(fis,'input',3);
        xlabel('Chord length (m)')
        title('3', 'FontSize', 15);
        
        subplot(2,3,4)
        plotmf(fis,'input',4);
        xlabel('Free-stream velocity (m/s)')
        title('4', 'FontSize', 15);
        
        subplot(2,3,6)
        plotmf(fis,'input',5);
        xlabel('Suction side displacement thickness (m)')
        title('5', 'FontSize', 15);
        
        sgt = sgtitle(name);
        sgt.FontSize = 20;
end