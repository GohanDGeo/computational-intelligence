% Koutroumpis Georgios, AEM 9668
% COMPUTATIONAL INTELLIGENCE
% ECE AUTh 2022
% Project 2, TSK

% Function that calculatres the RMSE, NMSE, NDEI, R^2 metrics
% @args:
% y      => true values
% y_pred => predicted values
function [rmse, nmse, ndei, r2] = get_metrics(y, y_pred)

    rmse = sqrt(mse(y, y_pred));
    
    ss_res = sum((y - y_pred).^2);
    ss_tot = sum((y - mean(y)).^2);
    
    r2 = 1 - ss_res/ss_tot;
    nmse = ss_res/ss_tot;
    ndei = sqrt(nmse);

end