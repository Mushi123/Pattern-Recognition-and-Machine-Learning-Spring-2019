clc;
clear;
c = 2;
m = [zeros(5,1) ones(5,1)];

S(:,:,1)=[0.8 0.2 0.1 0.05 0.01;
0.2 0.7 0.1 0.03 0.02;
0.1 0.1 0.8 0.02 0.01;
0.05 0.03 0.02 0.9 0.01;
0.01 0.02 0.01 0.01 0.8];

S(:,:,2)=[0.9 0.1 0.05 0.02 0.01;
0.1 0.8 0.1 0.02 0.02;
0.05 0.1 0.7 0.02 0.01;
0.02 0.02 0.02 0.6 0.02;
0.01 0.02 0.01 0.02 0.7];

P=[1/2 1/2];
N_list = [100 1000];
%%
for j = 1:length(N_list)
    N_train = N_list(j);
    X_train = [];
    y_train = [];
    rng(0);
    for i=1:c
      t = mvnrnd(m(:,i),S(:,:,i),fix(P(i)*N_train));
      X_train =[X_train ; t];
      y_train=[y_train;ones(fix(P(i)*N_train),1)*i];
    end
    y_train(y_train==2)=-1;

    %%
    rng(100);
    X_test = [];
    y_test = [];
    N_test = 10000;
    for i=1:c
      t = mvnrnd(m(:,i),S(:,:,i),fix(P(i)*N_test));
      X_test =[X_test ; t];
      y_test=[y_test;ones(fix(P(i)*N_test),1)*i];
    end
    y_test(y_test==2)=-1;
    %%
    X_1 = X_train(find(y_train==1),:);
    y_1 = y_train(1:50,:);

    X_2 = X_train(find(y_train == -1),:);
    y_2 = y_train(51:100,:);
    %%
    % NBClf = fitcnb(X_train,y_train,'ClassNames',{'1','-1'});
    % [label,Posterior,Cost] = predict(NBClf,X_test);
    % [p,idx] = max(Posterior');
    % idx(find(idx == 2)) = -1;
    % idx = idx';
    % errors_nb_f = (sum(~(idx == y_test))); % 1248 errors for 100 and 1213 for 1000

    means_1 = mean(X_train(y_train == 1,:));
    variances_1 = var(X_train(y_train == 1,:));
    means_2 = mean(X_train(y_train == -1,:));
    variances_2 = var(X_train(y_train == -1,:));
    preds_NB = [];
    for i = 1:N_test
        x = X_test(i,:);
        p_1 = p_x_given_y(x(1),means_1(1),variances_1(1))*p_x_given_y(x(2),means_1(2),variances_1(2))*p_x_given_y(x(3),means_1(3),variances_1(3))*p_x_given_y(x(4),means_1(4),variances_1(4))*p_x_given_y(x(5),means_1(5),variances_1(5));
        p_2 = p_x_given_y(x(1),means_2(1),variances_2(1))*p_x_given_y(x(2),means_2(2),variances_2(2))*p_x_given_y(x(3),means_2(3),variances_2(3))*p_x_given_y(x(4),means_2(4),variances_2(4))*p_x_given_y(x(5),means_2(5),variances_2(5));
        if(p_1>p_2)
            preds_NB(end+1) = 1;
        else
            preds_NB(end+1) = -1;
        end
    end
    errors_nb = sum(~(preds_NB' == y_test)); %1248 errors for 100 and 1213 for 1000
    disp(['There are ',num2str(errors_nb),' errors for Naive Bayes with N_train = ',num2str(N_train)]);
    %%
    mu_1 = sum(X_1)/length(X_1);
    sigma_1 = (X_1 - mu_1)'*(X_1 - mu_1);
    sigma_1 = sigma_1./length(X_1);
    % sigma_1 = cov(X_1);
    det_s_1 = det(sigma_1);
    f_1 = @(x) (1/((2*pi)^(5/2) * det_s_1^0.5))*exp(-0.5*(x-mu_1)*(sigma_1\(x-mu_1)'));

    mu_2 = sum(X_2)/length(X_2);
    sigma_2 = (X_2 - mu_2)'*(X_2 - mu_2);
    sigma_2 = sigma_2./length(X_2);
    % sigma_2 = cov(X_2);
    det_s_2 = det(sigma_2);
    f_2 = @(x) (1/( ((2*pi)^(5/2)) * det_s_2^0.5))*exp(-0.5*(x-mu_2)*(sigma_2\(x-mu_2)'));

    preds_MLE = [];
    for i = 1:N_test
        x = X_test(i,:);
        p_1 = log(f_1(x))*0.5;
        p_2 = log(f_2(x))*0.5;
        if (p_1>p_2)
            preds_MLE(end+1) = 1;
        else
            preds_MLE(end+1) = -1;
        end    
    end
    errors_mle = sum(~(preds_MLE' == y_test)); %1474 errors for 100 and 1209 for 1000
    disp(['There are ',num2str(errors_mle),' errors for Bayes classifier using MLE with N_train = ',num2str(N_train)]);
    %%
    mu_t_1 = [0 0 0 0 0];
    sigma_t_1 = [0.8 0.2 0.1 0.05 0.01;
    0.2 0.7 0.1 0.03 0.02;
    0.1 0.1 0.8 0.02 0.01;
    0.05 0.03 0.02 0.9 0.01;
    0.01 0.02 0.01 0.01 0.8];
    det_st_1 = det(sigma_t_1);
    f_1_t = @(x) (1/((2*pi)^(5/2) * det_st_1^0.5))*exp(-0.5*(x-mu_t_1)*(sigma_t_1\(x-mu_t_1)'));

    mu_t_2 = [1 1 1 1 1];
    sigma_t_2 = [0.9 0.1 0.05 0.02 0.01;
    0.1 0.8 0.1 0.02 0.02;
    0.05 0.1 0.7 0.02 0.01;
    0.02 0.02 0.02 0.6 0.02;
    0.01 0.02 0.01 0.02 0.7];
    det_st_2 = det(sigma_t_2);
    f_2_t = @(x) (1/( ((2*pi)^(5/2)) * det_st_2^0.5))*exp(-0.5*(x-mu_t_2)*(sigma_t_2\(x-mu_t_2)'));

    preds_t_MLE = [];
    for i = 1:N_test
        x = X_test(i,:);
        p_1 = log(f_1_t(x))*0.5;
        p_2 = log(f_2_t(x))*0.5;
        if (p_1>p_2)
            preds_t_MLE(end+1) = 1;
        else
            preds_t_MLE(end+1) = -1;
        end    
    end
    errors_t = sum(~(preds_t_MLE' == y_test)); % 1171 errors for 100 and 1000

    disp(['There are ',num2str(errors_t),' errors for Bayes classifier using true parameters with N_train = ',num2str(N_train)]);
end
%%
function p = p_x_given_y(x,mu,s)
p = 1/(sqrt(2*pi*s)) * exp((-(x-mu)^2)/(2*s));
end