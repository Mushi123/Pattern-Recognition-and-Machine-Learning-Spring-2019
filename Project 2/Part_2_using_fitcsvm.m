clc;
clear;
digits(5);
%rng('default');
rng(100);
n = 100;
C = 100;
class1=mvnrnd([1 3],[1 0; 0 1],60);
class2=mvnrnd([4 1],[2 0; 0 2],40);
Xi = [class1;class2];
yi = [ones(60,1);-1*ones(40,1)];
%%
cl = fitcsvm(Xi,yi,'KernelFunction','rbf',...
    'KernelScale',(2*(1.75)^2),'BoxConstraint',100,'ClassNames',[-1,1]);
