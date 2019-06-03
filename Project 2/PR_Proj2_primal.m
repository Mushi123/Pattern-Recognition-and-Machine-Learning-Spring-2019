clc;
clear;
rng('default');
n = 100;
p = 2;
class1=mvnrnd([1 3],[1 0; 0 1],60);
class2=mvnrnd([4 1],[2 0; 0 2],40);
Xi = [class1;class2];
yi = [ones(60,1);-1*ones(40,1)];
H = [eye(p)];
H(p+1,p+1) = 0;
f = zeros (p+1 ,1);
A = -[ diag(yi)*Xi yi ];
bb = -ones (n ,1) ;
x = quadprog (H,f,A,bb);
