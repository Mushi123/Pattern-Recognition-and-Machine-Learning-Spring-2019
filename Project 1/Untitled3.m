clc;
clear;
n = 100; % sample size up to 200000 !
rand ('seed' ,2); % fix the randomess
Xi = 4* rand (n ,2) ; % build the training set
q = 0; % add useless variables to see what ... up to 180;
Xi = [Xi 4* rand(n,q)];
[n,p] = size (Xi);
bt = -6; % define the separation line bias
wt = [4 ; -1]; % define the separation line vector
yi = sign (wt (1) * Xi (: ,1) + wt (2) * Xi (: ,2) + bt);

G = (yi*yi') .*( Xi*Xi');
e = ones (n ,1);
l = eps ^.5;
G = G + l*eye(n); % 7) the secret to make it work
tic
ad = quadprog (G,-e ,[] ,[] ,yi',0, zeros (n ,1) ,inf* ones (n ,1) );