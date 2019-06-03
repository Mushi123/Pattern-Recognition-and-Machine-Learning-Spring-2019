n = 100; % sample size up to 200000 !
rand ('seed' ,2); % fix the randomess
Xi = 4* rand (n ,2) ; % build the training set
q = 0; % add useless variables to see what ... up to 180;
Xi = [Xi  4*rand(n,q)];
[n,p] = size (Xi);
bt = -6; % define the separation line bias
wt = [4 ; -1]; % define the separation line vector
yi = sign(wt(1)*Xi(: ,1) + wt(2)*Xi(: ,2) + bt);

H = [eye(2)];
H(p+1,p+1) = 0;
f = zeros (p+1 ,1);
A = -[ diag(yi)*Xi yi ];
bb = -ones(n ,1) ;