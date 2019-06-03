rng(100);
n = 100;
C = .1;
class1=mvnrnd([1 3],[1 0; 0 1],60);
class2=mvnrnd([4 1],[2 0; 0 2],40);
Xi = [class1;class2];
X = Xi./(2*1.75^2);
yi = [ones(60,1);-1*ones(40,1)];
nsq=sum(Xi.^2,2);
K=bsxfun(@minus,nsq,(2*Xi)*Xi.');
K=bsxfun(@plus,nsq.',K);
K=exp(-K/(2*1.75^2));
% nms = sum(Xi.^2,2);
% Ks = exp(-(nms'.*ones(1,n) -ones(n,1).*nms + 2*(Xi*Xi'))/(2*1.75^2));
% H = produce_kernel_matrix(Xi,Xi,1/(2*1.75^2));
% G = kernelmatrix(Xi,Xi,1.75);
% function [ Kern ] = produce_kernel_matrix( X, t, beta )
% %
% X = X';
% t = t';
% X_T_2 = sum(X.^2,2) + sum(t.^2,2).' - (2*X)*t.'; % ||x||^2 + ||t||^2 - 2<x,t>
% Kern =exp(-beta*X_T_2); %
% end
function K = kernelmatrix(X,X2,sigma)

        n1sq = sum(X.^2,1);
        n1 = size(X,2);

        if isempty(X2)
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
        else
            n2sq = sum(X2.^2,1);
            n2 = size(X2,2);
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
        end
        K = exp(-D/(2*sigma^2));
end