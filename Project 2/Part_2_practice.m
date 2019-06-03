% reshape(A.',1,[])
% B = magic(3)*100;
% A = repmat(B,3,1)
%%
clc;
clear;
digits(5);
%rng('default');
rng(100);
n = 100;
C = 10;
class1=mvnrnd([1 3],[1 0; 0 1],60);
class2=mvnrnd([4 1],[2 0; 0 2],40);
Xi = [class1;class2];
yi = [ones(60,1);-1*ones(40,1)];
nsq=sum(Xi.^2,2);
K=bsxfun(@minus,nsq,(2*Xi)*Xi.');
K=bsxfun(@plus,nsq.',K);
K=exp(-K/(2*1.75^2));
%%

%%
G = (yi*yi') .* K;
e = ones (n ,1);
% l = eps ^.5;
% G = G + l*eye(n);
tic
ad = quadprog(G,-e,[],[],yi',0,zeros(n,1),C*ones(n,1));
ad = round(ad,3);
%%
d = weight(Xi,yi,ad);
results = [];
sup_vecs = find(ad>0 & ad<C);
final = 0;
syms x1 x2;
for i=1:100
    s = d(x1,x2);    
    x1 = Xi(i,1);
    x2 = Xi(i,2);
    o = vpa(subs(s));
    results(end+1) = o;
end
for j = 1:length(sup_vecs)
    final = final + (yi(sup_vecs(j)) - results(sup_vecs(j)));
end
b = final/length(sup_vecs);
% b = -2.6129;
% if (C == 10)
%     b = -0.4802;
% end
% if (C==10)
%     b = b + 0.1;
% else
%     b = b + 0.07;
% end
results = results + b;


%%
x1_min = min(Xi(:,1));
x1_max = max(Xi(:,1));
x2_min = min(Xi(:,2));
x2_max = max(Xi(:,2));
xx = -2:0.2:8;
yy = -2:0.2:6;
[X,Y] = meshgrid(xx,yy);
sz = size(X);
Z = [[]];
for i = 1:sz(1)
    for j = 1:sz(2)
        s = d(x1,x2);    
        x1 = X(i,j);
        x2 = Y(i,j);
        o = vpa(subs(s));
        Z(i,j) = o + b;        
    end
end
%%
% Need to subtract 0.07 for C = 100 and subtract 0.1 for C = 10. Both using
% only margin vectors
contourf(X,Y,Z,[0 0],'LineStyle','-','Fill','off','DisplayName','Decision Boundary');
hold on;
contourf(X,Y,Z,[1 1],'LineStyle','--','Fill','off','DisplayName','Margin Boundary');
hold on;
contourf(X,Y,Z,[-1 -1],'LineStyle','--','Fill','off','DisplayName','Margin Boundary');
hold on;
scatter(Xi(yi == 1,1),Xi(yi==1,2),50,'r','o','filled','DisplayName','Class 1','LineWidth',1.5,'MarkerEdgeColor',[0.5 0.5 0.5]);
hold on;
scatter(Xi(yi == -1,1),Xi(yi==-1,2),80,'g','square','filled','DisplayName','Class 2','LineWidth',1.5,'MarkerEdgeColor',[0.5 0.5 0.5]);
title(['C = ',num2str(C)]);
hold on;
scatter(Xi(( ad < C) & (ad > 0),1),Xi((ad < C) & (ad > 0),2),60,'.','blue','LineWidth',2,'DisplayName','Margin');
hold on
scatter(Xi(( ad == C) & (ad > 0),1),Xi((ad == C) & (ad > 0),2),60,'x','blue','LineWidth',1.5,'DisplayName','Missclassification/Violation');
hold on
%results = yi'.*results;

xlim([-2 8])
ylim([-2 6])
legend
%%
results = yi'.*results;
scatter(Xi(results < 0,1),Xi(results < 0,2),100,'d','DisplayName','Misclassifications');
title(['C = ',num2str(C),' Sup. Vecs. = ',num2str(sum(ad > 0)),' Number of Misclassifications = ',num2str(length(Xi(results < 0,:)))]);

missclfs = length(Xi(results < 0,:)); % 6 for C = 100, 8 for C = 10
function d = weight(Xi,yi,ad)
    syms x1 x2;    
    x = [x1*ones(100,1) x2*ones(100,1)];
    X_m = Xi-x;
    sq = sum(X_m.^2,2);
    sq_by_sigma = sq./(2*(1.75^2));
    sq_e = exp(-sq_by_sigma);
    result = sq_e.*yi.*ad;
    final = sum(result,1);
    d = @(x1,x2) final;
end

%b = -.4802 for 10
%b = -2.6129 for 100